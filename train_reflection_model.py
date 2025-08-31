import argparse
import logging
import math
import os
import shutil

import accelerate
import datasets
import diffusers
import torch
import transformers
import webdataset as wds
import io
import json

from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, ControlNetModel, EMAModel, get_scheduler, \
    StableDiffusionControlNetPipeline
from packaging import version
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from transformers.utils import ContextManagers

logger = get_logger(__name__, log_level="INFO")

DATASETS = {
    "CXR": "itsanmolgupta/mimic-cxr-dataset",
}

def _bytes_to_pil(x):
    if isinstance(x, bytes):
        return Image.open(io.BytesIO(x)).convert("RGB")
    if isinstance(x, Image.Image):
        return x
    if torch.is_tensor(x):
        return T.ToPILImage()(x)
    raise TypeError(f"unexpected image type: {type(x)}")

def load_img(x):
    return _bytes_to_pil(x)

def load_txt(x):
    return x if isinstance(x, str) else x.decode("utf-8")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="./")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default='./controlnet')
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--modality", type=str, default="CXR", choices=["CXR"])
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--resume_from_checkpoint", type=str, default='latest')

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=10,
        help=("Max number of checkpoints to store."),
    )

    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )

    return parser.parse_args()

def main():
    args = parse_args()
    pattern = "/data/sora/Medical_Reflection_Generation/out/ds_huatuo-{000000..000010}.tar"

    dataset = (
        wds.WebDataset(
            pattern,
            nodesplitter=wds.split_by_node,
            shardshuffle=False
        )   # 显式关掉分片shuffle
        # 不要调用 .decode(...)，避免触发内置的 json 解码器
        .to_tuple("real.png", "gen.png", "prompt.txt", "huatuo.json", "__key__")
        .map_tuple(
            load_img,          # real.png  -> PIL.Image
            load_img,          # gen.png   -> PIL.Image
            load_txt,          # prompt.txt-> str
            load_txt,          # huatuo.json(其实是纯文本)-> str
            lambda x: x        # __key__   -> str
        )
    )

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )



    accelerator = Accelerator()
    device = accelerator.device
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model, subfolder="text_encoder", revision=args.revision
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model, subfolder="vae", revision=args.revision
        )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model, subfolder="unet", revision=args.non_ema_revision
    )

    controlnet = ControlNetModel.from_unet(unet)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()

    if args.use_ema:
        ema_controlnet = ControlNetModel.from_unet(unet)
        ema_controlnet = EMAModel(ema_controlnet.parameters(), model_cls=ControlNetModel, model_config=ema_controlnet.config)
        ema_controlnet.to(device)

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_controlnet.save_pretrained(os.path.join(output_dir, "ema_controlnet"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "controlnet"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_controlnet"), ControlNetModel)
                ema_controlnet.load_state_dict(load_model.state_dict())
                ema_controlnet.to(device)
                del load_model

            for i in range(len(models)):
                model = models.pop()

                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    optimizer = AdamW(
        controlnet.parameters(),
        lr = args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )

    def collate_fn(batch):
        reals, gens, prompts, huatuos, keys = zip(*batch)
        reals = [train_transforms(real) for real in reals]
        gens = [train_transforms(gen) for gen in gens]
        reflections = tokenizer(
            list(huatuos),
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "real": torch.stack(reals, dim=0),   # [B,C,H,W]
            "gen":  torch.stack(gens,  dim=0),   # [B,C,H,W]
            "prompt": list(prompts),             # List[str]
            "reflection": reflections.input_ids,
            "key": list(keys),                   # List[str]
        }

    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)

    dataset_length = 0
    for batch in dataloader:
        dataset_length += len(batch)
    print("Total samples in dataloader:", dataset_length)

    num_update_steps_per_epoch = math.ceil(dataset_length)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500 * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    controlnet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, dataloader, lr_scheduler
    )

    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    unet.to(accelerator.device)

    total_batch_size = batch_size * accelerator.num_processes

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {dataset_length}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != 'latest':
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x : int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        inital=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process
    )

    for epoch in range(first_epoch, initial_global_step):
        train_loss = 0.0
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(controlnet):
                real_img = batch["real"]          # Tensor [B,C,H,W]
                gen_img  = batch["gen"]           # Tensor [B,C,H,W]
                prompt_txt = batch["prompt"]      # List[str]，长度 B
                reflection = batch["reflection"]
                key = batch["key"]                # List[str]

                with torch.no_grad():
                    latents = vae.encode(real_img).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device, dtype=torch.long
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(reflection)[0]

                down_res, mide_res = controlnet(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=gen_img, conditioning_scale=1.0, return_dict=False
                )

                pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residuals=mide_res
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.pprediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(pred, target)

                avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()
                train_loss += avg_loss.item()

                accelerator.backward(loss)

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_controlnet.step(controlnet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) > args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save(save_path)
                        logger.info(f"Checkpoint saved to {save_path}")
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            controlnet = accelerator.unwrap_model(controlnet)
            if args.use_ema:
                ema_controlnet.copy_to(controlnet.parameters())

            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                args.pretrained_model_name,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=controlnet,
                scheduler=noise_scheduler,
                safety_checker=None,

            )
            pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()

if __name__ == "__main__":
    main()