import os.path

from diffusers import StableDiffusionPipeline, UNet2DConditionModel


class MINIM:
    MODAL_IDX = {
        'OCT': 1,
        'CXR': 2,
        'Fundus': 3,
        'BrainMRI': 4,
        'BreastMRI': 4,
        'ChestCT': 4
    }

    def __init__(self, modal, model_path, device):
        self.modal = modal
        modal_id = f'''{self.MODAL_IDX[modal]}'''
        unet = UNet2DConditionModel.from_pretrained(os.path.join(model_path, 'unets', modal_id, 'unet'))
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            unet=unet,
            safety_checker=None,
        ).to(device)

    def __call__(self, prompt, num_inference_steps):
        input_to_model = f'''{self.modal}:{prompt}'''
        image = self.pipe(
            prompt=input_to_model,
            num_inference_steps=num_inference_steps
        ).images[0]
        return image


def build_model(modal, model_path, model, device):
    if model == 'minim':
        return MINIM(modal, model_path, device)
    else:
        raise NotImplementedError

