import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from torchvision import models, transforms
from pytorch_msssim import ms_ssim
from scipy import linalg
import numpy as np


class Metrics:

    def __init__(self, device):
        self.acts_r = []
        self.acts_g = []
        self.inception_preds = []
        self.total_ms_ssim_scores = 0.
        self.total_images = 0
        self.device = device
        self.model = models.inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()
        self.up = transforms.Resize((299, 299))
        self.feature = None

        def hook_fn(module, input, output):
            self.feature = output

        self.model.Mixed_7c.register_forward_hook(hook_fn)


    def add_images(self, real_imgs, pred_imgs):
        batch_acts_r, _ = self._get_activations(real_imgs)
        batch_acts_g, batch_pred = self._get_activations(pred_imgs)
        self.acts_r.append(batch_acts_r)
        self.acts_g.append(batch_acts_g)
        self.inception_preds.append(F.softmax(batch_pred, dim=1).cpu().numpy())
        self.total_ms_ssim_scores += self._compute_ms_ssim(real_imgs, pred_imgs, device=self.device)
        self.total_images += len(pred_imgs)


    def _get_activations(self, images):
        with torch.no_grad():
            batch = images.to(self.device)
            batch = self.up(batch)
            pred = self.model(batch)
            # use pool3 features instead of logits
            feat = self.feature
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)
            return feat.cpu().numpy(), pred
        # return np.concatenate(acts, axis=0)


    # FID
    def compute_fid(self):
        """Compute FID between two sets of images (tensors in [0,1], shape N,C,H,W)."""

        act_r = np.concatenate(self.acts_r, axis=0)
        act_g = np.concatenate(self.acts_g, axis=0)
        mu_r, sigma_r = act_r.mean(axis=0), np.cov(act_r, rowvar=False)
        mu_g, sigma_g = act_g.mean(axis=0), np.cov(act_g, rowvar=False)

        diff = mu_r - mu_g
        covmean, _ = linalg.sqrtm(sigma_r.dot(sigma_g), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff.dot(diff) + np.trace(sigma_r + sigma_g - 2 * covmean)
        return float(fid)

    # Inception Score
    def compute_inception_score(self, splits=10):
        """Compute Inception Score for a set of images."""
        preds = np.concatenate(self.inception_preds, axis=0)
        scores = []
        N = preds.shape[0]
        split_size = N // splits
        for k in range(splits):
            part = preds[k * split_size:(k + 1) * split_size]
            py = np.mean(part, axis=0, keepdims=True)
            kl = part * (np.log(part + 1e-16) - np.log(py + 1e-16))
            scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
        return float(np.mean(scores)), float(np.std(scores))

    def compute_ms_ssim(self):
        return self.total_ms_ssim_scores / self.total_images

    # MS-SSIM
    @staticmethod
    def _compute_ms_ssim(imgs1, imgs2, data_range=1.0, device='cuda'):
        """
        Compute MS-SSIM between two batches of images (tensors in [0,1], shape N, C, H, W).
        Requires: pip install pytorch-msssim
        """
        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)
        return float(ms_ssim(imgs1, imgs2, data_range=data_range, size_average=False).sum().item())

    def __str__(self):
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.add_row(["FID", f"{self.compute_fid():.4f}"])
        mean, std = self.compute_inception_score()
        table.add_row(["Inception Score", f"{mean:.4f} ± {std:.4f}"])
        ms_ssim = self.compute_ms_ssim()
        table.add_row(["MS-SSIM", f"{ms_ssim:.4f}"])
        return table.get_string()

