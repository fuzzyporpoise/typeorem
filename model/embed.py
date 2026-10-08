import numpy as np
import torch
from torchvision.models import vgg16, VGG16_Weights

CAFFE_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(rgb, mode):
    x = rgb.astype(np.float32)
    if mode == "caffe":
        return x[..., ::-1] - CAFFE_MEAN
    if mode == "rgb_mean":
        return x - CAFFE_MEAN
    if mode == "imagenet":
        return (x / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    if mode == "raw255":
        return x
    if mode == "raw01":
        return x / 255.0
    raise ValueError(f"unknown mode {mode}")


class Embedder:
    name = "base"
    dim = 0

    def embed_batch(self, images):
        raise NotImplementedError


class VGG16Pool5(Embedder):
    name = "vgg16_pool5"
    dim = 25088

    def __init__(self, mode="raw01"):
        self.mode = mode
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = model.features.eval().to(self.device)
        for p in self.features.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def embed_batch(self, images):
        arr = np.stack([preprocess(im, self.mode) for im in images])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float().to(self.device)
        return self.features(x).flatten(1).cpu().numpy()


class DINOv2(Embedder):
    name = "dinov2_vits14"
    dim = 384

    def __init__(self):
        import timm
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
        self.model = self.model.eval().to(self.device)
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed_batch(self, images):
        arr = np.stack([im.astype(np.float32) / 255.0 for im in images])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float()
        x = ((x - self._mean) / self._std).to(self.device)
        return self.model(x).cpu().numpy()

