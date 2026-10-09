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


class _QuickGELU(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class _CLIPMlp(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.c_fc = torch.nn.Linear(width, width * 4)
        self.gelu = _QuickGELU()
        self.c_proj = torch.nn.Linear(width * 4, width)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class _CLIPVisualBlock(torch.nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(width)
        self.attn = torch.nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln_2 = torch.nn.LayerNorm(width)
        self.mlp = _CLIPMlp(width)

    def forward(self, x):
        y = self.ln_1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class _CLIPVisualViT(torch.nn.Module):
    """The openai CLIP ViT-B/32 image tower, matching `visual.*` checkpoint keys."""

    def __init__(self, width=768, layers=12, heads=12, out_dim=512, grid=7):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, width, 32, 32, bias=False)
        self.class_embedding = torch.nn.Parameter(torch.zeros(width))
        self.positional_embedding = torch.nn.Parameter(torch.zeros(grid * grid + 1, width))
        self.ln_pre = torch.nn.LayerNorm(width)
        self.transformer = torch.nn.ModuleDict({
            "resblocks": torch.nn.ModuleList([_CLIPVisualBlock(width, heads) for _ in range(layers)])
        })
        self.ln_post = torch.nn.LayerNorm(width)
        self.proj = torch.nn.Parameter(torch.zeros(width, out_dim))

    def forward(self, x):
        x = self.conv1(x).flatten(2).permute(0, 2, 1)
        x = torch.cat([self.class_embedding.expand(x.shape[0], 1, -1), x], dim=1)
        x = self.ln_pre(x + self.positional_embedding)
        for block in self.transformer["resblocks"]:
            x = block(x)
        return self.ln_post(x[:, 0]) @ self.proj


class FontCLIP(Embedder):
    """FontCLIP (Tatsukawa et al. 2024, MIT): CLIP ViT-B/32 fine-tuned on font
    attribute data. Checkpoint: model_checkpoints/model.pt from the official
    repo (drive link in setup_data.py), cached under .cache/fontclip/."""

    name = "fontclip"
    dim = 512

    def __init__(self):
        from pathlib import Path
        ckpt = Path(__file__).resolve().parent / ".cache" / "fontclip" / "model.pt"
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = {k[len("visual."):]: v for k, v in obj["model_state_dict"].items() if k.startswith("visual.")}
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = _CLIPVisualViT()
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"fontclip visual tower missing keys: {missing}")
        self.model = self.model.eval().to(self.device)
        self._mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        self._std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed_batch(self, images):
        arr = np.stack([im.astype(np.float32) / 255.0 for im in images])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float()
        x = ((x - self._mean) / self._std).to(self.device)
        return self.model(x).cpu().numpy()

