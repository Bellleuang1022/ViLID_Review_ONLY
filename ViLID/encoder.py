import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from transformers import CLIPProcessor, CLIPTextModel, CLIPVisionModel, AutoConfig
import logging
from PIL import Image

logger = logging.getLogger(__name__)


def _get_projection_dim(model_name: str) -> int:
    """
    Determines a shared projection dimension from the model config.
    Tries projection_dim, then text_config.hidden_size, then config.hidden_size.
    """
    cfg = AutoConfig.from_pretrained(model_name)
    # Primary: projection_dim (e.g., for CLIP)
    if getattr(cfg, 'projection_dim', None) is not None:
        return cfg.projection_dim
    # Fallback: CLIP text hidden size
    text_cfg = getattr(cfg, 'text_config', None)
    if getattr(text_cfg, 'hidden_size', None) is not None:
        return text_cfg.hidden_size
    # Fallback: top-level hidden_size
    if getattr(cfg, 'hidden_size', None) is not None:
        return cfg.hidden_size
    raise ValueError(f"Cannot determine projection dimension for {model_name}")


class TextEncoder(nn.Module):
    """
    Text encoder using CLIP's text model with configurable pooling and optional projection.
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        pooling_strategy: str = "cls",
        projection_dim: int = None
    ):
        super().__init__()
        self.pooling_strategy = pooling_strategy
        # Load CLIP text model and processor
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Determine projection dimension
        target_dim = projection_dim or _get_projection_dim(model_name)
        in_dim = self.text_model.config.hidden_size
        self.proj = nn.Linear(in_dim, target_dim) if in_dim != target_dim else None

        logger.info(f"TextEncoder initialized: in_dim={in_dim}, proj_dim={target_dim}, pooling={pooling_strategy}")

    def forward(self, texts: list[str]) -> torch.Tensor:
        enc = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        device = next(self.text_model.parameters()).device
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        out = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hs = out.last_hidden_state  # [B, L, D]

        if self.pooling_strategy == "cls":
            pooled = hs[:, 0, :]
        elif self.pooling_strategy == "mean":
            mask = attention_mask.unsqueeze(-1)
            summed = (hs * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = summed / lengths
        else:
            raise ValueError(f"Unknown pooling: {self.pooling_strategy}")

        return self.proj(pooled) if self.proj is not None else pooled


class ImageEncoder(nn.Module):
    """
    Image encoder using CLIP's vision model or optional regional features,
    with optional projection to a shared dimension.
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        regional_features: bool = False,
        pooling_strategy: str = "mean",
        projection_dim: int = None
    ):
        super().__init__()
        self.regional_features = regional_features
        self.pooling_strategy = pooling_strategy
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        target_dim = projection_dim or _get_projection_dim(model_name)
        in_dim = self.vision_model.config.hidden_size
        self.proj = nn.Linear(in_dim, target_dim) if in_dim != target_dim else None

        if regional_features:
            self.region_detector = fasterrcnn_resnet50_fpn(pretrained=True)
            self.region_detector.eval()
            if pooling_strategy == "attention":
                self.attn_pool = nn.MultiheadAttention(
                    embed_dim=target_dim,
                    num_heads=8,
                    batch_first=True
                )
                self.query = nn.Parameter(torch.randn(1, 1, target_dim))

        logger.info(
            f"ImageEncoder initialized: in_dim={in_dim}, proj_dim={target_dim}, "+
            f"regional={regional_features}, pooling={pooling_strategy}"
        )

    def forward(self, images: torch.Tensor | list[Image.Image]) -> torch.Tensor:
        device = next(self.vision_model.parameters()).device
        if isinstance(images, torch.Tensor):
            pix = images.to(device)
            out = self.vision_model(pixel_values=pix, return_dict=True)
            emb = out.pooler_output
        else:
            if not self.regional_features:
                inputs = self.processor(images=images, return_tensors="pt")
                pix = inputs.pixel_values.to(device)
                out = self.vision_model(pixel_values=pix, return_dict=True)
                emb = out.pooler_output
            else:
                proposals = self._extract_regions(images)
                emb = self._extract_region_feats(images, proposals)

        return self.proj(emb) if self.proj is not None else emb


