import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from typing import Dict, Optional, Tuple, List

from encoder import TextEncoder, ImageEncoder


class CrossModalAttention(nn.Module):
    """
    Cross-modal transformer for fusing text, image, and rationale embeddings.
    """
    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        # Positional embeddings (learned)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        tokens: [B, T, D]
        attention_mask: [B, T]
        returns: [B, D] (fused CLS)
        """
        B, T, D = tokens.size()
        # prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)             # [B, T+1, D]
        attn = torch.cat([torch.ones(B, 1, device=x.device), attention_mask], dim=1)
        # apply transformer
        out = self.transformer(
            x,
            src_key_padding_mask=(attn == 0)
        )
        # return CLS output
        return out[:, 0]


class ViLID(nn.Module):
    """
    Vision-Language Inconsistency Detector.
    Uses TextEncoder and ImageEncoder for feature extraction,
    computes inconsistency scores, and fuses via CrossModalAttention.
    """
    def __init__(
        self,
        text_model_name: str = "openai/clip-vit-base-patch32",
        image_model_name: str = "openai/clip-vit-base-patch32",
        hidden_size: int = 512,
        num_fusion_layers: int = 4,
        num_fusion_heads: int = 8,
        dropout: float = 0.1,
        gamma: float = 0.1,
        beta: float = 5.0,
        freeze_encoders: bool = False
    ):
        super().__init__()
        # Embed & encode
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            pooling_strategy="cls"
        )
        self.image_encoder = ImageEncoder(
            model_name=image_model_name,
            regional_features=True,
            pooling_strategy="mean"
        )
        # Optionally freeze
        if freeze_encoders:
            for p in self.text_encoder.parameters(): p.requires_grad = False
            for p in self.image_encoder.parameters(): p.requires_grad = False

        # Fusion
        self.fusion = CrossModalAttention(
            hidden_size=hidden_size,
            num_layers=num_fusion_layers,
            num_heads=num_fusion_heads,
            dropout=dropout
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

        # Hyperparams
        self.gamma = gamma
        self.beta = beta

    def forward(
        self,
        texts: List[str],
        images: torch.Tensor,
        text_rationales: List[str],
        image_rationales: List[str]
    ) -> Dict[str, torch.Tensor]:
        # Encode
        text_emb = self.text_encoder(texts)             # [B, D]
        image_emb = self.image_encoder(images)           # [B, D]
        tr_emb = self.text_encoder(text_rationales)      # [B, D]
        ir_emb = self.text_encoder(image_rationales)     # [B, D]

        # Compute inconsistency
        t_n = F.normalize(text_emb, dim=1)
        i_n = F.normalize(image_emb, dim=1)
        S_inc = 1 - torch.sum(t_n * i_n, dim=1)
        # rationale S_r
        tr_n = F.normalize(tr_emb, dim=1)
        ir_n = F.normalize(ir_emb, dim=1)
        S_r = 1 - torch.sum(tr_n * ir_n, dim=1)

        # Alignment alpha via sigmoid(-beta * S_inc)
        alpha = torch.sigmoid(-self.beta * S_inc)

        # Fuse for classification
        # combine all features into one sequence
        seq = torch.stack([text_emb, image_emb, tr_emb, ir_emb], dim=1)  # [B,4,D]
        mask = torch.ones(seq.size(0), seq.size(1), device=seq.device)
        fused = self.fusion(seq, mask)  # [B, D]

        # Classifier input
        inp = torch.cat([fused, S_inc.unsqueeze(1), S_r.unsqueeze(1)], dim=1)
        logits = self.classifier(inp).squeeze(-1)
        y_pred = torch.sigmoid(logits)

        return {"y_pred": y_pred, "logits": logits,
                "S_inc": S_inc, "S_r": S_r, "alpha": alpha}


class ViLIDLoss(nn.Module):
    """
    Combined BCE loss with alignment regularization.
    """
    def __init__(self, gamma: float = 0.1, weight_decay: float = 0.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.weight_decay = weight_decay

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        S_inc: torch.Tensor,
        S_r: torch.Tensor,
        params: Optional[List[nn.Parameter]] = None
    ) -> Dict[str, torch.Tensor]:
        bce = self.bce(logits, labels.float())
        mask = (labels == 0)
        if mask.any():
            align = self.gamma * (S_inc[mask].mean() + S_r[mask].mean())
        else:
            align = logits.new_zeros(1)

        wd = torch.tensor(0., device=logits.device)
        if params and self.weight_decay > 0:
            wd = self.weight_decay * sum(p.pow(2).sum() for p in params)

        loss = bce + align + wd
        return {"total": loss, "bce": bce, "align": align, "wd": wd}
