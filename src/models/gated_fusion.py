from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class FusionOutput:
    logits: torch.Tensor
    gate: Optional[torch.Tensor] = None  # (B,1) when fusion_mode="gated"


class GatedFusionClassifier(nn.Module):
    """
    Two-stream sarcasm classifier:
      - text stream encodes x_text
      - emoji stream encodes x_emoji
    Shared transformer encoder weights (compact + consistent).

    fusion_mode:
      - "gated":  h = g*h_text + (1-g)*h_emoji
      - "concat": h = [h_text; h_emoji]
      - "text_only": h = h_text
      - "emoji_only": h = h_emoji
    """

    def __init__(
        self,
        pretrained_name: str = "roberta-base",
        num_labels: int = 2,
        fusion_mode: str = "gated",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_name)
        self.hidden = self.encoder.config.hidden_size

        self.fusion_mode = fusion_mode
        self.dropout = nn.Dropout(dropout)

        # scalar gate in [0,1]
        self.gate_layer = nn.Linear(self.hidden * 2, 1)

        # classifier input size depends on mode
        in_dim = self.hidden * 2 if fusion_mode == "concat" else self.hidden
        self.classifier = nn.Linear(in_dim, num_labels)

    @staticmethod
    def pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
        # RoBERTa: first token (<s>) works as pooled representation
        return last_hidden_state[:, 0, :]

    def encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.encoder(**inputs)
        return self.pool(out.last_hidden_state)

    def forward(
        self,
        text_inputs: Dict[str, torch.Tensor],
        emoji_inputs: Dict[str, torch.Tensor],
    ) -> FusionOutput:
        h_text = self.encode(text_inputs)
        h_emoji = self.encode(emoji_inputs)

        gate = None
        if self.fusion_mode == "gated":
            g = torch.sigmoid(self.gate_layer(torch.cat([h_text, h_emoji], dim=-1)))  # (B,1)
            h = g * h_text + (1.0 - g) * h_emoji
            gate = g
        elif self.fusion_mode == "concat":
            h = torch.cat([h_text, h_emoji], dim=-1)
        elif self.fusion_mode == "text_only":
            h = h_text
        elif self.fusion_mode == "emoji_only":
            h = h_emoji
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        h = self.dropout(h)
        logits = self.classifier(h)
        return FusionOutput(logits=logits, gate=gate)
