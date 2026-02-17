from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class FusionOutput:
    logits: torch.Tensor
    gate: torch.Tensor | None = None  # used only when fusion_mode == "gated"


class GatedFusionClassifier(nn.Module):
    def __init__(
        self,
        pretrained_name="roberta-base",
        num_labels=2,
        fusion_mode="gated",
        dropout=0.1,
    ):
        super().__init__()

        # one transformer encoder shared by both text and emoji inputs
        self.encoder = AutoModel.from_pretrained(pretrained_name)
        self.hidden = self.encoder.config.hidden_size

        self.fusion_mode = fusion_mode
        self.dropout = nn.Dropout(dropout)

        # for gated fusion: takes [text_vec ; emoji_vec] and outputs one number
        self.gate_layer = nn.Linear(self.hidden * 2, 1)

        # classifier input size depends on fusion mode
        if fusion_mode == "concat":
            in_dim = self.hidden * 2
        else:
            in_dim = self.hidden

        self.classifier = nn.Linear(in_dim, num_labels)

    @staticmethod
    def pool(last_hidden_state):
        # take the first token representation as a pooled vector (RoBERTa style)
        return last_hidden_state[:, 0, :]

    def encode(self, inputs):
        out = self.encoder(**inputs)
        return self.pool(out.last_hidden_state)

    def forward(self, text_inputs, emoji_inputs):
        h_text = self.encode(text_inputs)
        h_emoji = self.encode(emoji_inputs)

        gate = None

        if self.fusion_mode == "gated":
            both = torch.cat([h_text, h_emoji], dim=-1)
            g = torch.sigmoid(self.gate_layer(both))  # (B, 1)
            h = g * h_text + (1.0 - g) * h_emoji
            gate = g

        elif self.fusion_mode == "concat":
            h = torch.cat([h_text, h_emoji], dim=-1)

        elif self.fusion_mode == "text_only":
            h = h_text

        elif self.fusion_mode == "emoji_only":
            h = h_emoji

        else:
            raise ValueError("Unknown fusion_mode: " + str(self.fusion_mode))

        h = self.dropout(h)
        logits = self.classifier(h)

        return FusionOutput(logits=logits, gate=gate)
