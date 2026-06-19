"""SigLIP image encoder + MiniLM text encoder + trained MLP head classifier backend."""

from __future__ import annotations

import time
from typing import Protocol

import torch
from torch import nn

from .base_model import BaseVisionModel, PredictionResult

IMAGE_EMBED_DIM = 1152
TEXT_EMBED_DIM = 384


class ClassifierHead(nn.Module):
    """Trainable MLP fusing a frozen image embedding and a frozen text embedding."""

    def __init__(
        self,
        image_dim: int = IMAGE_EMBED_DIM,
        text_dim: int = TEXT_EMBED_DIM,
        hidden_dims: tuple[int, int] = (512, 128),
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(image_dim + text_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, image_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([image_embed, text_embed], dim=-1)
        return self.net(fused).squeeze(-1)


class CrossAttentionHead(nn.Module):
    """Fuses a frozen image embedding and a frozen text embedding via bidirectional
    cross-attention instead of plain concatenation. Each embedding is projected into a
    shared space and treated as a length-1 sequence; text attends over the image and
    image attends over the text (each with a residual + LayerNorm, transformer-block
    style) before the two attended vectors are concatenated and classified."""

    def __init__(
        self,
        image_dim: int = IMAGE_EMBED_DIM,
        text_dim: int = TEXT_EMBED_DIM,
        proj_dim: int = 256,
        num_heads: int = 4,
        hidden_dims: tuple[int, int] = (256, 128),
    ) -> None:
        super().__init__()
        self.image_proj = nn.Linear(image_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.text_to_image_attn = nn.MultiheadAttention(proj_dim, num_heads, batch_first=True)
        self.image_to_text_attn = nn.MultiheadAttention(proj_dim, num_heads, batch_first=True)
        self.norm_text = nn.LayerNorm(proj_dim)
        self.norm_image = nn.LayerNorm(proj_dim)
        h1, h2 = hidden_dims
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, image_embed: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        img = self.image_proj(image_embed).unsqueeze(1)
        txt = self.text_proj(text_embed).unsqueeze(1)

        text_attended, _ = self.text_to_image_attn(query=txt, key=img, value=img)
        image_attended, _ = self.image_to_text_attn(query=img, key=txt, value=txt)

        text_fused = self.norm_text(txt + text_attended).squeeze(1)
        image_fused = self.norm_image(img + image_attended).squeeze(1)

        fused = torch.cat([text_fused, image_fused], dim=-1)
        return self.classifier(fused).squeeze(-1)


HEAD_TYPES: dict[str, type[nn.Module]] = {
    "concat": ClassifierHead,
    "cross_attention": CrossAttentionHead,
}


class ImageTextEncoder(Protocol):
    def encode_image(self, image_path: str) -> torch.Tensor: ...
    def encode_text(self, text: str) -> torch.Tensor: ...


class SiglipClassifierModel(BaseVisionModel):
    """BaseVisionModel backend: frozen image/text encoders + trained MLP head."""

    def __init__(
        self,
        encoders: ImageTextEncoder,
        head: ClassifierHead,
        model_id: str = "siglip_classifier",
        threshold: float = 0.5,
    ) -> None:
        self._encoders = encoders
        self._head = head
        self._head.eval()
        self._model_id = model_id
        self._threshold = threshold

    def name(self) -> str:
        return self._model_id

    def predict_unanswerable(
        self,
        document_path: str,
        question: str,
        prompt_template: str,
        page_indices: list[int] | None = None,
    ) -> PredictionResult:
        t0 = time.perf_counter()
        try:
            image_embed = self._encoders.encode_image(document_path)
            text_embed = self._encoders.encode_text(question)
            with torch.no_grad():
                logit = self._head(image_embed.unsqueeze(0), text_embed.unsqueeze(0))
                prob = torch.sigmoid(logit).item()
        except Exception as exc:
            return PredictionResult(
                sample_id="",
                predicted_unanswerable=False,
                confidence=-1.0,
                raw_response=f"error: {exc}",
                inference_time_s=time.perf_counter() - t0,
                skipped=True,
            )
        return PredictionResult(
            sample_id="",
            predicted_unanswerable=prob >= self._threshold,
            confidence=prob,
            raw_response=f"p_unanswerable={prob:.4f}",
            inference_time_s=time.perf_counter() - t0,
            skipped=False,
        )

    @classmethod
    def from_pretrained(
        cls,
        head_checkpoint_path: str,
        siglip_model_id: str = "google/siglip-so400m-patch14-384",
        minilm_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        head_type: str = "concat",
        model_id: str = "siglip_classifier",
        threshold: float = 0.5,
    ) -> "SiglipClassifierModel":
        encoders = PretrainedEncoders(
            siglip_model_id=siglip_model_id, minilm_model_id=minilm_model_id, device=device,
        )
        head = HEAD_TYPES[head_type]()
        state = torch.load(head_checkpoint_path, map_location=device)
        head.load_state_dict(state)
        head.to(device)
        return cls(encoders=encoders, head=head, model_id=model_id, threshold=threshold)


class PretrainedEncoders:
    """Real frozen SigLIP + MiniLM encoders. Loads actual model weights — not unit-tested."""

    def __init__(self, siglip_model_id: str, minilm_model_id: str, device: str = "cpu") -> None:
        from PIL import Image
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModel, AutoProcessor

        self._Image = Image
        self._device = device
        self._siglip = AutoModel.from_pretrained(siglip_model_id).to(device).eval()
        self._processor = AutoProcessor.from_pretrained(siglip_model_id)
        self._minilm = SentenceTransformer(minilm_model_id, device=device)

    @torch.no_grad()
    def encode_image(self, image_path: str) -> torch.Tensor:
        return self.encode_images([image_path])[0]

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        return self.encode_texts([text])[0]

    def encode_image_window(self, image_paths: list[str]) -> torch.Tensor:
        embeds = self.encode_images(image_paths)
        return embeds.mean(dim=0)

    @torch.no_grad()
    def encode_images(self, image_paths: list[str], batch_size: int = 32) -> torch.Tensor:
        """Batched image encoding. Loads and forwards images batch_size at a time."""
        pooled_chunks = []
        for start in range(0, len(image_paths), batch_size):
            chunk_paths = image_paths[start:start + batch_size]
            images = [self._Image.open(p).convert("RGB") for p in chunk_paths]
            inputs = self._processor(images=images, return_tensors="pt").to(self._device)
            output = self._siglip.get_image_features(**inputs)
            pooled = output.pooler_output if hasattr(output, "pooler_output") else output
            pooled_chunks.append(pooled)
        return torch.cat(pooled_chunks, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 128) -> torch.Tensor:
        """Batched text encoding via SentenceTransformer's internal batching."""
        embeddings = self._minilm.encode(texts, convert_to_tensor=True, batch_size=batch_size)
        return embeddings.to(self._device)
