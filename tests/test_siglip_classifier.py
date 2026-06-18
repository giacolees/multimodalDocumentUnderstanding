import torch

from src.benchmark.models.siglip_classifier import (
    ClassifierHead,
    IMAGE_EMBED_DIM,
    TEXT_EMBED_DIM,
)


def test_classifier_head_forward_shape():
    head = ClassifierHead()
    image_embed = torch.randn(4, IMAGE_EMBED_DIM)
    text_embed = torch.randn(4, TEXT_EMBED_DIM)
    logits = head(image_embed, text_embed)
    assert logits.shape == (4,)
    assert logits.dtype == torch.float32


def test_classifier_head_gradients_flow():
    head = ClassifierHead()
    image_embed = torch.randn(2, IMAGE_EMBED_DIM)
    text_embed = torch.randn(2, TEXT_EMBED_DIM)
    labels = torch.tensor([1.0, 0.0])
    logits = head(image_embed, text_embed)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    grads = [p.grad for p in head.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
