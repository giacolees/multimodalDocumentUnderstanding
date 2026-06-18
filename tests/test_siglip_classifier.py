import torch
from unittest import mock

from src.benchmark.models.base_model import PredictionResult
from src.benchmark.models.siglip_classifier import (
    ClassifierHead,
    IMAGE_EMBED_DIM,
    TEXT_EMBED_DIM,
    SiglipClassifierModel,
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


class _FakeEncoders:
    def __init__(self, image_value: float, text_value: float):
        self.image_value = image_value
        self.text_value = text_value
        self.image_calls: list[str] = []
        self.text_calls: list[str] = []

    def encode_image(self, image_path: str) -> torch.Tensor:
        self.image_calls.append(image_path)
        return torch.full((IMAGE_EMBED_DIM,), self.image_value)

    def encode_text(self, text: str) -> torch.Tensor:
        self.text_calls.append(text)
        return torch.full((TEXT_EMBED_DIM,), self.text_value)


def _head_that_always_outputs(logit_value: float) -> ClassifierHead:
    head = ClassifierHead()
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.net[-1].bias.fill_(logit_value)
    return head


def test_predict_unanswerable_above_threshold():
    encoders = _FakeEncoders(image_value=1.0, text_value=1.0)
    head = _head_that_always_outputs(10.0)  # sigmoid(10) ~ 0.99995
    model = SiglipClassifierModel(encoders=encoders, head=head)

    result = model.predict_unanswerable(
        document_path="data/raw/docvqa/val/documents/416.png",
        question="What is the full form of FDA?",
        prompt_template="ignored",
    )

    assert isinstance(result, PredictionResult)
    assert result.predicted_unanswerable is True
    assert result.confidence > 0.99
    assert result.skipped is False
    assert encoders.image_calls == ["data/raw/docvqa/val/documents/416.png"]
    assert encoders.text_calls == ["What is the full form of FDA?"]


def test_predict_unanswerable_below_threshold():
    encoders = _FakeEncoders(image_value=1.0, text_value=1.0)
    head = _head_that_always_outputs(-10.0)  # sigmoid(-10) ~ 0.00005
    model = SiglipClassifierModel(encoders=encoders, head=head)

    result = model.predict_unanswerable(
        document_path="doc.png", question="q", prompt_template="ignored",
    )

    assert result.predicted_unanswerable is False
    assert result.confidence < 0.01


def test_predict_unanswerable_handles_encoder_failure():
    encoders = mock.Mock()
    encoders.encode_image.side_effect = FileNotFoundError("missing image")
    head = ClassifierHead()
    model = SiglipClassifierModel(encoders=encoders, head=head)

    result = model.predict_unanswerable(
        document_path="missing.png", question="q", prompt_template="ignored",
    )

    assert result.skipped is True
    assert result.predicted_unanswerable is False
    assert "missing image" in result.raw_response


def test_name_returns_model_id():
    model = SiglipClassifierModel(
        encoders=mock.Mock(), head=ClassifierHead(), model_id="siglip_classifier",
    )
    assert model.name() == "siglip_classifier"
