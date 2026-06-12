import pytest
import unittest.mock as mock
from src.benchmark.models.base_model import BaseVisionModel, PredictionResult


class _StubModel(BaseVisionModel):
    def predict_unanswerable(self, document_path, question, prompt_template, page_indices=None):
        return PredictionResult(sample_id="", predicted_unanswerable=False,
                                confidence=-1, raw_response="")
    def name(self):
        return "stub"


def test_generate_base_raises():
    stub = _StubModel()
    with pytest.raises(NotImplementedError):
        stub.generate("doc.png", "transcribe this")


def test_vllm_generate_returns_text(tmp_path):
    """VllmModel.generate() posts to the completions endpoint and returns raw text."""
    from src.benchmark.models.vllm_model import VllmModel
    from PIL import Image
    img = Image.new("RGB", (4, 4), color=(255, 255, 255))
    img_path = tmp_path / "page.png"
    img.save(img_path)

    model = VllmModel(base_url="http://fake:9999/v1", model_id="test/m", api_key="x")

    fake_response = mock.MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "Transcribed text from page."}}]
    }
    fake_response.raise_for_status = mock.MagicMock()

    with mock.patch.object(model._requests, "post", return_value=fake_response) as mock_post:
        result = model.generate(str(img_path), "Transcribe this page.", max_tokens=512)

    assert result == "Transcribed text from page."
    call_payload = mock_post.call_args[1]["json"]
    assert call_payload["max_tokens"] == 512
    assert call_payload["temperature"] == 0.0
    content = call_payload["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "Transcribe this page."
