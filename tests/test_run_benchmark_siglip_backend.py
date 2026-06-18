from unittest import mock

from src.benchmark.run_benchmark import load_model


def test_load_model_siglip_classifier_backend():
    fake_model = mock.Mock()
    with mock.patch(
        "src.benchmark.models.siglip_classifier.SiglipClassifierModel.from_pretrained",
        return_value=fake_model,
    ) as mock_from_pretrained:
        result = load_model({
            "backend": "siglip_classifier",
            "head_checkpoint_path": "models/siglip_classifier_head.pt",
            "siglip_model_id": "google/siglip-so400m-patch14-384",
            "minilm_model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
        })

    assert result is fake_model
    mock_from_pretrained.assert_called_once_with(
        head_checkpoint_path="models/siglip_classifier_head.pt",
        siglip_model_id="google/siglip-so400m-patch14-384",
        minilm_model_id="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )


def test_load_model_siglip_classifier_backend_defaults():
    fake_model = mock.Mock()
    with mock.patch(
        "src.benchmark.models.siglip_classifier.SiglipClassifierModel.from_pretrained",
        return_value=fake_model,
    ) as mock_from_pretrained:
        load_model({"backend": "siglip_classifier"})

    mock_from_pretrained.assert_called_once_with(
        head_checkpoint_path="models/siglip_classifier_head.pt",
        siglip_model_id="google/siglip-so400m-patch14-384",
        minilm_model_id="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
