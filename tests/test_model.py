import torch

from fall_detection.labels import FEATURE_DIM, NUM_CLASSES, SEQUENCE_LENGTH
from fall_detection.model import TemporalTransformerClassifier


def test_transformer_output_shape():
    model = TemporalTransformerClassifier()
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(2, SEQUENCE_LENGTH, FEATURE_DIM))
    assert output.shape == (2, NUM_CLASSES)


def test_transformer_rejects_wrong_feature_shape():
    model = TemporalTransformerClassifier()
    try:
        model(torch.zeros(2, SEQUENCE_LENGTH, FEATURE_DIM - 1))
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong feature shape should raise ValueError")
