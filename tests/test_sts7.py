import pytest
import torch

from scripts import eval_sts7
from src import evaluate


def test_sts7_uses_seven_fixed_test_splits_and_unweighted_mean(monkeypatch):
    calls = []

    def dataset(path, split):
        calls.append((path, split))
        return {
            "sentence1": ["a", "b", "c", "d"],
            "sentence2": ["w", "x", "y", "z"],
            "score": [1.0, 0.8, 0.4, 0.1],
        }

    first = torch.eye(4)
    second = torch.tensor(
        [[1.0, 0, 0, 0], [0.6, 0.8, 0, 0], [0.916515, 0, 0.4, 0], [0.994987, 0, 0, 0.1]]
    )

    def encode(model, tokenizer, sentences, *args):
        return {"embedding": first if sentences[0] == "a" else second}

    monkeypatch.setattr(evaluate, "load_dataset", dataset)
    monkeypatch.setattr(evaluate, "encode_sentences", encode)
    result = evaluate.sts_suite_spearman(None, None, "cpu")
    assert calls == [(path, "test") for path in evaluate.STS_SUITE.values()]
    assert result["avg_7task"] == sum(result[k] for k in evaluate.STS_SUITE) / 7
    assert result["avg_7task"] == pytest.approx(1.0)


def test_checkpoint_loading_uses_saved_architecture_and_rejects_missing_weights(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self, backbone, bottleneck, logits):
            super().__init__()
            assert (backbone, bottleneck, logits) == ("bert-base-uncased", 4, 8)
            self.linear = torch.nn.Linear(4, 8)

    monkeypatch.setattr(eval_sts7, "DinoTextModel", Model)
    expected = Model("bert-base-uncased", 4, 8)
    checkpoint = {
        "model_cfg": {
            "backbone": "bert-base-uncased",
            "head": {"bottleneck_dim": 4, "logit_dim": 8},
        },
        "state_dict": expected.state_dict(),
    }
    model = eval_sts7.load_eval_model(checkpoint, "cpu")
    assert not model.training
    assert all(
        torch.equal(model.state_dict()[key], value) for key, value in expected.state_dict().items()
    )
    checkpoint["state_dict"].pop("linear.bias")
    with pytest.raises(RuntimeError, match="Missing key"):
        eval_sts7.load_eval_model(checkpoint, "cpu")
