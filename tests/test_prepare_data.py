import sys

import pytest
import torch

from scripts import prepare_data


class Tokenizer:
    def __call__(self, sentences, **kwargs):
        return {
            "input_ids": torch.tensor([[0, 1, 2, 3]] * len(sentences)),
            "attention_mask": torch.tensor([[0, 1, 1, 1]] * len(sentences)),
            "special_tokens_mask": torch.tensor([[1, 0, 0, 1]] * len(sentences)),
        }


def test_statistics_use_only_content_tokens_and_keep_double_precision_accumulation():
    weights = torch.tensor([[100.0, 100.0], [1.0, 2.0], [3.0, 6.0], [200.0, 200.0]])
    embedding = torch.nn.Embedding.from_pretrained(weights)
    mean, std = prepare_data.compute_embed_stats(
        ["a", "b"], Tokenizer(), embedding, "cpu", batch_size=1
    )
    assert torch.equal(mean, torch.tensor([2.0, 4.0]))
    assert torch.equal(std, torch.tensor([1.0, 2.0]))


def test_existing_cache_is_rejected_before_download(tmp_path, monkeypatch):
    path = tmp_path / "sentences.jsonl"
    path.write_text("preserved data")
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        prepare_data.main()
    assert path.read_text() == "preserved data"
