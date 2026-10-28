import json
import sys
from types import SimpleNamespace

import optuna
import torch

from scripts import eval_study_sts7 as evaluate


def test_posthoc_cache_is_invalidated_by_pooling_and_skips_missing_collapsed(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "load_config", lambda _: {"model": {"backbone": "offline"}})
    trials = [SimpleNamespace(number=i, value=0.7 + i / 100, params={},
                              user_attrs={"collapsed": i == 2}, state=optuna.trial.TrialState.COMPLETE)
              for i in range(3)]
    monkeypatch.setattr(evaluate.optuna, "load_study", lambda **_: SimpleNamespace(trials=trials))
    checkpoint = tmp_path / "checkpoints" / "test_t0" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"state_dict": {}}, checkpoint)
    calls = []
    monkeypatch.setattr(evaluate, "load_eval_model", lambda *_: (object(), "offline", "flowdino"))
    monkeypatch.setattr(evaluate.AutoTokenizer, "from_pretrained", lambda _: object())

    def scores(*_, pooling):
        calls.append(pooling)
        return {"avg_7task": 0.6}

    monkeypatch.setattr(evaluate, "sts_suite_spearman", scores)
    argv = ["eval_study_sts7", "--study", "test", "--config", "offline.yaml"]
    monkeypatch.setattr(sys, "argv", argv)
    assert evaluate.main() == 0
    assert evaluate.main() == 0
    assert calls == ["last"]
    monkeypatch.setattr(sys, "argv", argv + ["--pooling", "cls"])
    assert evaluate.main() == 0
    assert calls == ["last", "cls"]
    result = json.loads((tmp_path / "results" / "analysis" / "test_sts7.json").read_text())
    assert list(result["trials"]) == ["0"]
    assert result["context"]["pooling"] == "cls"
