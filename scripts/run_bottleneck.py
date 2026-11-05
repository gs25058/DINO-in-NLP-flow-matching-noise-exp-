"""Paired 20×4000 Optuna, fresh 3-seed confirmations, component ablations, then STS7."""

import argparse
import copy
import os
import random
import signal
import sys
import time
from pathlib import Path

import optuna
import torch
import yaml
from optuna.trial import TrialState

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_bottleneck import render
from scripts.run_coviso_optuna import Children, complete_record, parallel_jobs, top_trials
from src.bottleneck import (
    ARMS,
    CONFIRM_SEEDS,
    SAMPLER_SEED,
    SEARCH_SEED,
    anchor_parameters,
    calibrate,
    common_space,
    trial_config,
)
from src.provenance import prepare, read_json, sha256_file, verify_source, write_json
from src.provenance import model_fingerprint
from src.model import DinoTextModel
from src.train import Trainer, load_config, setup_logger


def set_stage(campaign, stage, **extra):
    write_json(campaign / "state.json", {"stage": stage, "updated_unix": time.time(), **extra})
    render(campaign)


def save_config(campaign, manifest, base, label, seed):
    cfg = copy.deepcopy(base)
    protocol = manifest.get("protocol", "bottleneck_cosine")
    prefix = {"bottleneck_iso": "bneckiso", "bottleneck_dino": "bneckdino"}.get(protocol, "bneckcos")
    cfg.update(seed=seed, run_name=f"{prefix}_{campaign.name}_{label}_s{seed}")
    for key in ("sentences_path", "rank_eval_path", "embed_stats_path"):
        cfg["data"][key] = str(Path(manifest["workspace"]) / cfg["data"][key])
    cfg["experiment"] = {
        "source_manifest": str(campaign / "source_manifest.json"),
        "protocol": protocol,
        "label": label,
        "fresh_pretrained_start": True,
        "spec_sha256": manifest["spec_sha256"],
    }
    path = campaign / "configs" / f"{label}_s{seed}.yaml"
    if path.exists() and read_yaml(path) != cfg:
        raise RuntimeError(f"preserving changed config: {path}")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path, cfg


def read_yaml(path):
    return yaml.safe_load(path.read_text())


def execute_train(campaign, manifest, children, gpu, base, label, seed, smoke=False):
    path, cfg = save_config(campaign, manifest, base, label, seed)
    log = campaign / "results/logs" / cfg["run_name"]
    completion = log / "completion.json"
    if not completion.exists():
        verify_source(campaign, manifest)
        command = [sys.executable, "-m", "src.train", "--config", str(path), "--device", "cuda"]
        if smoke:
            command += ["--stop-after-steps", "60"]
        cal = read_json(campaign / "calibration.json")
        children.run(
            gpu,
            f"train_{label}_s{seed}",
            command,
            evaluation_path=log / "evaluations.jsonl",
            expected_initialization=cal.get("initial_student_sha256") if seed == SEARCH_SEED else None,
        )
    metrics = read_json(completion) if smoke else complete_record(completion)
    if smoke and (metrics["status"] != "smoke_complete" or metrics["completed_steps"] != 60):
        raise RuntimeError("60-step smoke incomplete")
    provenance = read_json(log / "provenance.json")
    order = read_json(log / "batch_order.json")
    return {
        "run_name": cfg["run_name"],
        "config": str(path),
        "metrics": metrics,
        "seed": seed,
        "label": label,
        "initial_student_sha256": provenance["initial_student_sha256"],
        "initial_shared_sha256": provenance.get("initial_shared_sha256", provenance["initial_student_sha256"]),
        "architecture": cfg["model"]["head"]["logit_dim"],
        "first_views": read_json(log / "first_views.json") if (log / "first_views.json").exists() else None,
        "batch_order": order,
        "data": provenance["data"],
    }


def audit_runs(campaign, runs, calibration):
    hashes = {}
    full_hashes = {}
    first_views = {}
    orders = {}
    data = None
    for run in runs:
        seed = run["seed"]
        fingerprint = run.get("initial_shared_sha256", run["initial_student_sha256"])
        expected = calibration.get("expected_shared_initial_sha256", {}).get(str(seed))
        if expected is not None and fingerprint != expected:
            raise RuntimeError("shared initialization differs from the measured fixed baseline")
        architecture = f"{seed}_{run.get('architecture', 'same')}"
        full = run["initial_student_sha256"]
        if full_hashes.setdefault(architecture, full) != full:
            raise RuntimeError("unequal initialization within the same architecture and seed")
        if run.get("first_views") is not None:
            if first_views.setdefault(seed, run["first_views"]) != run["first_views"]:
                raise RuntimeError("unequal initial noisy views within seed")
        order = run["batch_order"]["sha256"]
        if hashes.setdefault(seed, fingerprint) != fingerprint:
            raise RuntimeError("unequal pretrained initialization within seed")
        if orders.setdefault(seed, order) != order:
            raise RuntimeError("unequal data order within seed")
        identities = {k: v["sha256"] for k, v in run["data"].items()}
        if data is None:
            data = identities
        if data != identities:
            raise RuntimeError("training data changed between runs")
        if (
            seed == SEARCH_SEED
            and "first_indices" in calibration
            and run["batch_order"]["first_indices"] != calibration["first_indices"]
        ):
            raise RuntimeError("calibration minibatch differs from training")
    write_json(
        campaign / "comparability_audit.json",
        {
            "same_initialization_per_seed": True,
            "same_data_order_per_seed": True,
            "initial_sha256": hashes,
            "initial_hash_scope": "shared backbone and MLP; full architecture hashes checked separately",
            "full_initial_sha256": full_hashes,
            "first_views": first_views,
            "batch_order_sha256": orders,
            "data_sha256": data,
        },
    )


def search(campaign, manifest, base, calibration, children):
    studies = {}
    all_runs = []
    arms = manifest.get("search_arms", ARMS)
    for arm in arms:
        study = optuna.create_study(
            storage=f"sqlite:///{campaign / f'{arm}.db'}",
            study_name=f"{campaign.name}_{arm}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=SAMPLER_SEED, n_startup_trials=5, multivariate=True
            ),
            pruner=optuna.pruners.NopPruner(),
            load_if_exists=True,
        )
        if study.trials:
            raise RuntimeError("existing study preserved; no silent sampler restart")
        study.enqueue_trial(anchor_parameters(base, arm, calibration))
        studies[arm] = study
    for number in range(manifest["trials_per_arm"]):
        pending = {}
        for arm in arms:
            trial = studies[arm].ask()
            pending[arm] = (trial, trial_config(trial, base, arm, calibration))

        def execute(gpu, arm):
            trial, cfg = pending[arm]
            return execute_train(
                campaign, manifest, children, gpu, cfg, f"search_{arm}_t{trial.number}", SEARCH_SEED
            )

        def completed(arm, run):
            trial, _ = pending[arm]
            trial.set_user_attr("run", run)
            studies[arm].tell(trial, run["metrics"]["best_sts"])
            all_runs.append(run)
            audit_runs(campaign, all_runs, calibration)
            counts = {
                name: sum(t.state == TrialState.COMPLETE for t in study.trials)
                for name, study in studies.items()
            }
            set_stage(
                campaign, "search", completed_pairs=min(counts.values()), completed_by_arm=counts
            )

        try:
            parallel_jobs(arms, manifest["gpus"], children, execute, completed)
        except BaseException:
            for arm, (trial, _) in pending.items():
                if studies[arm].trials[trial.number].state == TrialState.RUNNING:
                    studies[arm].tell(trial, state=TrialState.FAIL)
            raise
    return studies, all_runs


def confirm(campaign, manifest, studies, children, calibration):
    arms = tuple(studies)
    shortlist = {arm: [t.number for t in top_trials(studies[arm])] for arm in arms}
    if any(len(ids) != 3 for ids in shortlist.values()):
        raise RuntimeError("three completed candidates required per arm")
    write_json(
        campaign / "shortlist.json",
        {"trials": shortlist, "new_seeds": CONFIRM_SEEDS, "test_scores_seen": False},
    )
    confirmed = {arm: {str(n): {} for n in shortlist[arm]} for arm in arms}
    jobs = [(arm, n, seed) for arm in arms for n in shortlist[arm] for seed in CONFIRM_SEEDS]

    def execute(gpu, job):
        arm, n, seed = job
        parent = load_config(studies[arm].trials[n].user_attrs["run"]["config"])
        return execute_train(campaign, manifest, children, gpu, parent, f"confirm_{arm}_t{n}", seed)

    def completed(job, run):
        arm, n, seed = job
        confirmed[arm][str(n)][str(seed)] = run
        write_json(campaign / "confirmations.json", confirmed)
        set_stage(campaign, "confirmation")

    parallel_jobs(jobs, manifest["gpus"], children, execute, completed)
    selected = {
        arm: min(
            shortlist[arm],
            key=lambda n: (
                -sum(confirmed[arm][str(n)][str(s)]["metrics"]["best_sts"] for s in CONFIRM_SEEDS)
                / len(CONFIRM_SEEDS),
                n,
            ),
        )
        for arm in arms
    }
    selection = {
        "selected_trials": selected,
        "selection_frozen_unix": time.time(),
        "test_scores_seen": False,
        "selection_metric": "mean 768d STS-B dev best over seeds101,102,103",
        "selected_params": {arm: studies[arm].trials[n].params for arm, n in selected.items()},
    }
    write_json(campaign / "selection.json", selection)
    groups = {arm: confirmed[arm][str(n)] for arm, n in selected.items()}
    write_json(campaign / "final_runs.json", groups)
    return groups, [
        r
        for candidates in confirmed.values()
        for seeds in candidates.values()
        for r in seeds.values()
    ]


def ablate(campaign, manifest, groups, children):
    settings = {"aux_off": "bneck_cosine_lambda"}
    if manifest.get("protocol") == "bottleneck_iso":
        settings["iso_off"] = "bneck_iso_lambda"
    all_runs = []
    for group, key in settings.items():
        off = load_config(groups["bottleneck"][str(CONFIRM_SEEDS[0])]["config"])
        off["loss"][key] = 0.0
        results = {}
        parallel_jobs(
            CONFIRM_SEEDS, manifest["gpus"], children,
            lambda gpu, seed: execute_train(campaign, manifest, children, gpu, off, group, seed),
            lambda seed, run: results.update({str(seed): run}),
        )
        groups[group] = results
        write_json(campaign / "final_runs.json", groups)
        for seed in CONFIRM_SEEDS:
            original = load_config(groups["bottleneck"][str(seed)]["config"])
            changed = load_config(results[str(seed)]["config"])
            for cfg in (original, changed):
                cfg.pop("run_name")
                cfg.pop("experiment")
                cfg["loss"].pop(key)
            if original != changed:
                raise RuntimeError(f"{group} changed another hyperparameter")
        all_runs.extend(results.values())
    return all_runs


def evaluate(campaign, manifest, groups, children):
    if not (campaign / "selection.json").exists():
        raise RuntimeError("test requires frozen selection")
    jobs = [
        (group, seed, which)
        for group in groups
        if group not in manifest.get("reference_groups", [])
        for seed in CONFIRM_SEEDS
        for which in ("best", "last")
    ]

    def execute(gpu, job):
        group, seed, which = job
        run = groups[group][str(seed)]
        output = campaign / "results/analysis" / f"{group}_s{seed}_{which}.json"
        children.run(
            gpu,
            f"eval_{group}_s{seed}_{which}",
            [
                sys.executable,
                "scripts/eval_sts7.py",
                "--runs",
                run["run_name"],
                "--which",
                which,
                "--device",
                "cuda",
                "--tf32",
                "--diagnose-bottleneck",
                "--out",
                str(output),
            ],
        )
        result = read_json(output)
        if (
            run["run_name"] not in result["runs"]
            or run["run_name"] not in result["bottleneck_diagnostic"]
        ):
            raise RuntimeError("missing requested primary or diagnostic test scores")
        result["experiment"] = {
            "group": group,
            "seed": seed,
            "which": which,
            "selection_frozen_before_test": True,
        }
        write_json(output, result)

    parallel_jobs(jobs, manifest["gpus"], children, execute, lambda *_: render(campaign))


def fixed_baseline(campaign, manifest, base):
    """Verify and import measured reference scores, never retune or rewrite them."""
    previous = Path(manifest["reference_baseline_campaign"])
    old = read_json(previous / "source_manifest.json")
    verify_source(previous, old)
    baseline = read_json(previous / "final_runs.json")["baseline"]
    expected_shared = {}
    for seed in (SEARCH_SEED, *CONFIRM_SEEDS):
        if seed == SEARCH_SEED:
            configs = sorted((previous / "configs").glob("search_baseline_t0_s42.yaml"))
            if len(configs) != 1:
                raise RuntimeError("reference seed42 initialization record missing")
            cfg = load_config(configs[0])
            log = previous / "results/logs" / cfg["run_name"]
            run = None
        else:
            run = baseline[str(seed)]
            cfg = load_config(run["config"])
            log = previous / "results/logs" / run["run_name"]
        complete_record(log / "completion.json")
        prov = read_json(log / "provenance.json")
        if load_config(log / "resolved.yaml") != cfg:
            raise RuntimeError("reference config differs from its recorded resolved config")
        if cfg["augment"] != base["augment"] or cfg["model"] != base["model"]:
            raise RuntimeError("reference noise/model differs from the declared fixed baseline")
        fixed = {
            "train": ("batch_size", "max_steps", "weight_decay", "warmup_frac", "grad_clip",
                      "momentum_start", "momentum_end", "momentum_ramp_frac", "tf32"),
            "loss": ("center_momentum", "cov_ema_momentum", "cov_iso_start_step", "teacher_temp_warmup_frac"),
            "data": ("max_tokens",),
            "eval": ("every_steps", "log_every_steps", "batch_kl_pairs"),
        }
        if any(cfg[section][key] != base[section][key] for section, keys in fixed.items() for key in keys):
            raise RuntimeError("fixed reference training/evaluation protocol differs")
        for key, item in prov["data"].items():
            current = Path(manifest["workspace"]) / base["data"][key]
            if sha256_file(current) != item["sha256"]:
                raise RuntimeError("reference data differs from current data")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            initial = DinoTextModel(cfg["model"]["backbone"], 256, 8192)
            if model_fingerprint(initial) != prov["initial_student_sha256"]:
                raise RuntimeError("reference initial model cannot be reconstructed exactly")
            expected_shared[str(seed)] = model_fingerprint(initial, ("head.expand.",))
            del initial
        if run is None:
            continue
        for target in ("results/logs", "checkpoints"):
            dest = campaign / target / run["run_name"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(previous / target / run["run_name"], target_is_directory=True)
        for which in ("best", "last"):
            source = previous / "results/analysis" / f"baseline_s{seed}_{which}.json"
            doc = read_json(source)
            doc["reference"] = {"campaign": str(previous), "copied_without_reevaluation": True}
            write_json(campaign / "results/analysis" / source.name, doc)
    write_json(campaign / "fixed_baseline.json", {
        "source_campaign": str(previous), "source_commit": old["git"]["commit"],
        "selected_by_previous_dev": read_json(previous / "selection.json"),
        "expected_shared_initial_sha256": expected_shared,
        "runs": baseline, "retuned": False, "prior_test_scores_already_known": True,
    })
    identities = read_json(previous / "evaluation_data_identity.json")
    for dataset in identities["datasets"].values():
        for cache in dataset["cache_files"]:
            if sha256_file(cache["path"]) != cache["sha256"]:
                raise RuntimeError("reference evaluation data changed")
    write_json(campaign / "evaluation_data_identity.json", identities)
    return baseline


def run(campaign, manifest):
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HOME=os.environ.get("HF_HOME", "/src/gs25058/scratch/.hf_home"),
        PYTHONPATH=str(campaign / "source"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    children = Children(campaign, env)
    requested_stop = False

    def stop(*_):
        nonlocal requested_stop
        requested_stop = True
        children.stop()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    protocol = manifest.get("protocol", "bottleneck_cosine")
    base = load_config(campaign / "source/configs" / protocol / "base.yaml")
    direct = protocol == "bottleneck_dino"
    stage = "calibration"
    try:
        if direct:
            stage = "reference_verification"
            set_stage(campaign, stage)
            reference = fixed_baseline(campaign, manifest, base)
            # Iso is tuned directly; the existing FlowDINO reference stays fixed.
            write_json(campaign / "calibration.json", {
                "not_used": "No cosine auxiliary; direct DINO tunes iso in the preregistered [1e-3,10] log range.",
                "expected_shared_initial_sha256": read_json(campaign / "fixed_baseline.json")["expected_shared_initial_sha256"],
            })
        else:
            set_stage(campaign, stage)
            path, _ = save_config(campaign, manifest, base, "calibration", SEARCH_SEED)
            children.run(
                manifest["gpus"][0], "calibration",
                [sys.executable, "scripts/run_bottleneck.py", "--campaign", str(campaign),
                 "--calibration-child", "--config", str(path)],
            )
        calibration = read_json(campaign / "calibration.json")
        write_json(
            campaign / "search_registration.json",
            {
                "registered_before_training_unix": time.time(),
                "common_space": common_space(base),
                **{key: calibration[key] for key in ("lambda_range", "lambda_anchor") if key in calibration},
                **{key: value for key, value in calibration.items() if key.startswith("iso_lambda_")},
                "trials_per_arm": manifest["trials_per_arm"],
                "steps": 4000,
                "pruning": "none",
                "search_seed": SEARCH_SEED,
                "confirm_seeds": CONFIRM_SEEDS,
                "calibration_sha256": sha256_file(campaign / "calibration.json"),
                "spec_sha256": manifest["spec_sha256"],
            },
        )
        stage = "smoke"
        set_stage(campaign, stage)

        def smoke(gpu, arm):
            if direct:
                cfg = trial_config(
                    optuna.trial.FixedTrial(anchor_parameters(base, arm, calibration)),
                    base, arm, calibration,
                )
                return execute_train(
                    campaign, manifest, children, gpu, cfg, f"smoke_{arm}", SEARCH_SEED, smoke=True
                )
            cfg = copy.deepcopy(base)
            cfg["loss"]["bneck_cosine_lambda"] = (
                calibration["lambda_anchor"] if arm == "bottleneck" else 0.0
            )
            if "bneck_iso_lambda" in cfg["loss"]:
                cfg["loss"]["bneck_iso_lambda"] = (
                    calibration["iso_lambda_anchor"] if arm == "bottleneck" else 0.0
                )
            return execute_train(
                campaign, manifest, children, gpu, cfg, f"smoke_{arm}", SEARCH_SEED, smoke=True
            )

        smoked = {}
        parallel_jobs(
            ARMS,
            manifest["gpus"],
            children,
            smoke,
            lambda arm, result: smoked.update({arm: result}),
        )
        audit_runs(campaign, list(smoked.values()), calibration)
        write_json(campaign / "smoke.json", smoked)
        stage = "search"
        set_stage(campaign, stage)
        studies, runs = search(campaign, manifest, base, calibration, children)
        stage = "confirmation"
        set_stage(campaign, stage)
        groups, confirmed = confirm(campaign, manifest, studies, children, calibration)
        if direct:
            groups["baseline"] = reference
            write_json(campaign / "final_runs.json", groups)
        off = []
        if not direct:
            stage = "aux_off"
            set_stage(campaign, stage)
            off = ablate(campaign, manifest, groups, children)
        audit_runs(campaign, runs + confirmed + off, calibration)
        stage = "test_evaluation"
        set_stage(campaign, stage)
        evaluate(campaign, manifest, groups, children)
        set_stage(campaign, "complete")
    except BaseException as exc:
        children.stop()
        set_stage(campaign, "stopped_by_user" if requested_stop else "failed",
                  interrupted_stage=stage, error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--gpus", default="auto,auto")
    parser.add_argument("--trials-per-arm", type=int, default=20)
    parser.add_argument("--budget-reason", default="")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--calibration-child", action="store_true")
    parser.add_argument("--config", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--with-bneck-iso", action="store_true",
                        help="joint cosine + 256d isotropy protocol, with iso-off confirmation")
    mode.add_argument("--direct-bottleneck-dino", action="store_true",
                      help="tune direct256 DINO+iso against the fixed, already-tuned FlowDINO reference")
    parser.add_argument("--reference-baseline", type=Path,
                        default=ROOT / "results/experiments/bneck_iso_20261104")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.calibration_child:
        cfg = load_config(args.config)
        torch.manual_seed(cfg["seed"])
        random.seed(cfg["seed"])

        class Writer:
            def add_scalar(self, *args):
                pass

        trainer = Trainer(
            cfg,
            "cuda",
            setup_logger(campaign / "results/logs" / cfg["run_name"]),
            Writer(),
            root=campaign / "source",
        )
        write_json(campaign / "calibration.json", calibrate(trainer))
        return
    if args.execute:
        manifest = read_json(campaign / "source_manifest.json")
        verify_source(campaign, manifest)
        if sha256_file(campaign / "OPTUNA_SPEC.md") != manifest["spec_sha256"]:
            raise RuntimeError("changed preregistration")
        run(campaign, manifest)
        return
    gpus = args.gpus.split(",")
    if len(gpus) not in (1, 2) or (len(set(gpus)) != len(gpus) and gpus != ["auto", "auto"]):
        parser.error("one/two distinct GPU indices or auto/auto,auto required")
    if "auto" in gpus and any(g != "auto" for g in gpus):
        parser.error("do not mix explicit and automatic GPU selection")
    if not 3 <= args.trials_per_arm <= 20:
        parser.error("3–20 trials per arm required")
    if args.trials_per_arm != 20 and not args.budget_reason.strip():
        parser.error("reduced equal budget requires a reason before starting")
    protocol = (
        "bottleneck_dino" if args.direct_bottleneck_dino else
        "bottleneck_iso" if args.with_bneck_iso else "bottleneck_cosine"
    )
    manifest = prepare(
        campaign,
        ROOT,
        gpus,
        ROOT / "docs/experiments" / protocol / "report.md",
        experiment_dir=protocol,
    )
    base = load_config(ROOT / "configs" / protocol / "base.yaml")
    manifest.update(
        trials_per_arm=args.trials_per_arm,
        budget_reason=args.budget_reason,
        common_space=common_space(base),
        launch_argv=sys.argv,
        protocol=protocol,
        initial_workspace_branch=manifest["git"]["branch"],
    )
    if args.direct_bottleneck_dino:
        manifest.update(
            search_arms=["bottleneck"], reference_groups=["baseline"],
            reference_baseline_campaign=str(args.reference_baseline.resolve()),
        )
    write_json(campaign / "source_manifest.json", manifest)
    if not args.prepare_only:
        os.execv(
            sys.executable,
            [
                sys.executable,
                str(campaign / "source/scripts/run_bottleneck.py"),
                "--campaign",
                str(campaign),
                "--execute",
            ],
        )


if __name__ == "__main__":
    main()
