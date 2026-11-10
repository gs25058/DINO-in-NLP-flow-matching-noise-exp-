"""Continue an interrupted weak-view study using its immutable training snapshot."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time

import optuna
from optuna.trial import TrialState


def retry_config(trial, weak, base, campaign, attempt):
    original=campaign/f'configs/search_bottleneck_t{trial.number}_s42.yaml'
    cfg=weak.p.load_config(original)
    expected=weak.suggest(optuna.trial.FixedTrial(trial.params),base)
    for section in ('model','train','augment','loss','eval'):
        if cfg[section]!=expected[section]:
            raise RuntimeError(f'interrupted trial {trial.number} config differs in {section}')
    return cfg,f'search_bottleneck_t{trial.number}_{attempt}'


def resume_campaign(c,record,weak):
    p=weak.p
    manifest=weak.read_json(c/'source_manifest.json')
    weak.verify_source(c,manifest)
    if manifest['protocol']!='weak_student_view' or manifest['trials_per_arm']!=20:
        raise RuntimeError('unexpected experiment/budget')
    if weak.read_json(c/'state.json')['stage'] not in ('failed','stopped_by_user'):
        raise RuntimeError('only stopped campaigns can be continued')
    if record.exists():
        raise RuntimeError('preserving existing continuation registration')
    record.parent.mkdir(parents=True,exist_ok=True)
    # Preserve pre-continuation DB/state/report without moving any original artifacts.
    with sqlite3.connect(c/'bottleneck.db') as db,sqlite3.connect(record.parent/'study_before.sqlite') as backup:
        db.backup(backup)
    for name in ('state.json','report.md'):
        (record.parent/f'before_{name}').write_bytes((c/name).read_bytes())
    study=optuna.load_study(study_name=c.name,storage=f"sqlite:///{c/'bottleneck.db'}",
        sampler=optuna.samplers.TPESampler(seed=20261111,n_startup_trials=5,multivariate=True,constant_liar=True),
        pruner=optuna.pruners.NopPruner())
    interrupted=[t for t in study.trials if t.state==TrialState.RUNNING]
    if len(interrupted)>2 or any(t.state not in (TrialState.COMPLETE,TrialState.RUNNING) for t in study.trials):
        raise RuntimeError('unexpected persisted study state')
    base=p.load_config(c/'source/configs/weak_student_view/base.yaml')
    retries={t.number:retry_config(t,weak,base,c,record.parent.name) for t in interrupted}
    weak.write_json(record,{
        'created_unix':time.time(),'reason':'User requested continuation after interrupted GPU jobs',
        'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'training_source_commit':manifest['git']['commit'],'training_snapshot_unchanged':True,
        'completed_preserved':[t.number for t in study.trials if t.state==TrialState.COMPLETE],
        'retries':{str(t.number):{'params':t.params,'label':retries[t.number][1],
                    'fresh_pretrained_start':True,'original_partial_artifacts_preserved':True} for t in interrupted},
        'total_search_budget':20,'max_gpu_jobs':2,'sampler_seed':20261111,'constant_liar':True,
        'sampler_note':'Old sampler RNG state was not persisted; continuation seed explicitly registered before new suggestions.'})
    calibration=weak.read_json(c/'calibration.json')
    records=[weak.read_json(c/'paired_pilot.json')['baseline']]
    records += [t.user_attrs['run'] for t in study.trials if t.state==TrialState.COMPLETE]
    env=dict(os.environ,PYTHONPATH=str(c/'source'),PYTHONDONTWRITEBYTECODE='1')
    children=p.Children(c,env)
    p.set_stage,p.render=weak.stage,weak.render
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
        children.stop()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def prepare(trial):
        return retries[trial.number][0] if trial.number in retries else weak.suggest(trial,base)
    def execute(number,gpu,cfg):
        label=retries[number][1] if number in retries else f'search_bottleneck_t{number}'
        return p.execute_train(c,manifest,children,gpu,cfg,label,42)
    def completed(run):
        records.append(run)
        p.audit_runs(c,records,calibration)
        weak.stage(c,'search',completed_trials=sum(t.state==TrialState.COMPLETE for t in study.trials),
                   continuation_registration=str(record))
    try:
        p.audit_runs(c,records,calibration)
        weak.stage(c,'search',completed_trials=sum(t.state==TrialState.COMPLETE for t in study.trials),
                   continuation_registration=str(record))
        weak.parallel_search(study,20,manifest['gpus'],prepare,execute,completed,children.stop)
        weak.stage(c,'confirmation')
        groups,confirmed=p.confirm(c,manifest,{'bottleneck':study},children,calibration)
        records.extend(confirmed)
        selected=p.load_config(groups['bottleneck']['101']['config'])
        no_weak=copy.deepcopy(selected);no_weak['augment'].pop('weak_view')
        weak.stage(c,'matched_controls')
        for group,cfg in [('baseline',base),('aux_off',no_weak)]:
            groups[group]={}
            def save(seed,run):
                groups[group][str(seed)]=run;records.append(run)
                weak.write_json(c/'final_runs.json',groups)
            p.parallel_jobs(p.CONFIRM_SEEDS,manifest['gpus'],children,
                lambda gpu,seed:p.execute_train(c,manifest,children,gpu,cfg,group,seed),save)
        p.audit_runs(c,records,calibration)
        weak.stage(c,'test_evaluation')
        p.evaluate(c,manifest,groups,children)
        weak.stage(c,'complete')
    except BaseException as exc:
        children.stop()
        weak.stage(c,'stopped_by_user' if stopped else 'failed',error=str(exc),continuation_registration=str(record))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--record',type=Path,required=True)
    args=parser.parse_args();c=args.campaign.resolve()
    # Training, search-space definitions and evaluation come from the original frozen source.
    sys.path.insert(0,str(c/'source'))
    from scripts import run_weak_view as weak
    resume_campaign(c,args.record.resolve(),weak)

if __name__=='__main__':main()
