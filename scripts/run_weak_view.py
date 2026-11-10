"""Two-GPU weak-student-view experiment, with the existing training/evaluation loop."""
import argparse
import copy
import os
from pathlib import Path
import signal
import statistics
import sys
import time

import optuna
from optuna.trial import TrialState

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_bottleneck as p
from scripts.parallel_full_ema import parallel_search
from src.provenance import prepare,read_json,read_records,write_json,verify_source

SPACE={'augment.weak_t_max':(.05,.30,True),
       'loss.cov_iso_mean_lambda':(.05,8.,True),'loss.cov_iso_cov_lambda':(.05,8.,True)}


def suggest(trial,base):
    cfg=copy.deepcopy(base)
    values={key:trial.suggest_float(key,lo,hi,log=log) for key,(lo,hi,log) in SPACE.items()}
    cfg['augment']['weak_view']={'t_lo':0.,'t_max':values.pop('augment.weak_t_max')}
    cfg['loss'].update({key.split('.')[1]:v for key,v in values.items()})
    return cfg


def render(campaign):
    state=read_json(campaign/'state.json')
    lines=['# Weak student view experiment','',f"Stage: {state['stage']}",
           'Clean EMA; CE8192 + iso768; one weak + two original strong student views.',
           'Control = full-clean-EMA trial1 hyperparameters; all runs start from pretrained BERT.',
           'Primary selection: 768d STS-B dev. Test only after selection freeze.',
           '','| trial | state | dev best | dev last |','|---|---|---|---|']
    if (campaign/'bottleneck.db').exists():
        study=optuna.load_study(study_name=campaign.name,storage=f"sqlite:///{campaign/'bottleneck.db'}")
        for t in study.trials:
            m=t.user_attrs.get('run',{}).get('metrics',{})
            lines.append(f"| {t.number} | {t.state.name} | {m.get('best_sts','')} | {m.get('last_sts','')} |")
    if (campaign/'final_runs.json').exists():
        lines+=['','| group | seed | dev best | dev last |','|---|---|---|---|']
        for group,seeds in read_json(campaign/'final_runs.json').items():
            for seed,r in seeds.items():
                lines.append(f"| {group} | {seed} | {r['metrics']['best_sts']} | {r['metrics']['last_sts']} |")
        lines+=['','`baseline`: original strong views. `bottleneck`: selected weak-view condition (internal arm name only; CE stays8192).',
                '`aux_off`: same selected weights with all three original strong views.']
        lines+=['','| group | seed | selected-step alignment | uniformity | effective rank | last weighted mean/cov |',
                '|---|---|---|---|---|---|']
        for group,seeds in read_json(campaign/'final_runs.json').items():
            for seed,r in seeds.items():
                log=campaign/'results/logs'/r['run_name']
                evaluation=next((e for e in read_records(log/'evaluations.jsonl') if e['loop_step']==r['metrics'].get('best_loop_step')),{})
                losses=read_records(log/'components.jsonl')
                loss=losses[-1] if losses else {}
                lines.append(f"| {group} | {seed} | {evaluation.get('alignment','')} | {evaluation.get('uniformity','')} | {evaluation.get('eff_rank','')} | {loss.get('weighted_mean','')} / {loss.get('weighted_cov','')} |")
        for group,seeds in read_json(campaign/'final_runs.json').items():
            if len(seeds)==3:
                for metric in ('best_sts','last_sts'):
                    values=[r['metrics'][metric]*100 for r in seeds.values()]
                    lines.append(f"\n{group} dev {metric}: {statistics.mean(values):.4f} ± {statistics.stdev(values):.4f} (sample SD, n=3)")
    aggregates={}
    for file in sorted((campaign/'results/analysis').glob('*.json')):
        doc=read_json(file)
        for run,scores in doc.get('runs',{}).items():
            lines.append(f"\n{file.stem}: 768d STS7 = {scores['avg_7task']*100:.4f}")
            experiment=doc.get('experiment',{})
            if experiment:
                key=(experiment['group'],experiment['which'])
                aggregates.setdefault(key,{})[str(experiment['seed'])]=scores['avg_7task']
    for (group,which),values in aggregates.items():
        if len(values)==3:
            v=[x*100 for x in values.values()]
            lines.append(f"\n{group} STS7 {which}: {statistics.mean(v):.4f} ± {statistics.stdev(v):.4f} (sample SD, n=3)")
    for comparison in ('baseline','aux_off'):
        weak,control=aggregates.get(('bottleneck','best'),{}),aggregates.get((comparison,'best'),{})
        if len(weak)==len(control)==3:
            deltas=[weak[s]-control[s] for s in ('101','102','103')]
            verdict='성립' if statistics.mean(deltas)>=.003 and sum(d>0 for d in deltas)>=2 else '기각' if statistics.mean(deltas)<=0 and sum(d<=0 for d in deltas)>=2 else '판정불가'
            lines.append(f"\nWeak vs {comparison}: {verdict}; paired mean Δ={statistics.mean(deltas)*100:.4f} percentage points.")
    if state.get('error'):lines+=['',f"Error: {state['error']}"]
    text='\n'.join(lines)+'\n'
    (campaign/'report.md').write_text(text)
    Path(read_json(campaign/'source_manifest.json')['report']).write_text(text)


def stage(campaign,name,**kwargs):
    write_json(campaign/'state.json',{'stage':name,'updated_unix':time.time(),**kwargs})
    render(campaign)


def run(campaign,manifest):
    verify_source(campaign,manifest)
    base=p.load_config(campaign/'source/configs/weak_student_view/base.yaml')
    pilot=p.load_config(campaign/'source/configs/weak_student_view/pilot.yaml')
    # Reuse orchestration helpers with this experiment's reporter, not a training-loop fork.
    p.set_stage,p.render=stage,render
    env=dict(os.environ,PYTHONPATH=str(campaign/'source'),PYTHONDONTWRITEBYTECODE='1')
    children=p.Children(campaign,env)
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
        children.stop()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    calibration={'expected_shared_initial_sha256':{'42':manifest['baseline']['initial_shared_sha256']},
                 'initial_student_sha256':manifest['baseline']['initial_student_sha256']}
    write_json(campaign/'calibration.json',calibration)
    records=[]
    def train(gpu,cfg,label,seed=42,smoke=False):
        result=p.execute_train(campaign,manifest,children,gpu,cfg,label,seed,smoke)
        if result['initial_student_sha256'] != manifest['baseline']['initial_student_sha256'] and seed==42:
            raise RuntimeError('initialization differs from selected clean EMA reference')
        return result
    try:
        stage(campaign,'smoke')
        smoke={}
        p.parallel_jobs(['baseline','weak'],manifest['gpus'],children,
            lambda gpu,arm:train(gpu,base if arm=='baseline' else pilot,f'smoke_{arm}',smoke=True),
            lambda arm,r:smoke.update({arm:r}))
        p.audit_runs(campaign,list(smoke.values()),calibration)
        write_json(campaign/'smoke.json',smoke)
        stage(campaign,'paired_pilot')
        study=optuna.create_study(study_name=campaign.name,storage=f"sqlite:///{campaign/'bottleneck.db'}",
            direction='maximize',sampler=optuna.samplers.TPESampler(seed=20261110,n_startup_trials=5,multivariate=True,constant_liar=True),
            pruner=optuna.pruners.NopPruner())
        anchor={'augment.weak_t_max':.15,**{f'loss.{key}':base['loss'][key] for key in ('cov_iso_mean_lambda','cov_iso_cov_lambda')}}
        study.enqueue_trial(anchor)
        first=study.ask();first_cfg=suggest(first,base)
        paired={}
        p.parallel_jobs(['baseline','weak'],manifest['gpus'],children,
            lambda gpu,arm:train(gpu,base if arm=='baseline' else first_cfg,
                                 'control_s42' if arm=='baseline' else f'search_bottleneck_t{first.number}'),
            lambda arm,r:paired.update({arm:r}))
        first.set_user_attr('run',paired['weak']);study.tell(first,paired['weak']['metrics']['best_sts'])
        records.extend(paired.values())
        p.audit_runs(campaign,records,calibration)
        write_json(campaign/'paired_pilot.json',paired)
        stage(campaign,'search',completed_trials=1)
        def completed(r):
            records.append(r)
            p.audit_runs(campaign,records,calibration)
            stage(campaign,'search',completed_trials=sum(t.state==TrialState.COMPLETE for t in study.trials))
        parallel_search(study,20,manifest['gpus'],lambda t:suggest(t,base),
            lambda number,gpu,cfg:train(gpu,cfg,f'search_bottleneck_t{number}'),completed,children.stop)
        stage(campaign,'confirmation')
        groups,confirmed=p.confirm(campaign,manifest,{'bottleneck':study},children,calibration)
        records.extend(confirmed)
        selected=p.load_config(groups['bottleneck']['101']['config'])
        no_weak=copy.deepcopy(selected);no_weak['augment'].pop('weak_view')
        # Fresh matched-seed controls: fixed reference, plus selected weights without weak view.
        stage(campaign,'matched_controls')
        for group,cfg in [('baseline',base),('aux_off',no_weak)]:
            groups[group]={}
            def save(seed,r):
                groups[group][str(seed)]=r;records.append(r)
                write_json(campaign/'final_runs.json',groups)
            p.parallel_jobs(p.CONFIRM_SEEDS,manifest['gpus'],children,
                lambda gpu,seed:train(gpu,cfg,group,seed),save)
        p.audit_runs(campaign,records,calibration)
        stage(campaign,'test_evaluation')
        p.evaluate(campaign,manifest,groups,children)
        stage(campaign,'complete')
    except BaseException as exc:
        children.stop()
        stage(campaign,'stopped_by_user' if stopped else 'failed',error=str(exc))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();c=args.campaign.resolve()
    if args.execute:
        run(c,read_json(c/'source_manifest.json'));return
    manifest=prepare(c,ROOT,['auto','auto'],ROOT/'docs/experiments/weak_student_view/report.md',experiment_dir='weak_student_view')
    manifest.update(protocol='weak_student_view',search_arms=['bottleneck'],common_space=SPACE,
                    trials_per_arm=20,reference_groups=[],launch_argv=sys.argv)
    write_json(c/'source_manifest.json',manifest)
    write_json(c/'search_registration.json',{'registered_unix':time.time(),'space':SPACE,'trials':20,'steps':4000,
        'search_seed':42,'confirmation_seeds':[101,102,103],'test_only_after_selection':True,'sampler_seed':20261110})
    os.execv(sys.executable,[sys.executable,str(c/'source/scripts/run_weak_view.py'),'--campaign',str(c),'--execute'])

if __name__=='__main__':main()
