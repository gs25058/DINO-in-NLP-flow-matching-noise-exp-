"""Paired A/B/C context study: shared 12 parameter combinations, fresh-seed checks."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time

import optuna

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_bottleneck as p
from src.provenance import prepare,read_json,read_records,write_json,verify_source,sha256_file

ARMS={'A':'none','B':'real','C':'unrelated','D':'far'}
SEEDS=(101,102,103)
SPACE={'backbone_lr':(3.4382477416615245e-5,1.3752990966646098e-4,True),
       'head_lr':(.00029124167925665,.0011649667170266,True),
       'beta':(.25,2.,True),'weak_t_max':(.02,.20,True)}


def parameters(trial):
    return {k:trial.suggest_float(k,lo,hi,log=log) for k,(lo,hi,log) in SPACE.items()}


def configuration(base,params,arm):
    cfg=copy.deepcopy(base)
    cfg['train'].update(lr=params['backbone_lr'],head_lr=params['head_lr'])
    cfg['context'].update(mode=ARMS[arm],beta=params['beta'],weak_t_max=params['weak_t_max'])
    cfg['loss']['view_weights']=[1.,1.,params['beta']]
    return cfg


def audit(campaign,records,calibration):
    p.audit_runs(campaign,records,calibration)
    first={};layouts={}
    for r in records:
        cfg=p.load_config(r['config']);info=read_json(campaign/'results/logs'/r['run_name']/'context_first_views.json')
        seed=r['seed'];key=(seed,r['label'].split('_')[0]=='smoke')
        common={k:info[k] for k in ('center_noise_sha256','center_token_sha256','teacher_clean_sha256')}
        if first.setdefault(key,common)!=common:raise RuntimeError('B tokens/noise or teacher differs across conditions')
        if cfg['context']['mode']!='none':
            layout={k:info[k] for k in ('third_tokens','third_padded_length','third_b_start')}
            if layouts.setdefault(key,layout)!=layout:raise RuntimeError('context layout differs between B/C/D')
    write_json(campaign/'context_comparability_audit.json',{'same_B_tokens_noise_teacher':True,'same_context_layout':True,'runs':len(records)})


def render(campaign):
    state=read_json(campaign/'state.json')
    lines=['# Surrounding-context asymmetric distillation','',f"Stage: {state['stage']}",
        'Teacher: clean B only. Student: two solo B views plus a weighted third view.',
        'All arms: content-only B pooling; covariance on solo view1 once/step.',
        'Native wiki1m scores are historical references only; comparisons use the same new Wikipedia centers.',
        '', '| combination | A dev best | B dev best | C dev best | B−C (pp) |','|---|---|---|---|---|']
    pairs=read_json(campaign/'pairs.json') if (campaign/'pairs.json').exists() else {}
    for n,runs in pairs.items():
        scores={k:r['metrics']['best_sts']*100 for k,r in runs.items()}
        delta=scores['B']-scores['C'] if set(('B','C'))<=scores.keys() else ''
        lines.append(f"| {n} | {scores.get('A','')} | {scores.get('B','')} | {scores.get('C','')} | {delta} |")
    if (campaign/'final_runs.json').exists():
        groups=read_json(campaign/'final_runs.json')
        lines+=['','| group | seed | dev best | last | wall seconds | student / teacher nonpad tokens | total padded tokens | peak memory bytes |','|---|---|---|---|---|---|---|---|']
        for group,seeds in groups.items():
            for seed,r in seeds.items():
                m=r['metrics'];lines.append(f"| {group} | {seed} | {m['best_sts']*100:.4f} | {m['last_sts']*100:.4f} | {m.get('wall_seconds','')} | {m.get('student_nonpad_tokens','')} / {m.get('teacher_nonpad_tokens','')} | {m.get('total_padded_tokens','')} | {m.get('peak_cuda_memory_bytes','')} |")
            if len(seeds)==3:
                for metric in ('best_sts','last_sts'):
                    values=[r['metrics'][metric]*100 for r in seeds.values()]
                    lines.append(f"\n{group} {metric}: {statistics.mean(values):.4f} ± {statistics.stdev(values):.4f} (sample SD).")
        lines+=['','| group | seed | best alignment / uniformity / rank | last CE solo1 / solo2 / third | weighted mean / cov |',
                '|---|---|---|---|---|']
        for group,seeds in groups.items():
            for seed,r in seeds.items():
                log=campaign/'results/logs'/r['run_name']
                best=next((e for e in read_records(log/'evaluations.jsonl') if e['loop_step']==r['metrics'].get('best_loop_step')),{})
                components=read_records(log/'components.jsonl');last=components[-1] if components else {}
                geometry=' / '.join(str(best.get(k,'')) for k in ('alignment','uniformity','eff_rank'))
                ce=' / '.join(str(last.get(f'L_DINO_view_{i}','')) for i in (1,2,3))
                lines.append(f"| {group} | {seed} | {geometry} | {ce} | {last.get('weighted_mean','')} / {last.get('weighted_cov','')} |")
    aggregates={}
    for file in sorted((campaign/'results/analysis').glob('*.json')):
        doc=read_json(file)
        if 'experiment' in doc:
            ex=doc['experiment'];scores=next(iter(doc['runs'].values()))
            aggregates.setdefault((ex['group'],ex['which']),{})[str(ex['seed'])]=scores['avg_7task']
            lines.append(f"\n{file.stem}: {scores['avg_7task']*100:.4f}. Individual tasks: `{file}`.")
    for key,values in aggregates.items():
        if len(values)==3:
            vals=[v*100 for v in values.values()]
            lines.append(f"\nSTS7 {key}: {statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f} (sample SD).")
    for control in ('A','C','C_matched_B','D'):
        b=aggregates.get(('B','best'),{});other=aggregates.get((control,'best'),{})
        if len(b)==len(other)==3:
            delta=[b[str(s)]-other[str(s)] for s in SEEDS]
            verdict='성립' if statistics.mean(delta)>=.003 and sum(d>0 for d in delta)>=2 else '기각' if statistics.mean(delta)<=0 and sum(d<=0 for d in delta)>=2 else '판정불가'
            lines.append(f"\nB vs {control}: {verdict}, paired Δ={statistics.mean(delta)*100:.4f} pp. C_matched_B is the causal comparison.")
    lines+=['','Context diagnostic JSON files are separate; attention visualization is not used as evidence.',
            'Missing results are unexecuted, not estimated. All checkpoint selection uses solo-sentence dev STS.']
    if state.get('error'):lines.append('Error: '+state['error'])
    text='\n'.join(lines)+'\n';(campaign/'report.md').write_text(text)
    Path(read_json(campaign/'source_manifest.json')['report']).write_text(text)


def stage(campaign,name,**extra):
    write_json(campaign/'state.json',{'stage':name,'updated_unix':time.time(),**extra});render(campaign)


def run(campaign,manifest):
    verify_source(campaign,manifest)
    base=p.load_config(campaign/'source/configs/context_distill/base.yaml')
    env=dict(os.environ,PYTHONPATH=str(campaign/'source'),PYTHONDONTWRITEBYTECODE='1')
    children=p.Children(campaign,env);stopped=False
    p.render,p.set_stage=render,stage
    def stop(*_):
        nonlocal stopped
        stopped=True;children.stop()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    calibration={}
    write_json(campaign/'calibration.json',calibration)
    anchor={'backbone_lr':base['train']['lr'],'head_lr':base['train']['head_lr'],'beta':1.,'weak_t_max':.1}
    records=[]
    def train(gpu,arm,params,label,seed=42,smoke=False):
        return p.execute_train(campaign,manifest,children,gpu,configuration(base,params,arm),label,seed,smoke)
    try:
        stage(campaign,'smoke')
        smokes={}
        p.parallel_jobs(['A','B','C'],manifest['gpus'],children,
            lambda gpu,arm:train(gpu,arm,anchor,f'smoke_{arm}',smoke=True),lambda arm,r:smokes.update({arm:r}))
        audit(campaign,list(smokes.values()),calibration);write_json(campaign/'smoke.json',smokes)
        study=optuna.create_study(study_name=campaign.name,storage=f"sqlite:///{campaign/'paired.db'}",direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=20261111,n_startup_trials=4,multivariate=True),pruner=optuna.pruners.NopPruner())
        study.enqueue_trial(anchor);pairs={}
        for n in range(12):
            trial=study.ask();params=parameters(trial);pairs[str(n)]={}
            stage(campaign,'paired_search',combination=n,completed_combinations=n)
            def completed(arm,r):
                pairs[str(n)][arm]=r;records.append(r)
                write_json(campaign/'pairs.json',pairs);audit(campaign,records,calibration);render(campaign)
            p.parallel_jobs(['A','B','C'],manifest['gpus'],children,
                lambda gpu,arm:train(gpu,arm,params,f'search_{arm}_t{n}'),completed)
            trial.set_user_attr('runs',pairs[str(n)])
            study.tell(trial,statistics.mean(r['metrics']['best_sts'] for r in pairs[str(n)].values()))
        shortlist={arm:sorted(range(12),key=lambda n:(-pairs[str(n)][arm]['metrics']['best_sts'],n))[:3] for arm in ('A','B','C')}
        write_json(campaign/'shortlist.json',{'trials':shortlist,'seeds':SEEDS,'test_seen':False})
        confirmed={arm:{str(n):{} for n in shortlist[arm]} for arm in shortlist}
        jobs=[(arm,n,s) for arm in shortlist for n in shortlist[arm] for s in SEEDS]
        stage(campaign,'confirmation')
        def confirm(gpu,job):
            arm,n,seed=job
            return train(gpu,arm,study.trials[n].params,f'confirm_{arm}_t{n}',seed)
        def saved(job,r):
            arm,n,s=job;confirmed[arm][str(n)][str(s)]=r;records.append(r)
            write_json(campaign/'confirmations.json',confirmed);audit(campaign,records,calibration)
        p.parallel_jobs(jobs,manifest['gpus'],children,confirm,saved)
        selected={arm:min(shortlist[arm],key=lambda n:(-statistics.mean(r['metrics']['best_sts'] for r in confirmed[arm][str(n)].values()),n)) for arm in shortlist}
        groups={arm:confirmed[arm][str(n)] for arm,n in selected.items()}
        write_json(campaign/'selection.json',{'selected_trials':selected,'selected_params':{arm:study.trials[n].params for arm,n in selected.items()},
            'selection_frozen_unix':time.time(),'test_scores_seen':False,'selection_metric':'new three-seed solo 768d STS-B dev best mean'})
        write_json(campaign/'final_runs.json',groups)
        # Exact-parameter C control for the selected B configuration.
        bparams=study.trials[selected['B']].params
        if selected['B'] in shortlist['C']:
            groups['C_matched_B']=confirmed['C'][str(selected['B'])]
        else:
            stage(campaign,'matched_unrelated_control');groups['C_matched_B']={}
            p.parallel_jobs(SEEDS,manifest['gpus'],children,
                lambda gpu,s:train(gpu,'C',bparams,'C_matched_B',s),lambda s,r:groups['C_matched_B'].update({str(s):r}))
            records.extend(groups['C_matched_B'].values())
        deltas=[groups['B'][str(s)]['metrics']['best_sts']-groups['C_matched_B'][str(s)]['metrics']['best_sts'] for s in SEEDS]
        promising=statistics.mean(deltas)>=.003 and sum(d>0 for d in deltas)>=2
        write_json(campaign/'far_context_decision.json',{'before_test':True,'dev_deltas_vs_matched_C':deltas,'run_far_control':promising,'min_sentence_distance':8})
        if promising:
            stage(campaign,'far_context_control');groups['D']={}
            p.parallel_jobs(SEEDS,manifest['gpus'],children,
                lambda gpu,s:train(gpu,'D',bparams,'D_far',s),lambda s,r:groups['D'].update({str(s):r}))
            records.extend(groups['D'].values())
        write_json(campaign/'final_runs.json',groups);audit(campaign,records,calibration)
        stage(campaign,'test_evaluation');p.evaluate(campaign,manifest,groups,children)
        stage(campaign,'context_diagnostics')
        def diagnose(gpu,seed):
            run=groups['B'][str(seed)]
            children.run(gpu,f'context_diagnostics_s{seed}',[sys.executable,'scripts/diagnose_context.py',
                '--checkpoint',str(campaign/'checkpoints'/run['run_name']/'best.pt'),'--config',run['config'],
                '--samples','64','--out',str(campaign/'results/analysis'/f'context_diagnostics_s{seed}.json')])
        p.parallel_jobs(SEEDS,manifest['gpus'],children,diagnose,lambda *_:None)
        stage(campaign,'complete')
    except BaseException as exc:
        children.stop();stage(campaign,'stopped_by_user' if stopped else 'failed',error=str(exc));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args();c=args.campaign.resolve()
    if args.execute:run(c,read_json(c/'source_manifest.json'));return
    base=p.load_config(ROOT/'configs/context_distill/base.yaml')
    data=ROOT/base['data']['context_manifest_path']
    if not data.exists():parser.error('prepare documented source triples first; wiki1m row adjacency is forbidden')
    cache=read_json(data)
    for name,h in cache['files'].items():
        if sha256_file(data.parent/name)!=h:raise RuntimeError('prepared cache hash mismatch')
    if sha256_file(data.parent/'embed_stats_bert_base_uncased.pt')!=cache['statistics']['file_sha256']:raise RuntimeError('statistics changed')
    manifest=prepare(c,ROOT,['auto','auto'],ROOT/'docs/experiments/context_distill/report.md',experiment_dir='context_distill')
    manifest.update(protocol='context_distill',reference_groups=[],trials_per_arm=12,launch_argv=sys.argv,common_space=SPACE,
                    data_manifest_sha256=sha256_file(data),data_counts=cache['counts'])
    write_json(c/'source_manifest.json',manifest)
    write_json(c/'search_registration.json',{'space':SPACE,'paired_combinations':12,'arms':['A','B','C'],'steps':4000,
        'search_seed':42,'new_seeds':SEEDS,'sampler_seed':20261111,'objective':'mean A/B/C dev best; same parameter points in all arms',
        'active_axes':{'A':3,'B':4,'C':4},'registration_unix':time.time(),'test_scores_seen':False})
    os.execv(sys.executable,[sys.executable,str(c/'source/scripts/run_context_distill.py'),'--campaign',str(c),'--execute'])

if __name__=='__main__':main()
