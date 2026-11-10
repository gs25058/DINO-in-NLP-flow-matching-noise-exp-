"""Held-out context interventions; never used for model/checkpoint selection."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.eval_sts7 import load_eval_model
from src.model import EMATeacher
from src.augment import FlowNoiseAug
from src.context import ContextViews


@torch.no_grad()
def diagnose(model,teacher,tokenizer,rows,context_cfg,stats,center,student_temp,teacher_temp,seed,device):
    aug=FlowNoiseAug(stats['mean'],stats['std'],num_student_views=3,t_lo=.3919166650119331,
                     t_start=.4638073616095715,t_max=.8811226221985871,warmup_steps=100)
    gen=ContextViews(context_cfg,tokenizer,aug,seed,device)
    result=[];swaps=[]
    for strength in (.05,.4,.8):
        values={key:[] for key in ('real','unrelated','none','blocked')}
        changes=[]
        for j in range(0,len(rows),8):
            batch=rows[j:j+8]
            actual=None
            for mode in values:
                target,embeds,logits,_,_=gen.forward(model,teacher,batch,j+1,
                    mode='real' if mode=='blocked' else mode,blocked=mode=='blocked',noise_bounds=(strength,strength))
                p=F.softmax((target.logits-center)/teacher_temp,dim=-1)
                log_p=p.clamp_min(1e-12).log();log_s=F.log_softmax(logits[2]/student_temp,dim=-1)
                ce=-(p*log_s).sum(-1)
                kl=(p*(log_p-log_s)).sum(-1)
                cosine=(target.embedding*embeds[2]).sum(-1)
                values[mode].extend(zip(ce.cpu().tolist(),kl.cpu().tolist(),cosine.cpu().tolist()))
                if mode=='real':actual=(target.embedding,embeds[2])
            replaced=[];matched_original=[]
            for row in batch:
                candidate=next((r for r in rows if r['doc_id']!=row['doc_id'] and len(r['center_ids'])>=len(row['center_ids'])),None)
                if candidate is None:continue
                r=copy.deepcopy(row);r['center_ids']=candidate['center_ids'][:len(row['center_ids'])]
                replaced.append(r);matched_original.append(row)
            if replaced:
                original_target,original_embeddings,_,_,_=gen.forward(model,teacher,matched_original,j+1,mode='real',noise_bounds=(strength,strength))
                target,embeds,_,_,_=gen.forward(model,teacher,replaced,j+1,mode='real',noise_bounds=(strength,strength))
                changes.extend(zip((1-(original_target.embedding*target.embedding).sum(-1)).cpu().tolist(),
                                   (1-(original_embeddings[2]*embeds[2]).sum(-1)).cpu().tolist()))
        for mode,items in values.items():
            tensor=torch.tensor(items)
            result.append({'center_noise_t':strength,'mode':mode,'samples':len(items),
                'mean_CE':tensor[:,0].mean().item(),'mean_KL':tensor[:,1].mean().item(),
                'mean_cosine_to_clean_teacher':tensor[:,2].mean().item()})
        if changes:
            tensor=torch.tensor(changes)
            swaps.append({'center_noise_t':strength,'samples':len(changes),
                'clean_teacher_one_minus_cosine_after_B_swap':tensor[:,0].mean().item(),
                'context_student_one_minus_cosine_after_B_swap':tensor[:,1].mean().item()})
    return {'noise_interventions':result,'center_swap':swaps,
            'note':'Same A/C and equal center token length during B swap; replacement is cropped from a different held-out document. Diagnostics are not STS evidence.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--samples',type=int,default=64)
    args=p.parse_args()
    from src.train import load_config
    cfg=load_config(args.config)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=load_eval_model(ckpt,args.device);teacher=EMATeacher(model,.99)
    teacher.model.load_state_dict(ckpt['teacher_state_dict'])
    tokenizer=AutoTokenizer.from_pretrained(cfg['model']['backbone'])
    rows=[];per_doc={}
    for line in Path(cfg['data']['context_heldout_path']).open():
        r=json.loads(line)
        if per_doc.get(r['doc_id'],0)>=2:continue
        rows.append(r);per_doc[r['doc_id']]=per_doc.get(r['doc_id'],0)+1
        if len(rows)>=args.samples:break
    stats=torch.load(cfg['data']['embed_stats_path'],map_location='cpu',weights_only=True)
    step=ckpt['loop_step'];loss=cfg['loss'];train=cfg['train']
    from src.train import cosine_ramp
    temp=cosine_ramp(step,loss['warmup_teacher_temp'],loss['teacher_temp'],max(1,int(loss['teacher_temp_warmup_frac']*train['max_steps'])))
    result=diagnose(model,teacher,tokenizer,rows,cfg['context'],stats,ckpt['dino_state_dict']['center'].to(args.device),
                    loss['student_temp'],temp,cfg['seed'],args.device)
    result.update(checkpoint=str(args.checkpoint),heldout_documents=sorted({r['doc_id'] for r in rows}),seed=cfg['seed'],
                  attention_block='All B/non-B edges blocked in both directions at every layer, including CLS/SEP relays.')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
