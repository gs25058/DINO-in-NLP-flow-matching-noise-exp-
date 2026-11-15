"""Context inputs, independent sentence-noise streams and center-only pooling.

All experimental arms share this code. Inference always sees B alone. The
teacher sees clean B alone unless ``teacher_mode`` gives it clean context.
"""
from dataclasses import dataclass
import hashlib

import torch


@dataclass
class ContextBatch:
    input_ids: torch.Tensor
    sentence_span_mask: torch.Tensor  # -1 special/pad, 0=A, 1=B, 2=C
    center_pooling_mask: torch.Tensor
    special_tokens_mask: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor

    def blocked_attention(self):
        # Block BOTH directions between B and every non-B token at EVERY layer.
        # CLS/SEP cannot act as a relay. Padding queries may read non-B keys only.
        center=self.center_pooling_mask.bool()
        allowed=(center[:,:,None]==center[:,None,:]) & self.attention_mask[:,None,:].bool()
        return allowed


def pack(rows,tokenizer,mode,device,side_budget=48,total_budget=226):
    if mode not in ('none','real','unrelated','far'):raise ValueError('unknown context mode')
    sequences=[];spans=[]
    for row in rows:
        b=row['center_ids']
        if not 1<=len(b)<=126:raise ValueError('B token budget exceeded')
        if mode=='none':a,c=[],[]
        else:
            key='' if mode=='real' else mode+'_'
            a,c=row[key+'left_ids'],row[key+'right_ids']
        if len(a)>side_budget or len(c)>side_budget:raise ValueError('context budget exceeded')
        if mode=='none':
            ids=[tokenizer.cls_token_id]+b+[tokenizer.sep_token_id]
            labels=[-1]+[1]*len(b)+[-1]
        else:
            ids=[tokenizer.cls_token_id]+a+[tokenizer.sep_token_id]+b+[tokenizer.sep_token_id]+c+[tokenizer.sep_token_id]
            labels=[-1]+[0]*len(a)+[-1]+[1]*len(b)+[-1]+[2]*len(c)+[-1]
        sequences.append(ids);spans.append(labels)
    length=max(map(len,sequences))
    if length>total_budget:raise ValueError('context exceeds preregistered total budget')
    ids=torch.full((len(rows),length),tokenizer.pad_token_id,dtype=torch.long,device=device)
    span=torch.full_like(ids,-1);attention=torch.zeros_like(ids)
    for i,(seq,labels) in enumerate(zip(sequences,spans)):
        ids[i,:len(seq)]=torch.tensor(seq,device=device)
        span[i,:len(seq)]=torch.tensor(labels,device=device);attention[i,:len(seq)]=1
    return ContextBatch(ids,span,span==1,span<0,attention,torch.zeros_like(ids))


def stream(seed,step,view,part,device):
    key=f'context-noise-v1:{seed}:{step}:{view}:{part}'
    number=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(2**63-1)
    return torch.Generator(device=device).manual_seed(number)


def sentence_noise(embeds,batch,span_id,mu,sigma,lo,hi,generator,budget):
    """One t per sentence; fixed-budget epsilon keeps B noise arm/length-independent."""
    n,_,dim=embeds.shape
    t=lo+(hi-lo)*torch.rand(n,device=embeds.device,dtype=embeds.dtype,generator=generator)
    eps=torch.randn(n,budget,dim,device=embeds.device,dtype=embeds.dtype,generator=generator)
    result=embeds.clone()
    mu,sigma=mu.to(embeds),sigma.to(embeds)
    for i in range(n):
        mask=batch.sentence_span_mask[i]==span_id
        x=embeds[i,mask];k=len(x)
        if k>budget:raise ValueError('sentence noise budget exceeded')
        standardized=(x-mu)/sigma
        result[i,mask]=((1-t[i])*standardized+t[i]*eps[i,:k])*sigma+mu
    return result


class ContextViews:
    def __init__(self,cfg,tokenizer,augment,seed,device):
        self.cfg,self.tokenizer,self.augment,self.seed,self.device=cfg,tokenizer,augment,seed,device
        if cfg['mode'] not in ('none','real','unrelated','far'):raise ValueError('invalid context mode')
        self.teacher_mode=cfg.get('teacher_mode','none')
        if self.teacher_mode not in ('none','real','unrelated'):raise ValueError('invalid teacher context mode')
        if self.teacher_mode!='none' and cfg['mode']!='none':raise ValueError('context goes to the teacher or the student, not both')
        self.side_budget=cfg.get('side_budget',48);self.total_budget=cfg.get('total_budget',226)
        if not 0<=cfg['weak_t_max']<augment.t_lo:raise ValueError('context noise must be weaker than B')

    def forward(self,student,teacher,rows,step,mode=None,blocked=False,noise_bounds=None):
        solo=pack(rows,self.tokenizer,'none',self.device)
        context=pack(rows,self.tokenizer,mode or self.cfg['mode'],self.device,self.side_budget,self.total_budget)
        center=student.get_input_embeddings()(solo.input_ids)
        # Clean teacher; with teacher_mode it reads B inside clean context, pooling B only.
        tbatch=solo if self.teacher_mode=='none' else pack(rows,self.tokenizer,self.teacher_mode,self.device,self.side_budget,self.total_budget)
        with torch.no_grad():
            teacher_embeds=teacher.model.get_input_embeddings()(tbatch.input_ids)
            target=teacher(inputs_embeds=teacher_embeds,attention_mask=tbatch.attention_mask,
                           pooling_mask=tbatch.center_pooling_mask,token_type_ids=tbatch.token_type_ids)
        lo,hi=noise_bounds if noise_bounds is not None else (self.augment.t_lo,self.augment.t_hi(step))
        views=[]
        center_hashes=[]
        for view in range(3):
            noisy=sentence_noise(center,solo,1,self.augment.mu,self.augment.sigma,lo,hi,
                stream(self.seed,step,view,'B',self.device),126)
            if step == 0:
                center_hashes.append(hashlib.sha256(noisy[solo.center_pooling_mask].detach().float().cpu().numpy().tobytes()).hexdigest())
            if view<2 or (mode or self.cfg['mode'])=='none':
                views.append((noisy,solo,False))
            else:
                embeds=student.get_input_embeddings()(context.input_ids)
                embeds=sentence_noise(embeds,context,0,self.augment.mu,self.augment.sigma,0.,self.cfg['weak_t_max'],
                    stream(self.seed,step,view,'A',self.device),48)
                embeds=sentence_noise(embeds,context,2,self.augment.mu,self.augment.sigma,0.,self.cfg['weak_t_max'],
                    stream(self.seed,step,view,'C',self.device),48)
                # Copy the SAME noised B token vectors into a different input layout.
                embeds=embeds.clone();embeds[context.center_pooling_mask]=noisy[solo.center_pooling_mask]
                views.append((embeds,context,blocked))
        # Two solo views share a pass; the longer third view uses the same model.
        first=student(inputs_embeds=torch.cat([v[0] for v in views[:2]]),
            attention_mask=torch.cat([solo.attention_mask]*2),pooling_mask=torch.cat([solo.center_pooling_mask]*2),
            token_type_ids=torch.cat([solo.token_type_ids]*2))
        embeddings=list(first.embedding.chunk(2));logits=list(first.logits.chunk(2));bottlenecks=list(first.bottleneck.chunk(2))
        embeds,batch,block=views[2]
        attention=batch.blocked_attention() if block else batch.attention_mask
        third=student(inputs_embeds=embeds,attention_mask=attention,pooling_mask=batch.center_pooling_mask,
                      token_type_ids=batch.token_type_ids)
        embeddings.append(third.embedding);logits.append(third.logits);bottlenecks.append(third.bottleneck)
        info={'center_noise_sha256':center_hashes,'center_token_sha256':hashlib.sha256(solo.input_ids.cpu().numpy().tobytes()).hexdigest(),
              'solo_tokens':int(solo.attention_mask.sum()),'third_tokens':int(batch.attention_mask.sum()),
              'solo_padded_length':solo.input_ids.shape[1],
              'third_padded_length':batch.input_ids.shape[1],'third_b_start':[int((s==1).nonzero()[0]) for s in batch.sentence_span_mask],
              'teacher_clean_sha256':hashlib.sha256(teacher_embeds.float().cpu().numpy().tobytes()).hexdigest() if step == 0 else None,
              'teacher_tokens':int(tbatch.attention_mask.sum()),'teacher_padded_length':tbatch.input_ids.shape[1]}
        return target,tuple(embeddings),logits,tuple(bottlenecks),info
