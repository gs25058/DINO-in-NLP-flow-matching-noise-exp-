import copy
import os
import sys
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import BertConfig,BertModel

from src.context import ContextViews,pack,sentence_noise,stream
from src.context_data import triples_from_document,split_document,add_unrelated_context,validate_row
from src.augment import FlowNoiseAug
from src.loss import DINOLoss,DINOObjective
from src.model import DinoTextModel,DINOHead,EMATeacher,masked_mean_pool


class Tokenizer:
    cls_token_id=1;sep_token_id=2;pad_token_id=0
    def __call__(self,texts,**kwargs):
        return {'input_ids':[[3+(sum(map(ord,w))%90) for w in t.split()] for t in texts]}


def row(doc='1'):
    return {'doc_id':doc,'unrelated_doc_id':'other','A':'Left context words.', 'B':'The central sentence.',
        'C':'Right context words.','unrelated_texts':['Different left context.','Different right context.'],
        'unrelated_center_text':'Another center.', 'doc_sentence_index':10,'far_sentence_indices':[0,20],
        'center_ids':[11,12,13,14],'left_ids':[21,22],'right_ids':[31,32,33],
        'unrelated_left_ids':[41,42],'unrelated_right_ids':[51,52,53],
        'far_left_ids':[61,62],'far_right_ids':[71,72,73]}


def tiny_model(attention="eager"):
    # Actual two-layer BERT attention, not an attention-free linear stand-in.
    torch.manual_seed(42)
    m=DinoTextModel.__new__(DinoTextModel);nn.Module.__init__(m)
    cfg=BertConfig(vocab_size=100,hidden_size=16,num_hidden_layers=2,num_attention_heads=2,
                   intermediate_size=32,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
    cfg._attn_implementation=attention
    m.backbone=BertModel(cfg);m.head=DINOHead(16,8,32);m.pooling_policy='content'
    return m.eval()


def objective_cfg(beta=1.):
    return {'center_momentum':.92,'cov_ema_momentum':.82,'cov_iso_start_step':0,
            'cov_iso_mean_lambda':5.87,'cov_iso_cov_lambda':5.87,'student_temp':.15,
            'view_weights':[1.,1.,beta],'cov_iso_view':0}


def test_document_offsets_paragraph_adjacency_and_ambiguous_structure_rejection():
    sentences=[f'This sentence number {i} contains distinct words.' for i in range(20)]
    text='Title\n'+' '.join(sentences)+'\n\n'+' '.join(reversed(sentences))+'\n* A list item.\nDr. Smith is here. Another sentence. Third sentence.\n'
    doc={'id':'doc','text':text}
    triples=triples_from_document(doc,Tokenizer())
    assert triples
    for t in triples:
        a,b,c=t['source_positions']
        assert text[a[0]:a[1]]==t['A'] and text[b[0]:b[1]]==t['B'] and text[c[0]:c[1]]==t['C']
        assert '\n' not in text[a[0]:c[1]]
        assert t['sentence_index']>=1
        assert all(abs(i-t['doc_sentence_index'])>=8 for i in t['far_sentence_indices'])
        assert 'Smith' not in t['B']
    assert not triples_from_document({'id':'short','text':'One sentence here.\nAnother sentence here.\nThird sentence here.'},Tokenizer())


def test_unrelated_exact_lengths_other_documents_and_duplicate_rejection():
    r=row();r['source_positions']=[[0,1],[2,3],[4,5]]
    other=copy.deepcopy(r);other.update(doc_id='2',A='Different left text.',B='Different center text.',C='Different right text.')
    matched=add_unrelated_context([r,other])
    assert len(matched)==2
    for t in matched:validate_row(t)
    bad=copy.deepcopy(matched[0]);bad['unrelated_doc_id']=bad['doc_id']
    with pytest.raises(ValueError):validate_row(bad)
    bad=copy.deepcopy(matched[0]);bad['unrelated_texts'][0]=bad['B']
    with pytest.raises(ValueError):validate_row(bad)


def test_pack_masks_center_budget_positions_and_no_third_segment():
    r=row();short=copy.deepcopy(r);short['center_ids']=[15,16]
    a=pack([r,short],Tokenizer(),'real','cpu');c=pack([r,short],Tokenizer(),'unrelated','cpu')
    solo=pack([r,short],Tokenizer(),'none','cpu')
    assert torch.equal(a.sentence_span_mask,c.sentence_span_mask)
    assert torch.equal(a.attention_mask,c.attention_mask)
    assert not a.token_type_ids.any()
    assert not (a.center_pooling_mask & a.special_tokens_mask).any()
    assert torch.equal(a.input_ids[a.center_pooling_mask],solo.input_ids[solo.center_pooling_mask])
    hidden=torch.zeros(2,a.input_ids.shape[1],3);hidden[a.center_pooling_mask]=2
    hidden[~a.center_pooling_mask]=100
    assert torch.equal(masked_mean_pool(hidden,a.center_pooling_mask),torch.full((2,3),2.))
    with pytest.raises(ValueError):pack([{**r,'center_ids':[3]*127}],Tokenizer(),'real','cpu')


@pytest.mark.parametrize("attention",["eager","sdpa"])
def test_attention_changes_center_and_blocking_closes_all_multilayer_relays(attention):
    m=tiny_model(attention);a=pack([row()],Tokenizer(),'real','cpu');b=pack([row()],Tokenizer(),'unrelated','cpu')
    def encode(batch,block):
        return m(inputs_embeds=m.get_input_embeddings()(batch.input_ids),
                 attention_mask=batch.blocked_attention() if block else batch.attention_mask,
                 pooling_mask=batch.center_pooling_mask,token_type_ids=batch.token_type_ids).embedding
    with torch.no_grad():
        assert not torch.allclose(encode(a,False),encode(b,False),atol=1e-7,rtol=0)
        torch.testing.assert_close(encode(a,True),encode(b,True),atol=1e-7,rtol=0)
    center=a.center_pooling_mask[0];mask=a.blocked_attention()[0]
    assert not mask[center][:,~center].any() and not mask[~center][:,center].any()


def test_sentence_noise_special_pad_exclusion_matching_B_and_context_independence():
    torch.manual_seed(7);m=tiny_model();teacher=EMATeacher(m,.99)
    aug=FlowNoiseAug(torch.zeros(16),torch.ones(16),num_student_views=3,t_lo=.39,t_start=.46,t_max=.88,warmup_steps=100)
    cfg={'mode':'real','weak_t_max':.15}
    gen=ContextViews(cfg,Tokenizer(),aug,42,'cpu')
    state=torch.get_rng_state()
    none=gen.forward(m,teacher,[row()],0,mode='none')
    real=gen.forward(m,teacher,[row()],0,mode='real')
    unrelated=gen.forward(m,teacher,[row()],0,mode='unrelated')
    assert torch.equal(state,torch.get_rng_state())
    assert none[-1]['center_noise_sha256']==real[-1]['center_noise_sha256']==unrelated[-1]['center_noise_sha256']
    assert real[-1]['third_b_start']==unrelated[-1]['third_b_start']
    assert real[-1]['third_tokens']==unrelated[-1]['third_tokens']
    assert torch.equal(none[0].logits,real[0].logits) and not real[0].logits.requires_grad
    for view in (0,1):assert torch.equal(none[2][view],real[2][view])
    b=pack([row()],Tokenizer(),'real','cpu');emb=m.get_input_embeddings()(b.input_ids)
    noisy=sentence_noise(emb,b,1,torch.zeros(16),torch.ones(16),.8,.8,stream(42,0,0,'B','cpu'),126)
    assert torch.equal(noisy[b.special_tokens_mask],emb[b.special_tokens_mask])
    assert torch.equal(noisy[b.sentence_span_mask==0],emb[b.sentence_span_mask==0])
    real[2][-1].square().mean().backward()
    assert m.backbone.embeddings.word_embeddings.weight.grad is not None
    assert m.head.mlp[0].weight.grad is not None
    assert all(p.grad is None for p in teacher.model.parameters())


def test_weighted_CE_previous_center_and_one_covariance_update(monkeypatch):
    torch.manual_seed(8)
    t=torch.randn(4,32,requires_grad=True);ss=[torch.randn(4,32,requires_grad=True) for _ in range(3)]
    z=[torch.nn.functional.normalize(torch.randn(4,16),dim=-1).requires_grad_() for _ in range(3)]
    obj=DINOObjective(objective_cfg(2.),16,32,8)
    updates=[];old=obj.covariance._update_ema
    monkeypatch.setattr(obj.covariance,'_update_ema',lambda *args:(updates.append(1),old(*args))[-1])
    target=torch.softmax(t.detach()/.1,dim=-1)
    ces=[-(target*torch.log_softmax(x/.15,dim=-1)).sum(-1).mean() for x in ss]
    loss=obj(t,z,ss,0,.1)
    torch.testing.assert_close(loss.dino,(ces[0]+ces[1]+2*ces[2])/4)
    torch.testing.assert_close(obj.dino.center,t.detach().mean(0)*.08)
    assert len(updates)==1
    loss.total.backward()
    assert t.grad is None and z[0].grad is not None and z[1].grad is None and z[2].grad is None
    old_teacher=EMATeacher(tiny_model(),.99);student=tiny_model()
    before=old_teacher.model.backbone.embeddings.word_embeddings.weight.clone()
    with torch.no_grad():student.backbone.embeddings.word_embeddings.weight.add_(1)
    old_teacher.update(student)
    torch.testing.assert_close(old_teacher.model.backbone.embeddings.word_embeddings.weight,
                              before*.99+student.backbone.embeddings.word_embeddings.weight*.01)


def test_document_splits_and_cached_positions_are_auditable(tmp_path):
    from src.context_data import prepare_cache,document_split
    docs=[];counts={k:0 for k in ('train','stats','heldout')}
    for i in range(1000):
        key=document_split(str(i))
        if counts[key]>=2:continue
        counts[key]+=1
        docs.append({'id':str(i),'text':' '.join(f'Document {i} sentence {j} has distinct central content.' for j in range(20))})
        if all(n==2 for n in counts.values()):break
    output=tmp_path/'cache'
    manifest=prepare_cache(docs,Tokenizer(),output,{'shard':'fixture.parquet','revision':'fixture'},max_documents=len(docs))
    sets={}
    for split in ('train','stats','heldout'):
        rows=[json.loads(x) for x in (output/f'{split}.jsonl').read_text().splitlines()]
        sets[split]={r['doc_id'] for r in rows}
        for r in rows:
            raw=next(d['text'] for d in docs if d['id']==r['doc_id'])
            assert [raw[a:b] for a,b in r['source_positions']]==[r['A'],r['B'],r['C']]
            validate_row(r)
    assert not sets['train']&sets['stats'] and not sets['train']&sets['heldout'] and not sets['stats']&sets['heldout']
    assert manifest['counts']['train']>0


def test_checkpoint_restore_content_pooling_solo_eval_and_single_optimizer_EMA(tmp_path,monkeypatch):
    import logging
    from src.train import Trainer,TrainingStep
    from scripts import eval_sts7
    from src.evaluate import encode_sentences
    from transformers import BatchEncoding
    trainer=Trainer.__new__(Trainer);trainer.student=tiny_model();trainer.teacher=EMATeacher(trainer.student,.997)
    trainer.objective=DINOObjective(objective_cfg(),16,32,8)
    trainer.cfg={'model':{'backbone':'tiny','pooling_policy':'content','head':{'bottleneck_dim':8,'logit_dim':32}},
                 'train':{'grad_clip':5.,'momentum_start':.997,'momentum_end':.999},'eval':{},'loss':objective_cfg()}
    trainer.parameters=list(trainer.student.parameters());trainer.optimizer=torch.optim.AdamW(trainer.parameters,lr=1e-3)
    trainer.scheduler=torch.optim.lr_scheduler.LambdaLR(trainer.optimizer,lambda _:1.)
    trainer.gradient_steps=set();trainer.momentum_steps=100
    trainer.best_step=0;trainer.best_sts=.7;trainer.root=tmp_path;trainer.cfg['run_name']='test'
    batch=pack([row()],Tokenizer(),'none','cpu')
    target=trainer.teacher(inputs_embeds=trainer.teacher.model.get_input_embeddings()(batch.input_ids),attention_mask=batch.attention_mask,pooling_mask=batch.center_pooling_mask)
    outputs=[trainer.student(inputs_embeds=trainer.student.get_input_embeddings()(batch.input_ids),attention_mask=batch.attention_mask,pooling_mask=batch.center_pooling_mask) for _ in range(3)]
    calls=[];old=trainer.teacher.update
    monkeypatch.setattr(trainer.teacher,'update',lambda student:(calls.append(1),old(student))[-1])
    terms=trainer.objective(target.logits,[o.embedding for o in outputs],[o.logits for o in outputs],0,.1)
    trainer.optimize(TrainingStep(terms,[o.logits for o in outputs],.1),0)
    assert calls==[1] and all(p.grad is None for p in trainer.teacher.model.parameters())
    trainer.save_checkpoint('best',0,.7)
    ckpt=torch.load(tmp_path/'checkpoints/test/best.pt',weights_only=True)
    restored=DINOObjective(objective_cfg(),16,32,8)
    restored.dino.load_state_dict(ckpt['dino_state_dict']);restored.covariance.load_state_dict(ckpt['cov_iso_state_dict'])
    assert torch.equal(restored.dino.center,trainer.objective.dino.center)
    assert torch.equal(restored.covariance.mu_ema,trainer.objective.covariance.mu_ema)
    monkeypatch.setattr(eval_sts7,'DinoTextModel',lambda *args,**kwargs:tiny_model())
    loaded=eval_sts7.load_eval_model(ckpt,'cpu')
    assert loaded.pooling_policy=='content'
    class EvalTokenizer:
        def __call__(self,texts,**kwargs):
            assert kwargs['return_special_tokens_mask']
            n=len(texts)
            return BatchEncoding({'input_ids':torch.tensor([[1,11,12,2]]*n),
                'attention_mask':torch.ones(n,4,dtype=torch.long),'special_tokens_mask':torch.tensor([[1,0,0,1]]*n)})
    encoded=encode_sentences(loaded,EvalTokenizer(),['B only','Another B'],'cpu')
    assert encoded['embedding'].shape==(2,16)
    with torch.no_grad():
        direct=loaded(inputs_embeds=loaded.get_input_embeddings()(torch.tensor([[1,11,12,2]])),
            attention_mask=torch.ones(1,4),pooling_mask=torch.tensor([[0,1,1,0]])).embedding
    torch.testing.assert_close(encoded['embedding'][:1],direct)


def test_heldout_diagnostics_have_all_interventions_and_do_not_update_model_or_teacher():
    from scripts.diagnose_context import diagnose
    m=tiny_model();teacher=EMATeacher(m,.99)
    a=row();b=row('2');b['center_ids']=[15,16,17,18]
    before={k:v.clone() for k,v in m.state_dict().items()}
    teacher_before={k:v.clone() for k,v in teacher.model.state_dict().items()}
    result=diagnose(m,teacher,Tokenizer(),[a,b],{'mode':'real','weak_t_max':.1},
        {'mean':torch.zeros(16),'std':torch.ones(16)},torch.zeros(32),.15,.1,42,'cpu')
    assert len(result['noise_interventions'])==12 and len(result['center_swap'])==3
    assert all(r['samples']==2 for r in result['noise_interventions'])
    assert {r['mode'] for r in result['noise_interventions']}=={'real','unrelated','none','blocked'}
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert all(torch.equal(v,teacher.model.state_dict()[k]) for k,v in teacher_before.items())


def test_paired_context_protocol_uses_same_points_budget_and_freezes_before_test(tmp_path,monkeypatch):
    from scripts import run_context_distill as runner
    from src.provenance import prepare,write_json,read_json
    from src.train import load_config
    p=runner.p;c=tmp_path/'campaign'
    m=prepare(c,p.ROOT,['auto','auto'],tmp_path/'report.md',experiment_dir='context_distill')
    m.update(protocol='context_distill',trials_per_arm=12,common_space=runner.SPACE)
    write_json(c/'source_manifest.json',m)
    calls=[]
    def train(campaign,manifest,children,gpu,cfg,label,seed,smoke=False):
        path,resolved=p.save_config(campaign,manifest,cfg,label,seed)
        calls.append((label,seed,smoke,cfg))
        # Force a promising real-context result to exercise the optional far arm.
        score=.7+(.01 if cfg['context']['mode']=='real' else 0)+seed*1e-6
        log=c/'results/logs'/resolved['run_name'];log.mkdir(parents=True,exist_ok=True)
        is_context=cfg['context']['mode']!='none'
        write_json(log/'context_first_views.json',{
            'center_noise_sha256':[f'noise{seed}']*3,'center_token_sha256':f'B{seed}',
            'teacher_clean_sha256':f'teacher{seed}','third_tokens':20 if is_context else 10,
            'third_padded_length':20 if is_context else 10,'third_b_start':[5] if is_context else [1]})
        return {'run_name':resolved['run_name'],'config':str(path),'seed':seed,'label':label,
            'metrics':{'best_sts':score,'last_sts':score-.001,'completed_steps':60 if smoke else 4000},
            'initial_student_sha256':f'full{seed}','initial_shared_sha256':f'shared{seed}','architecture':8192,
            'first_views':None,'batch_order':{'sha256':f'order{seed}'},'data':{'sentences':{'sha256':'same'}}}
    evaluated=[]
    def evaluate(campaign,manifest,groups,children):
        selection=read_json(campaign/'selection.json')
        assert selection['test_scores_seen'] is False and (campaign/'far_context_decision.json').exists()
        assert set(groups)=={'A','B','C','C_matched_B','D'}
        for seed in ('101','102','103'):
            b=load_config(groups['B'][seed]['config']);cc=load_config(groups['C_matched_B'][seed]['config'])
            for cfg in (b,cc):
                cfg.pop('run_name');cfg.pop('experiment');cfg['context'].pop('mode')
            assert b==cc
        evaluated.append(True)
    monkeypatch.setattr(p,'execute_train',train);monkeypatch.setattr(p,'evaluate',evaluate)
    monkeypatch.setattr(p,'set_stage',p.set_stage);monkeypatch.setattr(p,'render',p.render)
    monkeypatch.setattr(p.Children,'run',lambda self,*args,**kwargs:None)
    monkeypatch.setattr(runner.signal,'signal',lambda *args:None)
    runner.run(c,m)
    assert read_json(c/'state.json')['stage']=='complete' and evaluated
    assert sum(smoke for _,_,smoke,_ in calls)==3
    assert sum(label.startswith('search_') for label,_,_,_ in calls)==36
    assert sum(label.startswith('confirm_') for label,_,_,_ in calls)==27
    pairs=read_json(c/'pairs.json')
    assert len(pairs)==12
    for runs in pairs.values():
        configs=[load_config(runs[arm]['config']) for arm in ('A','B','C')]
        for cfg in configs:
            cfg.pop('run_name');cfg.pop('experiment');cfg['context'].pop('mode')
        assert configs[0]==configs[1]==configs[2]


def test_windows_use_two_or_three_whole_same_paragraph_sentences_per_side():
    from src.context_data import windows_from_document
    sentences=[f'This sentence number {i} contains distinct words.' for i in range(9)]
    text='Title\n'+' '.join(sentences)+'\nShort paragraph one. Short paragraph two. Short paragraph three. Short four.\n'
    rows=windows_from_document({'id':'doc','text':text},Tokenizer())
    assert [r['window_sentences'] for r in rows]==[6,7,7,7,6]
    for r in rows:
        spans=r['source_positions']
        assert len(spans)==r['window_sentences'] and '\n' not in text[spans[0][0]:spans[-1][1]]
        assert ' '.join(text[a:b] for a,b in spans)==' '.join(r['left_texts']+[r['B']]+r['right_texts'])
        assert 2<=len(r['left_texts'])<=3 and 2<=len(r['right_texts'])<=3
        assert len(r['left_ids'])==sum(len(Tokenizer()([t])['input_ids'][0]) for t in r['left_texts'])
        assert 'far_left_ids' not in r
    # A paragraph of 4 sentences cannot give 2 sentences on both sides.
    assert all('Short' not in r['B'] for r in rows)
    # The nearest two sentences must fit whole, otherwise B is dropped (no truncation).
    assert not windows_from_document({'id':'doc','text':text},Tokenizer(),side_budget=13)
    tight=windows_from_document({'id':'doc','text':text},Tokenizer(),side_budget=14)
    assert tight and all(r['window_sentences']==5 for r in tight)


def test_v2_validation_without_far_arm_and_larger_side_budget():
    r=row();r.update(left_ids=list(range(3,103)),unrelated_left_ids=list(range(3,103)))
    for k in ('far_left_ids','far_right_ids','far_sentence_indices'):r.pop(k)
    with pytest.raises(ValueError):validate_row(r)
    validate_row(r,160)
    bad=copy.deepcopy(r);bad['unrelated_right_ids']=bad['unrelated_right_ids'][:-1]
    with pytest.raises(ValueError):validate_row(bad,160)


def test_teacher_context_mode_pools_B_inside_clean_context_and_keeps_student_solo():
    aug=FlowNoiseAug(torch.zeros(16),torch.ones(16),num_student_views=3,t_lo=.39,t_start=.46,t_max=.88,warmup_steps=10)
    rows=[row('1'),row('2')];out={}
    for mode in ('none','real','unrelated'):
        m=tiny_model();t=EMATeacher(m,.99)
        views=ContextViews({'mode':'none','teacher_mode':mode,'weak_t_max':.1,'beta':1.},Tokenizer(),aug,42,'cpu')
        out[mode]=views.forward(m,t,rows,step=3)
    base=out['none']
    for mode in ('real','unrelated'):
        target,embeddings,logits,_,info=out[mode]
        # Student views are identical solo noisy B; only the teacher target moves.
        assert all(torch.equal(a,b) for a,b in zip(embeddings,base[1]))
        assert info['center_noise_sha256']==base[4]['center_noise_sha256']
        assert info['third_padded_length']==info['solo_padded_length']
        assert not torch.allclose(target.logits,base[0].logits)
        assert info['teacher_tokens']>base[4]['teacher_tokens']
    assert out['real'][4]['teacher_tokens']==out['unrelated'][4]['teacher_tokens']
    with pytest.raises(ValueError):
        ContextViews({'mode':'real','teacher_mode':'real','weak_t_max':.1,'beta':1.},Tokenizer(),aug,42,'cpu')
    with pytest.raises(ValueError):
        ContextViews({'mode':'none','teacher_mode':'far','weak_t_max':.1,'beta':1.},Tokenizer(),aug,42,'cpu')


def test_pack_side_and_total_budgets_are_configurable():
    r=row();r.update(left_ids=[5]*100,right_ids=[6]*100)
    with pytest.raises(ValueError):pack([r],Tokenizer(),'real','cpu')
    b=pack([r],Tokenizer(),'real','cpu',side_budget=160,total_budget=450)
    assert b.input_ids.shape[1]==1+100+1+4+1+100+1
    with pytest.raises(ValueError):pack([r],Tokenizer(),'real','cpu',side_budget=160,total_budget=200)


def test_context_teacher_protocol_pairs_arms_by_seed_and_freezes_before_test(tmp_path,monkeypatch):
    from scripts import run_context_teacher as runner
    from src.provenance import prepare,write_json,read_json
    from src.train import load_config
    p=runner.p;c=tmp_path/'campaign'
    m=prepare(c,p.ROOT,['auto','auto'],tmp_path/'report.md',experiment_dir='context_teacher')
    m.update(protocol='context_teacher');write_json(c/'source_manifest.json',m)
    calls=[]
    def train(campaign,manifest,children,gpu,cfg,label,seed,smoke=False):
        path,resolved=p.save_config(campaign,manifest,cfg,label,seed)
        calls.append((label,seed,smoke,cfg))
        mode=cfg['context']['teacher_mode'];score=.7+seed*1e-6
        log=campaign/'results/logs'/resolved['run_name'];log.mkdir(parents=True,exist_ok=True)
        write_json(log/'context_first_views.json',{'center_noise_sha256':[f'noise{seed}']*3,'center_token_sha256':f'B{seed}',
            'third_padded_length':10,'teacher_tokens':10 if mode=='none' else 40,'teacher_padded_length':10 if mode=='none' else 40})
        return {'run_name':resolved['run_name'],'config':str(path),'seed':seed,'label':label,
            'metrics':{'best_sts':score,'last_sts':score-.001,'completed_steps':60 if smoke else 4000},
            'initial_student_sha256':f'full{seed}','initial_shared_sha256':f'shared{seed}','architecture':8192,
            'first_views':None,'batch_order':{'sha256':f'order{seed}'},'data':{'sentences':{'sha256':'same'}}}
    evaluated=[]
    def evaluate(campaign,manifest,groups,children):
        assert read_json(campaign/'selection.json')['test_scores_seen'] is False
        assert set(groups)=={'A','B','C'} and all(set(v)=={'101','102','103'} for v in groups.values())
        evaluated.append(True)
    monkeypatch.setattr(p,'execute_train',train);monkeypatch.setattr(p,'evaluate',evaluate)
    monkeypatch.setattr(p,'set_stage',p.set_stage);monkeypatch.setattr(p,'render',p.render)
    monkeypatch.setattr(runner.signal,'signal',lambda *args:None)
    runner.run(c,m)
    assert read_json(c/'state.json')['stage']=='complete' and evaluated
    assert sum(smoke for _,_,smoke,_ in calls)==3 and len(calls)==12
    full=[(label,seed,cfg) for label,seed,smoke,cfg in calls if not smoke]
    for seed in (101,102,103):
        configs=[cfg for _,s,cfg in full if s==seed];assert len(configs)==3
        modes=sorted(cfg['context']['teacher_mode'] for cfg in configs)
        assert modes==['none','real','unrelated']
        stripped=[copy.deepcopy(cfg) for cfg in configs]
        for cfg in stripped:cfg['context'].pop('teacher_mode')
        assert stripped[0]==stripped[1]==stripped[2]
        assert all(cfg['context']['mode']=='none' for cfg in configs)
    # Mismatched teacher layouts between B and C must stop the campaign.
    bad=tmp_path/'bad';m2=prepare(bad,p.ROOT,['auto','auto'],tmp_path/'r2.md',experiment_dir='context_teacher')
    m2.update(protocol='context_teacher');write_json(bad/'source_manifest.json',m2)
    def train_bad(campaign,manifest,children,gpu,cfg,label,seed,smoke=False):
        r=train(campaign,manifest,children,gpu,cfg,label,seed,smoke)
        if cfg['context']['teacher_mode']=='unrelated':
            f=campaign/'results/logs'/r['run_name']/'context_first_views.json';d=read_json(f);d['teacher_tokens']=39;write_json(f,d)
        return r
    monkeypatch.setattr(p,'execute_train',train_bad)
    with pytest.raises(RuntimeError):runner.run(bad,m2)
    assert read_json(bad/'state.json')['stage']=='failed'


def test_context_teacher_tuned_design_gives_every_arm_an_equal_independent_search(tmp_path,monkeypatch):
    from scripts import run_context_teacher as runner
    from src.provenance import prepare,write_json,read_json
    from src.train import load_config
    p=runner.p;c=tmp_path/'campaign'
    m=prepare(c,p.ROOT,['auto','auto'],tmp_path/'report.md',experiment_dir='context_teacher')
    m.update(protocol='context_teacher',design='tuned');write_json(c/'source_manifest.json',m)
    calls=[]
    def train(campaign,manifest,children,gpu,cfg,label,seed,smoke=False):
        path,resolved=p.save_config(campaign,manifest,cfg,label,seed)
        calls.append((label,seed,smoke,cfg))
        mode=cfg['context']['teacher_mode']
        # Arm-specific optimum so independent studies must find different points.
        best={'none':5e-5,'real':1e-4,'unrelated':4e-5}[mode]
        score=.7-abs(cfg['train']['lr']-best)*100+seed*1e-6
        log=campaign/'results/logs'/resolved['run_name'];log.mkdir(parents=True,exist_ok=True)
        write_json(log/'context_first_views.json',{'center_noise_sha256':[f'noise{seed}']*3,'center_token_sha256':f'B{seed}',
            'third_padded_length':10,'teacher_tokens':10 if mode=='none' else 40,'teacher_padded_length':10 if mode=='none' else 40})
        return {'run_name':resolved['run_name'],'config':str(path),'seed':seed,'label':label,
            'metrics':{'best_sts':score,'last_sts':score-.001,'completed_steps':60 if smoke else 4000},
            'initial_student_sha256':f'full{seed}','initial_shared_sha256':f'shared{seed}','architecture':8192,
            'first_views':None,'batch_order':{'sha256':f'order{seed}'},'data':{'sentences':{'sha256':'same'}}}
    evaluated=[]
    def evaluate(campaign,manifest,groups,children):
        selection=read_json(campaign/'selection.json')
        assert selection['test_scores_seen'] is False
        assert set(groups)=={'A','B','C','C_matched_B'}
        for seed in ('101','102','103'):
            b=load_config(groups['B'][seed]['config']);cc=load_config(groups['C_matched_B'][seed]['config'])
            assert cc['context']['teacher_mode']=='unrelated' and b['context']['teacher_mode']=='real'
            for cfg in (b,cc):cfg.pop('run_name');cfg.pop('experiment');cfg['context'].pop('teacher_mode')
            assert b==cc
        evaluated.append(True)
    monkeypatch.setattr(p,'execute_train',train);monkeypatch.setattr(p,'evaluate',evaluate)
    monkeypatch.setattr(p,'set_stage',p.set_stage);monkeypatch.setattr(p,'render',p.render)
    monkeypatch.setattr(runner.signal,'signal',lambda *args:None)
    runner.run(c,m)
    assert read_json(c/'state.json')['stage']=='complete' and evaluated
    labels=[label for label,_,smoke,_ in calls if not smoke]
    for arm in 'ABC':
        assert sum(l.startswith(f'search_{arm}_') for l in labels)==runner.TRIALS
        assert sum(l.startswith(f'confirm_{arm}_') for l in labels)==9
    assert labels.count('matched_C_B')==3 and len(calls)==3+3*runner.TRIALS+27+3
    search=read_json(c/'search.json')
    anchor=runner.anchor(load_config(p.ROOT/'configs/context_teacher/base.yaml'))
    for arm in 'ABC':
        assert search[arm]['0']['params']==anchor
    # Independent studies: after the shared anchor and startup, arms explore different points.
    assert search['A']['5']['params']!=search['B']['5']['params']
    sel=read_json(c/'selection.json')
    assert sel['C_matched_B_params']==sel['selected_params']['B']
    assert 'Design: tuned' in (c/'report.md').read_text()


def test_full_space_anchors_are_valid_and_cover_strong_sharpening():
    from scripts import run_context_teacher as runner
    from src.train import load_config
    for arm in 'BC':
        base=load_config(runner.p.ROOT/f'configs/context_teacher/{arm}.yaml')
        plain,sharp=runner.full_anchors(base)
        cfg=runner.full_configuration(base,plain)
        # The measured base point is reproduced exactly (only derived noise bounds are recomputed).
        for section in ('train','loss','context'):
            for k,v in base[section].items():
                assert cfg[section][k]==pytest.approx(v),(section,k)
        for k in ('t_lo','t_start','t_max','warmup_frac'):
            assert cfg['augment'][k]==pytest.approx(base['augment'][k])
        assert runner.full_configuration(base,sharp)['loss']['teacher_temp']==.05
        bad=dict(plain);bad['loss.teacher_temp']=.02
        with pytest.raises(ValueError):runner.full_configuration(base,bad)
    assert len(runner.FULL_SPACE)==20


def test_context_teacher_full_design_prunes_scores_late_dev_and_selects_without_test(tmp_path,monkeypatch):
    import optuna
    from scripts import run_context_teacher as runner
    from src.provenance import prepare,write_json,read_json
    from src.train import load_config
    monkeypatch.setattr(runner,'FULL_TRIALS',10)
    p=runner.p;c=tmp_path/'campaign'
    m=prepare(c,p.ROOT,['auto','auto'],tmp_path/'report.md',experiment_dir='context_teacher')
    m.update(protocol='context_teacher',design='full');write_json(c/'source_manifest.json',m)
    calls=[]
    def train(campaign,manifest,children,gpu,cfg,label,seed,smoke=False,trial=None,report='best'):
        path,resolved=p.save_config(campaign,manifest,cfg,label,seed)
        calls.append((label,seed,smoke,cfg,trial,report))
        mode=cfg['context']['teacher_mode']
        if label.startswith('search_') and label.endswith(('t3','t6')):
            raise optuna.TrialPruned('fake')
        temp=cfg['loss']['teacher_temp']
        # Sharp teachers stay stable; flat ones peak early and collapse (dev best != late dev).
        late=.72-temp*0.3+seed*1e-6;best=max(late,.73)
        log=campaign/'results/logs'/resolved['run_name'];log.mkdir(parents=True,exist_ok=True)
        # Like the real run: noise depends on seed AND the (tuned) noise settings.
        aug=json.dumps(cfg['augment'],sort_keys=True)
        write_json(log/'context_first_views.json',{'center_noise_sha256':[f'noise{seed}{aug}']*3,'center_token_sha256':f'B{seed}',
            'third_padded_length':10,'teacher_tokens':40,'teacher_padded_length':40})
        with (log/'evaluations.jsonl').open('w') as f:
            for step,v in ((250,best),(3500,late),(3750,late),(3999,late)):
                f.write(json.dumps({'loop_step':step,'sts_b_dev':v})+'\n')
        return {'run_name':resolved['run_name'],'config':str(path),'seed':seed,'label':label,
            'metrics':{'best_sts':best,'last_sts':late,'completed_steps':60 if smoke else 4000},
            'initial_student_sha256':f'full{seed}','initial_shared_sha256':f'shared{seed}','architecture':8192,
            'first_views':None,'batch_order':{'sha256':f'order{seed}'},'data':{'sentences':{'sha256':'same'}}}
    evaluated=[]
    def evaluate(campaign,manifest,groups,children):
        sel=read_json(campaign/'selection.json')
        assert sel['test_scores_seen'] is False and sel['primary_test_checkpoint']=='last'
        assert set(groups)=={'B','C','C_matched_B'}
        evaluated.append(True)
    monkeypatch.setattr(p,'execute_train',train);monkeypatch.setattr(p,'evaluate',evaluate)
    monkeypatch.setattr(p,'set_stage',p.set_stage);monkeypatch.setattr(p,'render',p.render)
    monkeypatch.setattr(runner.signal,'signal',lambda *args:None)
    runner.run(c,m)
    assert read_json(c/'state.json')['stage']=='complete' and evaluated
    smokes=[x for x in calls if x[2]];assert sorted(x[3]['context']['teacher_mode'] for x in smokes)==['real','unrelated']
    search=[x for x in calls if x[0].startswith('search_')]
    assert len(search)==20 and all(x[4] is not None and x[5]=='current' for x in search)
    s=read_json(c/'search.json')
    for arm in 'BC':
        assert s[arm]['3']['pruned'] and s[arm]['6']['pruned'] and 'late_dev' in s[arm]['0']
        assert s[arm]['1']['params']['loss.teacher_temp']==.05
        short=read_json(c/'shortlist.json')['trials'][arm]
        assert '3' not in short and '6' not in short
        # Ranked by late dev, not by the early-peak best.
        assert short==sorted(short,key=lambda n:-s[arm][n]['late_dev'])
    assert sum(x[0].startswith('confirm_') for x in calls)==18 and sum(x[0]=='matched_C_B' for x in calls)==3
    sel=read_json(c/'selection.json');assert sel['C_matched_B_params']==sel['selected_params']['B']
    report=(c/'report.md').read_text();assert 'Design: full' in report and 'pruned' in report


@pytest.mark.parametrize("report,expected",[("best",[.72,.72,.72]),("current",[.72,.65,.60])])
def test_children_report_mode_sends_running_best_or_current_dev(tmp_path,monkeypatch,report,expected):
    from scripts import run_coviso_optuna as rc
    campaign=tmp_path/'c';(campaign/'source/scripts').mkdir(parents=True);(campaign/'commands').mkdir();(campaign/'status').mkdir()
    evaluations=tmp_path/'evaluations.jsonl'
    rows=[{'loop_step':s,'sts_b_dev':v} for s,v in ((2000,.72),(2500,.65),(3000,.60))]
    script=tmp_path/'fake.py'
    script.write_text('import json,sys,time\nrows=%r\nwith open(%r,"w") as f:\n    for r in rows:f.write(json.dumps(r)+"\\n")\ntime.sleep(2.5)\n'%(rows,str(evaluations)))
    # gpu_guard is bypassed: the fake child runs directly.
    (campaign/'source/scripts/gpu_guard.py').write_text('import subprocess,sys\nsys.exit(subprocess.call(sys.argv[sys.argv.index("--")+1:]))\n')
    reported=[]
    class Trial:
        def report(self,value,step):reported.append(value)
        def should_prune(self):return False
    children=rc.Children(campaign,dict(os.environ))
    children.run('auto','fake',[sys.executable,str(script)],trial=Trial(),evaluation_path=evaluations,report=report)
    assert reported==pytest.approx(expected)


def test_context_teacher_audit_compares_noise_only_under_equal_noise_settings(tmp_path,monkeypatch):
    from scripts import run_context_teacher as runner
    from src.provenance import write_json
    import yaml
    monkeypatch.setattr(runner.p,'audit_runs',lambda *a:None)
    def record(name,label,aug,noise_hash):
        cfg=tmp_path/f'{name}.yaml';cfg.write_text(yaml.safe_dump({'augment':aug}))
        log=tmp_path/'results/logs'/name;log.mkdir(parents=True)
        write_json(log/'context_first_views.json',{'center_noise_sha256':[noise_hash],'center_token_sha256':'B',
            'third_padded_length':10,'teacher_tokens':40,'teacher_padded_length':40})
        return {'run_name':name,'label':label,'seed':42,'config':str(cfg)}
    ok=[record('b0','search_B_t0',{'t_lo':.4},'n1'),record('c0','search_C_t0',{'t_lo':.4},'n1'),
        record('b2','search_B_t2',{'t_lo':.1},'n2')]
    runner.audit(tmp_path,ok,{})
    bad=ok+[record('c2','search_C_t2',{'t_lo':.1},'n3')]
    with pytest.raises(RuntimeError,match='noise'):runner.audit(tmp_path,bad,{})
