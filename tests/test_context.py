import copy
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
