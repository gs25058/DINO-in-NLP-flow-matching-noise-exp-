"""Auditable document/paragraph triples; never infer adjacency from wiki1m rows."""
import hashlib
import json
import random
import re
from pathlib import Path

PREPROCESS_VERSION='context-triples-v1'
DATASET='wikimedia/wikipedia'
REVISION='b04c8d1ceb2f5cd4588862100d08de323dccfbaa'
SHARD='20231101.en/train-00000-of-00041.parquet'
ABBREVIATION=re.compile(r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|Fig|No|Inc|Ltd)\.|\b[A-Za-z]\.|\b\d+\.\d+|\b(?:e\.g|i\.e)\.',re.I)
SENTENCE=re.compile(r'.+?[.!?][\"”’\')\]]*(?=\s+[A-Z0-9\"“‘(]|\s*$)')


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalized(text):
    return ' '.join(text.lower().split())


def split_document(doc):
    """Treat each nonempty source line as a conservative paragraph boundary.

    Reject ambiguous abbreviation/initial/decimal paragraphs, titles/list/table lines,
    and paragraphs with an unparsed tail. Return exact raw-text character offsets.
    """
    raw=doc['text'];result=[];offset=0;index=0
    for paragraph_id,line in enumerate(raw.splitlines(keepends=True)):
        stripped=line.strip()
        start=offset+len(line)-len(line.lstrip());offset+=len(line)
        if (not stripped or not stripped[-1] in '.!?\"”’\')]' or
            re.match(r'^[=*#;:|!•\-]',stripped) or any(x in stripped for x in ('\t','{|','|}','{{','}}')) or
            ABBREVIATION.search(stripped)):
            continue
        matches=list(SENTENCE.finditer(stripped))
        if len(matches)<3 or stripped[matches[-1].end():].strip():continue
        sentences=[]
        for sentence_index,m in enumerate(matches):
            text=m.group().strip();leading=len(m.group())-len(m.group().lstrip())
            a=start+m.start()+leading;b=a+len(text)
            assert raw[a:b]==text
            sentences.append({'text':text,'sentence_index':sentence_index,'doc_sentence_index':index,
                              'start':a,'end':b,'paragraph_id':paragraph_id})
            index+=1
        result.append(sentences)
    return result


def document_split(doc_id):
    n=int(digest('context-split-v1:'+str(doc_id))[:8],16)%100
    return 'stats' if n<5 else 'heldout' if n<10 else 'train'


def triples_from_document(doc,tokenizer):
    paragraphs=split_document(doc)
    all_sentences=[s for p in paragraphs for s in p]
    if not all_sentences:return []
    ids=tokenizer([s['text'] for s in all_sentences],add_special_tokens=False,truncation=False)['input_ids']
    for s,tokens in zip(all_sentences,ids):
        s['ids']=tokens;s['normalized']=normalized(s['text'])
    triples=[]
    for paragraph in paragraphs:
        for i in range(1,len(paragraph)-1):
            a,b,c=paragraph[i-1:i+2]
            if not 3<=len(b['ids'])<=126 or min(len(a['ids']),len(c['ids']))<1:continue
            if min(len(x['text']) for x in (a,b,c))<10:continue
            if len({normalized(x['text']) for x in (a,b,c)})!=3:continue
            left,right=a['ids'][-48:],c['ids'][:48]
            excluded={x['normalized'] for x in (a,b,c)}
            def eligible(s,n):
                return abs(s['doc_sentence_index']-b['doc_sentence_index'])>=8 and len(s['ids'])>=n and s['normalized'] not in excluded
            fa=next((s for s in all_sentences if eligible(s,len(left))),None)
            fc=next((s for s in reversed(all_sentences) if s is not fa and eligible(s,len(right))),None)
            # Retain the same B set for the preregistered optional far-context arm.
            if fa is None or fc is None:continue
            triples.append({'doc_id':str(doc['id']),'url':doc.get('url',''),'title':doc.get('title',''),
                'paragraph_id':b['paragraph_id'],'sentence_index':b['sentence_index'],
                'doc_sentence_index':b['doc_sentence_index'],'A':a['text'],'B':b['text'],'C':c['text'],'text':b['text'],
                'source_positions':[[x['start'],x['end']] for x in (a,b,c)],
                'center_ids':b['ids'],'left_ids':left,'right_ids':right,
                'far_left_ids':fa['ids'][-len(left):],'far_right_ids':fc['ids'][:len(right)],
                'far_positions':[[fa['start'],fa['end']],[fc['start'],fc['end']]],
                'far_sentence_indices':[fa['doc_sentence_index'],fc['doc_sentence_index']],
                'far_texts':[fa['text'],fc['text']]})
    return triples


WINDOW_VERSION='context-windows-v2'


def windows_from_document(doc,tokenizer,min_side=2,max_side=3,side_budget=160):
    """B plus 2..3 whole same-paragraph sentences per side (5..7 sentences total).

    Context sentences are never truncated: the nearest ``min_side`` sentences of a
    side must fit ``side_budget`` WordPieces or B is dropped; a farther sentence is
    added only while it still fits.
    """
    rows=[]
    for paragraph in split_document(doc):
        if len(paragraph)<2*min_side+1:continue
        ids=tokenizer([s['text'] for s in paragraph],add_special_tokens=False,truncation=False)['input_ids']
        for s,tokens in zip(paragraph,ids):s['ids']=tokens
        for i in range(min_side,len(paragraph)-min_side):
            b=paragraph[i]
            if not 3<=len(b['ids'])<=126:continue
            sides=[]
            for order in (paragraph[i-1::-1] if i else [],paragraph[i+1:]):
                chosen=[];used=0
                for s in order[:max_side]:
                    if used+len(s['ids'])>side_budget:break
                    chosen.append(s);used+=len(s['ids'])
                sides.append(chosen)
            left,right=sides[0][::-1],sides[1]
            if min(len(left),len(right))<min_side:continue
            window=left+[b]+right
            if min(len(x['text']) for x in window)<10:continue
            if len({normalized(x['text']) for x in window})!=len(window):continue
            rows.append({'doc_id':str(doc['id']),'url':doc.get('url',''),'title':doc.get('title',''),
                'paragraph_id':b['paragraph_id'],'sentence_index':b['sentence_index'],
                'doc_sentence_index':b['doc_sentence_index'],
                'A':' '.join(x['text'] for x in left),'B':b['text'],'C':' '.join(x['text'] for x in right),'text':b['text'],
                'left_texts':[x['text'] for x in left],'right_texts':[x['text'] for x in right],
                'window_sentences':len(window),
                'source_positions':[[x['start'],x['end']] for x in window],
                'center_ids':b['ids'],'left_ids':[t for x in left for t in x['ids']],
                'right_ids':[t for x in right for t in x['ids']]})
    return rows


def add_unrelated_context(rows,seed=19,candidates_per_row=64):
    """Exact content-length matching, from other documents in this same split."""
    rng=random.Random(seed);matched=[]
    for row in rows:
        l,r=len(row['left_ids']),len(row['right_ids'])
        donors=[]
        # Deterministic bounded rejection; unmatched centers are dropped for every arm.
        candidates=[rng.randrange(len(rows)) for _ in range(candidates_per_row)]
        for i in candidates:
            d=rows[i]
            if d['doc_id']==row['doc_id']:continue
            if len(d['left_ids'])<l or len(d['right_ids'])<r:continue
            target=normalized(row['B'])
            if any(normalized(d[k])==target for k in ('A','B','C')):continue
            donors.append(d);break
        if not donors:continue
        d=donors[0];copy=dict(row)
        copy.update(unrelated_left_ids=d['left_ids'][-l:],unrelated_right_ids=d['right_ids'][:r],
            unrelated_doc_id=d['doc_id'],unrelated_positions=d['source_positions'],
            unrelated_source_row=d.get('source_row'),unrelated_paragraph_id=d.get('paragraph_id'),
            unrelated_texts=[d['A'],d['C']],unrelated_center_text=d['B'])
        matched.append(copy)
    return matched


def validate_row(row,side_budget=48):
    """v1 triples carry a far-context arm; v2 windows do not (``far_*`` absent)."""
    if row['unrelated_doc_id']==row['doc_id']:raise ValueError('unrelated context shares document')
    if any(normalized(t)==normalized(row['B']) for t in row['unrelated_texts']+[row['unrelated_center_text']]):
        raise ValueError('unrelated context duplicates center sentence')
    kinds=('unrelated','far') if 'far_sentence_indices' in row else ('unrelated',)
    if 'far' in kinds and any(abs(i-row['doc_sentence_index'])<8 for i in row['far_sentence_indices']):
        raise ValueError('far context is adjacent')
    for side in ('left','right'):
        n=len(row[f'{side}_ids'])
        if not 1<=n<=side_budget or any(len(row[f'{kind}_{side}_ids'])!=n for kind in kinds):
            raise ValueError('context content lengths differ')
    if not 3<=len(row['center_ids'])<=126:raise ValueError('invalid center length')


def prepare_cache(documents,tokenizer,out,source,max_documents=20000,max_train=150000,window=None):
    """Write disjoint document splits, offsets, token IDs and source fingerprints.

    ``window=None`` keeps the v1 triple cache; a dict of ``windows_from_document``
    arguments builds the v2 5..7-sentence cache instead.
    """
    out=Path(out)
    if out.exists():raise FileExistsError('use a new data output directory')
    out.mkdir(parents=True)
    groups={s:[] for s in ('train','stats','heldout')}
    seen_docs=set();seen_doc_text=set();seen_centers=set();counts={'documents_read':0,'duplicates':0}
    for row_number,doc in enumerate(documents):
        if row_number>=max_documents:break
        counts['documents_read']+=1
        if row_number%1000==0:print(f'Prepared source documents: {row_number}',flush=True)
        doc_id=str(doc['id']);doc_hash=digest(normalized(doc['text']))
        if doc_id in seen_docs or doc_hash in seen_doc_text:counts['duplicates']+=1;continue
        seen_docs.add(doc_id);seen_doc_text.add(doc_hash)
        group=document_split(doc_id)
        triples=triples_from_document(doc,tokenizer) if window is None else windows_from_document(doc,tokenizer,**window)
        for row in triples:
            key=digest(normalized(row['B']))
            if key in seen_centers:continue
            seen_centers.add(key)
            row.update(source_row=row_number,source_shard=source['shard'],source_revision=source['revision'])
            if group!='train' or len(groups[group])<max_train:groups[group].append(row)
    doc_ids={};counts['before_unrelated_matching']={k:len(v) for k,v in groups.items()}
    for split,rows in groups.items():
        rows=add_unrelated_context(rows,candidates_per_row=64 if window is None else 256)
        if not rows:raise ValueError(f'no usable {split} triples; supply more source documents')
        for row in rows:validate_row(row,48 if window is None else window['side_budget'])
        groups[split]=rows
        doc_ids[split]=sorted({r['doc_id'] for r in rows})
        with (out/f'{split}.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    for a,b in (('train','stats'),('train','heldout'),('stats','heldout')):
        if set(doc_ids[a])&set(doc_ids[b]):raise AssertionError('document split leakage')
    with (out/'rank_eval_sentences.jsonl').open('w') as f:
        for row in groups['heldout'][:2048]:f.write(json.dumps({'text':row['B']})+'\n')
    manifest={'source':source,'preprocess_version':PREPROCESS_VERSION if window is None else WINDOW_VERSION,'split':'sha256(doc ID), 90/5/5 train/stats/heldout',
        'splitter':'conservative punctuation+capital rule; reject abbreviation/initial/decimal paragraphs; one source line per paragraph',
        'center_rule':'exclude centers outside 3..126 WordPieces; no center truncation',
        **({'context_budget':48,'far_min_sentence_distance':8} if window is None else
           {'window':window,'context_rule':'same paragraph, whole sentences only, no far arm','unrelated_candidates':256}),'max_documents':max_documents,'max_train':max_train,
        'selection':'first source rows, not a representative random Wikipedia sample',
        'counts':{**counts,**{k:len(v) for k,v in groups.items()}},'document_ids':doc_ids,
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
