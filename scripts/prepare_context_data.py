"""Prepare fixed-revision Wikipedia document triples and shared B embedding statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.context_data import DATASET,REVISION,SHARD,prepare_cache


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-shard',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--max-documents',type=int,default=20000)
    p.add_argument('--max-train',type=int,default=150000)
    p.add_argument('--backbone',default='bert-base-uncased')
    args=p.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer,AutoModel
    import torch
    from scripts.prepare_data import compute_embed_stats
    tokenizer=AutoTokenizer.from_pretrained(args.backbone)
    def documents():
        for batch in pq.ParquetFile(args.raw_shard).iter_batches(batch_size=128):
            yield from batch.to_pylist()
    manifest=prepare_cache(documents(),tokenizer,args.out_dir,{
        'dataset':DATASET,'revision':REVISION,'shard':SHARD,'licenses':['cc-by-sa-3.0','gfdl'],
        'raw_file':str(args.raw_shard.resolve()),'raw_sha256':hashlib.file_digest(args.raw_shard.open('rb'),'sha256').hexdigest(),
        'tokenizer':args.backbone},args.max_documents,args.max_train)
    # Statistics documents are disjoint from training and context diagnostics.
    stats=[json.loads(line)['B'] for line in (args.out_dir/'stats.jsonl').open()][:10000]
    model=AutoModel.from_pretrained(args.backbone).eval()
    mean,std=compute_embed_stats(stats,tokenizer,model.get_input_embeddings(),'cpu',batch_size=64)
    torch.save({'mean':mean,'std':std},args.out_dir/'embed_stats_bert_base_uncased.pt')
    manifest['statistics']={'sentences':len(stats),'split':'stats','backbone':args.backbone,'device':'cpu word lookup only',
        'file_sha256':hashlib.file_digest((args.out_dir/'embed_stats_bert_base_uncased.pt').open('rb'),'sha256').hexdigest()}
    (args.out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['counts']),flush=True)

if __name__=='__main__':main()
