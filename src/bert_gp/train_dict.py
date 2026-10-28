"""MacBERT + gated dictionary token features + GlobalPointer. Separate experiment."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer
import train as gp

sys.path.insert(0,str(gp.ROOT/'baseline'))
from dict_span_features import DictSpanAnnotator

FEATURE_NAMES = ([f'cover_{k}' for k in gp.LABELS] + [f'begin_{k}' for k in gp.LABELS]
                 + [f'end_{k}' for k in gp.LABELS] + ['matched','log_frequency','length','conflict'])
DIM=len(FEATURE_NAMES)


def dictionary_from_counts(counts,min_length=1):
    # Reconstruct exactly the public matcher data structures without repeating records.
    obj=DictSpanAnnotator([],min_count=1,min_length=min_length)
    obj.label_counts={word:Counter(values) for word,values in counts.items()}
    for word,values in obj.label_counts.items():
        if not word or len(word)<min_length:continue
        total=sum(values.values());majority=sorted(values,key=lambda k:(-values[k],k))[0]
        obj.entries.append((word,majority,total))
        obj.by_first_char[word[0]].append((word,majority,total))
    return obj


def text_group_counts(records):
    groups=defaultdict(lambda:defaultdict(Counter))
    for r in records:
        for e in r['entities']:groups[r['text']][e['entity']][e['type']]+=1
    return groups


def character_features(text,matches,exclude=None):
    """No gold accepted here. exclude removes this training text group's dictionary contribution."""
    features=np.zeros((len(text),DIM),dtype=np.float32);used=0
    for match in matches:
        s,e=match['start_idx'],match['end_idx']
        if text[s:e+1]!=match['entity']:raise ValueError('Dictionary offset mismatch')
        counts=Counter(match['type_counts'])
        if exclude:counts.subtract(exclude.get(match['entity'],{}))
        counts=Counter({k:n for k,n in counts.items() if n>0 and k in gp.LABELS})
        total=sum(counts.values())
        if not total:continue
        used+=1
        distribution=np.array([counts[k]/total for k in gp.LABELS],dtype=np.float32)
        features[s:e+1,:9]=np.maximum(features[s:e+1,:9],distribution)
        features[s,9:18]=np.maximum(features[s,9:18],distribution)
        features[e,18:27]=np.maximum(features[e,18:27],distribution)
        extra=np.array([1,min(math.log1p(total)/math.log(101),1),min((e-s+1)/32,1),float(len(counts)>1)])
        features[s:e+1,27:]=np.maximum(features[s:e+1,27:],extra)
    # Conflict also covers different labels introduced by overlapping words.
    features[:,30]=np.maximum(features[:,30],(features[:,:9]>0).sum(axis=1)>1)
    return features,used


def add_features(records,windows,matcher,exclude_groups=None):
    by_record=defaultdict(list)
    for w in windows:by_record[w['record']].append(w)
    stats=dict(records=len(records),raw_matches=0,used_matches=0,tokens=0,matched_tokens=0)
    for ri,r in enumerate(records):
        matches=matcher.match_all(r['text'])
        chars,used=character_features(r['text'],matches,exclude_groups.get(r['text']) if exclude_groups is not None else None)
        stats['raw_matches']+=len(matches);stats['used_matches']+=used
        for w in by_record[ri]:
            matrix=np.zeros((len(w['ids']),DIM),dtype=np.float32)
            for i,(s,e) in enumerate(w['offsets']):
                if e>s:
                    matrix[i]=chars[s:e].max(axis=0)
                    stats['tokens']+=1;stats['matched_tokens']+=int(matrix[i,27]>0)
            w['lex']=matrix
    return stats


def lexical_batch(items,device):
    result=np.zeros((len(items),max(len(x['ids']) for x in items),DIM),dtype=np.float32)
    for b,x in enumerate(items):result[b,:len(x['ids'])]=x['lex']
    return torch.from_numpy(result).to(device)


class DictionaryGlobalPointer(gp.GlobalPointer):
    def __init__(self,name,feature_dropout=.3):
        super().__init__(name)
        size=self.encoder.config.hidden_size
        self.lex_projection=nn.Linear(DIM,size,bias=False)
        self.lex_gate=nn.Linear(size+DIM,1)
        self.feature_dropout=feature_dropout
        nn.init.zeros_(self.lex_projection.weight)
        nn.init.constant_(self.lex_gate.bias,-2.)

    def forward(self,ids,attention,valid,lex=None):
        # Reuse the unchanged span head; inject fused embeddings via an encoder hook.
        if lex is None:raise ValueError('Dictionary model requires lexical features')
        if self.training and self.feature_dropout:
            keep=(torch.rand((*lex.shape[:2],1),device=lex.device)>=self.feature_dropout)
            lex=lex*keep
        def fuse(module,inputs,output):
            h=output.last_hidden_state
            gate=torch.sigmoid(self.lex_gate(torch.cat([h,lex.to(h.dtype)],dim=-1)))
            delta=self.lex_projection(lex.to(h.dtype))
            output.last_hidden_state=h+gate*delta*(lex[...,27:28]>0)
            return output
        hook=self.encoder.register_forward_hook(fuse)
        try:return super().forward(ids,attention,valid)
        finally:hook.remove()


@torch.no_grad()
def predict(model,windows,records,batch,device,threshold):
    model.eval();result=[dict(id=r.get('id',str(i)),text=r['text'],entities=[]) for i,r in enumerate(records)]
    found=[set() for _ in records]
    for start in range(0,len(windows),batch):
        chunk=windows[start:start+batch];ids,att,valid,_=gp.collate(chunk,device)
        scores=model(ids,att,valid,lexical_batch(chunk,device)).cpu()
        for b,w in enumerate(chunk):
            for k,s,e in (scores[b]>threshold).nonzero().tolist():
                left=w['offsets'][s][0];right=w['offsets'][e][1]-1;ri=w['record'];key=left,right,k
                if key in found[ri]:continue
                found[ri].add(key)
                result[ri]['entities'].append(dict(start_idx=left,end_idx=right,type=gp.LABELS[k],entity=records[ri]['text'][left:right+1]))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-name',required=True)
    p.add_argument('--model',default='hfl/chinese-macbert-base')
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--accumulation',type=int,default=8)
    p.add_argument('--max-length',type=int,default=256)
    p.add_argument('--overlap',type=int,default=64)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--warmup-ratio',type=float,default=.1)
    p.add_argument('--feature-dropout',type=float,default=.3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--train-limit',type=int,default=0)
    p.add_argument('--tune-limit',type=int,default=0)
    p.add_argument('--check-data',action='store_true')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--final',action='store_true',help='Frozen checkpoint inference on complete eval_test')
    p.add_argument('--threshold',type=float,default=0)
    p.add_argument('--output-dir',type=Path,default=None,help='Explicit output directory; defaults to outputs/bert_gp/<run-name>')
    p.add_argument('--quiet',action='store_true',help='Suppress the full manifest JSON on stdout')
    a=p.parse_args()
    if a.final and (not a.checkpoint or a.tune_limit or a.train_limit or a.check_data):
        p.error('--final requires a checkpoint and the complete test split, no limits/check-data')
    if Path(a.run_name).name!=a.run_name or a.run_name in ('.','..'):p.error('Invalid run name')
    if min(a.epochs,a.batch_size,a.accumulation)<1 or min(a.train_limit,a.tune_limit)<0:p.error('Invalid counts')
    if not 8<=a.max_length<=512 or not 0<=a.overlap<a.max_length-2:p.error('Invalid window')
    if not 0<=a.feature_dropout<1 or not 0<=a.warmup_ratio<1 or not math.isfinite(a.lr) or a.lr<=0:p.error('Invalid numeric option')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=(a.output_dir.resolve() if a.output_dir else gp.ROOT/'outputs/bert_gp'/a.run_name)
    if out.exists():p.error('Run exists; use a new name')
    out.mkdir(parents=True)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False) if a.checkpoint else None
    if checkpoint:
        for k in ['model','max_length','overlap','feature_dropout']:setattr(a,k,checkpoint['args'][k])
        if a.final:a.threshold=checkpoint['args']['threshold']
        matcher=dictionary_from_counts(checkpoint['dictionary'])
        train=[];windows=[]
    else:
        train=gp.read(gp.ROOT/'data'/'data_cleaned'/'CMeEE_train_cleaned_final.json')
        if a.train_limit:train=train[:a.train_limit]
        matcher=DictSpanAnnotator(train,min_count=1,min_length=1)
    tokenizer=AutoTokenizer.from_pretrained(a.model,use_fast=True)
    tune_path=gp.ROOT/'data'/'data_eval'/('eval_test.jsonl' if a.final else 'eval_tune.jsonl');tune=gp.read(tune_path)
    if a.tune_limit:tune=tune[:a.tune_limit]
    tw,coverage=gp.prepare(tune,tokenizer,a.max_length,a.overlap)
    tune_features=add_features(tune,tw,matcher)
    args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    metadata=dict(args=args,labels=gp.LABELS,feature_names=FEATURE_NAMES,device=str(device),
        torch=torch.__version__,numpy=np.__version__,tune_sha256=hashlib.sha256(tune_path.read_bytes()).hexdigest(),
        tune_coverage=coverage,tune_features=tune_features,dictionary_words=len(matcher.entries),
        training_dictionary_policy='subtract all annotations from current identical-text group before generating features; dropout per token during training',
        dictionary_source='selected training records only; stored with checkpoint; no tune gold used')
    metadata['evaluation_split']='eval_test' if a.final else 'eval_tune'
    if checkpoint:metadata['checkpoint_training_args']=checkpoint['args']
    if not checkpoint:
        windows,coverage=gp.prepare(train,tokenizer,a.max_length,a.overlap)
        metadata.update(train_coverage=coverage,train_features=add_features(train,windows,matcher,text_group_counts(train)),
            train_sha256=hashlib.sha256((gp.ROOT/'data'/'data_cleaned'/'CMeEE_train_cleaned_final.json').read_bytes()).hexdigest())
    gp.save(out/'manifest.json',metadata)
    if not a.quiet:print(json.dumps(metadata,ensure_ascii=False),flush=True)
    if a.check_data:return
    model=DictionaryGlobalPointer(a.model,a.feature_dropout).to(device)
    if checkpoint:model.load_state_dict(checkpoint['model'])
    else:
        model.encoder.gradient_checkpointing_enable()
        heads=[p for name,p in model.named_parameters() if not name.startswith('encoder.')]
        optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':a.lr},{'params':heads,'lr':a.lr}],weight_decay=.01)
        updates=math.ceil(math.ceil(len(windows)/a.batch_size)/a.accumulation)*a.epochs
        scheduler,warmup=gp.make_scheduler(optimizer,updates,a.warmup_ratio)
        metadata['scheduler']=dict(total_updates=updates,warmup_steps=warmup);gp.save(out/'manifest.json',metadata)
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');best=-1;done=0;started=time.perf_counter()
        for epoch in range(a.epochs):
            model.train();random.shuffle(windows);optimizer.zero_grad(set_to_none=True);steps=math.ceil(len(windows)/a.batch_size);total=0
            for step,start in enumerate(range(0,len(windows),a.batch_size)):
                chunk=windows[start:start+a.batch_size];ids,att,valid,gold=gp.collate(chunk,device)
                group=min(a.accumulation,steps-step//a.accumulation*a.accumulation)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                    loss=gp.loss_fn(model(ids,att,valid,lexical_batch(chunk,device)),gold)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                total+=loss.item();scaler.scale(loss/group).backward()
                if (step+1)%a.accumulation==0 or step+1==steps:
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                    old=scaler.get_scale();scaler.step(optimizer);scaler.update()
                    if scaler.get_scale()>=old:scheduler.step();done+=1
                    optimizer.zero_grad(set_to_none=True)
                if (step+1)%20==0:print(f'epoch {epoch+1} step {step+1}/{steps} loss {loss.item():.4f}',flush=True)
            pred=predict(model,tw,tune,a.batch_size,device,a.threshold);m=gp.evaluate(tune,pred)
            row=dict(epoch=epoch+1,loss=total/steps,f1=m['f1'],optimizer_updates=done,lr=optimizer.param_groups[0]['lr'])
            with (out/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print(row,flush=True)
            if m['f1']>best:
                best=m['f1'];torch.save(dict(model=model.state_dict(),args=args,epoch=epoch+1,dictionary={w:dict(c) for w,c in matcher.label_counts.items()}),out/'best.pt')
        gp.save(out/'timing.json',dict(training_and_epoch_eval_seconds=time.perf_counter()-started))
        state=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(state['model'])
    pred=predict(model,tw,tune,a.batch_size,device,a.threshold);metrics=gp.evaluate(tune,pred)
    errors=metrics.pop('errors',[]);gp.save(out/'metrics.json',metrics);gp.save(out/'errors.json',errors)
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for row in pred:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    tokenizer.save_pretrained(out/'tokenizer');print('Completed:',out,'F1:',metrics['f1'],flush=True)


if __name__=='__main__':main()
