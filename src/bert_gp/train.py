"""BERT + GlobalPointer medical NER; run from any directory."""
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['dis','sym','dru','equ','pro','bod','ite','mic','dep']


def read(path):
    text = path.read_text(encoding='utf-8-sig')
    return [json.loads(x) for x in text.splitlines() if x.strip()] if path.suffix=='.jsonl' else json.loads(text)


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(records, tokenizer, length, overlap):
    windows, stats = [], dict(records=len(records), gold=0, covered=0, unaligned=0)
    for ri, record in enumerate(records):
        enc = tokenizer(record['text'], max_length=length, truncation=True, stride=overlap,
                        return_overflowing_tokens=True, return_offsets_mapping=True)
        covered=set()
        for wi, offsets in enumerate(enc['offset_mapping']):
            starts={s:i for i,(s,e) in enumerate(offsets) if e>s}
            ends={e:i for i,(s,e) in enumerate(offsets) if e>s}
            spans=[]
            for ei,e in enumerate(record['entities']):
                if record['text'][e['start_idx']:e['end_idx']+1]!=e['entity']:
                    raise ValueError('Invalid source entity')
                if e['start_idx'] in starts and e['end_idx']+1 in ends:
                    spans.append((LABELS.index(e['type']),starts[e['start_idx']],ends[e['end_idx']+1]))
                    covered.add(ei)
            windows.append(dict(record=ri, ids=enc['input_ids'][wi], offsets=offsets, spans=spans))
        stats['gold']+=len(record['entities']);stats['covered']+=len(covered)
    stats['unaligned']=stats['gold']-stats['covered'];stats['windows']=len(windows)
    return windows,stats


def collate(items, device):
    n=max(len(x['ids']) for x in items)
    ids=torch.zeros(len(items),n,dtype=torch.long)
    valid=torch.zeros(len(items),n,dtype=torch.bool)
    attention=torch.zeros_like(ids)
    gold=torch.zeros(len(items),len(LABELS),n,n)
    for b,x in enumerate(items):
        l=len(x['ids']);ids[b,:l]=torch.tensor(x['ids']);attention[b,:l]=1
        valid[b,:l]=torch.tensor([e>s for s,e in x['offsets']])
        for k,s,e in x['spans']:gold[b,k,s,e]=1
    return ids.to(device),attention.to(device),valid.to(device),gold.to(device)


class GlobalPointer(nn.Module):
    def __init__(self, name, head=64, load_pretrained=True):
        super().__init__()
        if load_pretrained:
            self.encoder=AutoModel.from_pretrained(name)
        else:
            # A fine-tuned checkpoint contains the full encoder state. Build the
            # same architecture from config so inference does not require a
            # second download (or a complete Hugging Face weight cache).
            self.encoder=AutoModel.from_config(AutoConfig.from_pretrained(name))
        self.head=head
        self.projection=nn.Linear(self.encoder.config.hidden_size,len(LABELS)*head*2)

    def forward(self,ids,attention,valid):
        h=self.encoder(input_ids=ids,attention_mask=attention).last_hidden_state
        b,l,_=h.shape
        q,k=self.projection(h).reshape(b,l,len(LABELS),2*self.head).chunk(2,dim=-1)
        angles=torch.arange(l,device=h.device).float()[:,None] / (10000**(torch.arange(0,self.head,2,device=h.device).float()/self.head))
        sin=angles.sin().repeat_interleave(2,dim=-1)[None,:,None,:]
        cos=angles.cos().repeat_interleave(2,dim=-1)[None,:,None,:]
        def rotate(x):
            paired=torch.stack((-x[...,1::2],x[...,::2]),dim=-1).flatten(-2)
            return x*cos+paired*sin
        scores=torch.einsum('bmhd,bnhd->bhmn',rotate(q),rotate(k))/math.sqrt(self.head)
        mask=valid[:,None,:,None]&valid[:,None,None,:]
        mask=mask & torch.ones(l,l,device=h.device,dtype=torch.bool).triu()[None,None]
        return scores.float().masked_fill(~mask,-1e12)


def loss_fn(scores,gold):
    # Multilabel categorical cross entropy over all start/end pairs per type.
    y=gold.flatten(2); s=scores.flatten(2)
    signed=(1-2*y)*s
    neg=signed-y*1e12;pos=signed-(1-y)*1e12
    zeros=torch.zeros_like(signed[...,:1])
    return (torch.logsumexp(torch.cat([neg,zeros],-1),-1)+torch.logsumexp(torch.cat([pos,zeros],-1),-1)).mean()


@torch.no_grad()
def predict(model,windows,records,batch,device,threshold):
    model.eval();out=[dict(id=r.get('id',str(i)),text=r['text'],entities=[]) for i,r in enumerate(records)];seen=[set() for _ in records]
    for start in range(0,len(windows),batch):
        chunk=windows[start:start+batch];ids,att,valid,_=collate(chunk,device)
        scores=model(ids,att,valid).cpu()
        for b,w in enumerate(chunk):
            for label,s,e in (scores[b]>threshold).nonzero().tolist():
                left=w['offsets'][s][0];right=w['offsets'][e][1]-1
                key=left,right,label;ri=w['record']
                if key not in seen[ri]:
                    seen[ri].add(key);out[ri]['entities'].append(dict(start_idx=left,end_idx=right,type=LABELS[label],entity=records[ri]['text'][left:right+1]))
    return out


def evaluate(gold,pred):
    sys.path.insert(0,str(ROOT/'baseline'))
    from metrics import evaluate_records
    return evaluate_records(gold,pred)


def make_optimizer(model,encoder_lr,head_lr):
    """Separate LR only; retain the original weight decay for a controlled change."""
    return torch.optim.AdamW([
        {'params':list(model.encoder.parameters()),'lr':encoder_lr,'group_name':'encoder'},
        {'params':list(model.projection.parameters()),'lr':head_lr,'group_name':'span_head'},
    ],weight_decay=.01)


def make_scheduler(optimizer,total_steps,warmup_ratio):
    warmup_steps=math.ceil(total_steps*warmup_ratio)
    def multiplier(step):
        if step<warmup_steps:
            return step/max(1,warmup_steps)
        return max(0.,(total_steps-step)/max(1,total_steps-warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier),warmup_steps


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='google-bert/bert-base-chinese')
    p.add_argument('--run-name',required=True)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--accumulation',type=int,default=8)
    p.add_argument('--max-length',type=int,default=256)
    p.add_argument('--overlap',type=int,default=64)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--encoder-lr',type=float,default=None,help='BERT learning rate; defaults to --lr')
    p.add_argument('--head-lr',type=float,default=None,help='GlobalPointer projection learning rate; defaults to --lr')
    p.add_argument('--schedule',choices=['constant','linear'],default='constant')
    p.add_argument('--warmup-ratio',type=float,default=0.1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--train-limit',type=int,default=0)
    p.add_argument('--tune-limit',type=int,default=0)
    p.add_argument('--threshold',type=float,default=0)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--final',action='store_true')
    p.add_argument('--check-data',action='store_true')
    p.add_argument('--overfit-check',action='store_true',help='Evaluate training records each epoch; select checkpoint by train F1')
    p.add_argument('--model-dir',type=Path,help='Local config/tokenizer directory for checkpoint inference')
    p.add_argument('--output-dir',type=Path,default=None,help='Explicit output directory; defaults to outputs/bert_gp/<run-name>')
    p.add_argument('--quiet',action='store_true',help='Suppress the full manifest JSON on stdout')
    a=p.parse_args()
    if not math.isfinite(a.warmup_ratio) or not 0<=a.warmup_ratio<1:
        p.error('warmup-ratio must be in [0, 1)')
    if a.encoder_lr is None:a.encoder_lr=a.lr
    if a.head_lr is None:a.head_lr=a.lr
    if not all(math.isfinite(x) and x>0 for x in (a.lr,a.encoder_lr,a.head_lr)):
        p.error('Learning rates must be finite positive numbers')
    if Path(a.run_name).name!=a.run_name or a.run_name in ('.','..'):p.error('Invalid run name')
    if min(a.epochs,a.batch_size,a.accumulation)<1 or min(a.train_limit,a.tune_limit)<0:p.error('Invalid counts')
    if not 8<=a.max_length<=512 or not 0<=a.overlap<a.max_length-2:p.error('Invalid length/overlap')
    if a.final and not a.checkpoint:p.error('--final requires --checkpoint')
    if a.overfit_check and (a.checkpoint or not 1<=a.train_limit<=256):
        p.error('--overfit-check requires training with --train-limit between 1 and 256')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folder=(a.output_dir.resolve() if a.output_dir else ROOT/'outputs/bert_gp'/a.run_name)
    if folder.exists():p.error('Run exists; use a new name. Never overwrite results.')
    folder.mkdir(parents=True)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False) if a.checkpoint else None
    if checkpoint:
        for key in ['model','max_length','overlap','threshold']:
            setattr(a,key,checkpoint['args'][key])
        if a.model_dir:a.model=str(a.model_dir.resolve())
    tokenizer=AutoTokenizer.from_pretrained(a.model,use_fast=True)
    source=ROOT/'data'/'data_eval'/('eval_test.jsonl' if a.final else 'eval_tune.jsonl')
    tune=read(source)
    if a.tune_limit and not a.final:tune=tune[:a.tune_limit]
    tw,ts=prepare(tune,tokenizer,a.max_length,a.overlap)
    metadata=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},device=str(device),labels=LABELS,
        torch=torch.__version__,numpy=np.__version__,tune_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),tokenizer_commit=tokenizer.init_kwargs.get('_commit_hash'),tune_coverage=ts)
    metadata['evaluation_split']='eval_test' if a.final else 'eval_tune'
    if checkpoint:
        train=[];windows=[]
    else:
        path=ROOT/'data'/'data_cleaned'/'CMeEE_train_cleaned_final.json';train=read(path)
        if a.train_limit:train=train[:a.train_limit]
        windows,stats=prepare(train,tokenizer,a.max_length,a.overlap)
        metadata.update(train_coverage=stats,train_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    save(folder/'manifest.json',metadata)
    if not a.quiet:print(json.dumps(metadata,ensure_ascii=False),flush=True)
    if a.check_data:return
    model=GlobalPointer(a.model, load_pretrained=checkpoint is None).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.encoder.gradient_checkpointing_enable()
        optimizer=make_optimizer(model,a.encoder_lr,a.head_lr)
        total_updates=math.ceil(math.ceil(len(windows)/a.batch_size)/a.accumulation)*a.epochs
        scheduler=None
        if a.schedule=='linear':
            scheduler,warmup_steps=make_scheduler(optimizer,total_updates,a.warmup_ratio)
            metadata['scheduler']=dict(total_updates=total_updates,warmup_steps=warmup_steps)
            save(folder/'manifest.json',metadata)
        successful_updates=0
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
        best=-1
        for epoch in range(a.epochs):
            model.train();random.shuffle(windows);optimizer.zero_grad(set_to_none=True)
            steps=math.ceil(len(windows)/a.batch_size);total=0
            for step,start in enumerate(range(0,len(windows),a.batch_size)):
                ids,att,valid,gold=collate(windows[start:start+a.batch_size],device)
                # Final partial accumulation group uses its actual batch count.
                group=min(a.accumulation,steps-(step//a.accumulation)*a.accumulation)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                    loss=loss_fn(model(ids,att,valid),gold)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                total+=loss.item();scaler.scale(loss/group).backward()
                if (step+1)%a.accumulation==0 or step+1==steps:
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                    old_scale=scaler.get_scale()
                    scaler.step(optimizer);scaler.update()
                    if scaler.get_scale()>=old_scale:
                        successful_updates+=1
                        if scheduler is not None:scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if (step+1)%20==0:print(f'epoch {epoch+1} step {step+1}/{steps} loss {loss.item():.4f}',flush=True)
            pred=predict(model,tw,tune,a.batch_size,device,a.threshold);metrics=evaluate(tune,pred)
            row=dict(epoch=epoch+1,loss=total/steps,f1=metrics['f1'])
            row.update(optimizer_updates=successful_updates,
                       encoder_lr=optimizer.param_groups[0]['lr'],head_lr=optimizer.param_groups[1]['lr'])
            selection_f1=metrics['f1']
            if a.overfit_check:
                train_pred=predict(model,windows,train,a.batch_size,device,a.threshold)
                train_metrics=evaluate(train,train_pred)
                row.update(train_f1=train_metrics['f1'],train_precision=train_metrics['precision'],
                           train_recall=train_metrics['recall'],train_predictions=train_metrics['predicted_entities'])
                selection_f1=train_metrics['f1']
            with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print(row,flush=True)
            if selection_f1>best:
                best=selection_f1;torch.save(dict(model=model.state_dict(),args=metadata['args'],epoch=epoch+1),folder/'best.pt')
        state=torch.load(folder/'best.pt',map_location=device,weights_only=False);model.load_state_dict(state['model'])
        if a.overfit_check:
            train_pred=predict(model,windows,train,a.batch_size,device,a.threshold)
            train_metrics=evaluate(train,train_pred)
            train_errors=train_metrics.pop('errors',[])
            save(folder/'train_metrics.json',train_metrics)
            save(folder/'train_errors.json',train_errors)
            with (folder/'train_predictions.jsonl').open('w',encoding='utf-8') as f:
                for record in train_pred:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    pred=predict(model,tw,tune,a.batch_size,device,a.threshold);metrics=evaluate(tune,pred)
    errors=metrics.pop('errors',[]);save(folder/'metrics.json',metrics);save(folder/'errors.json',errors)
    with (folder/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for record in pred:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    tokenizer.save_pretrained(folder/'tokenizer')
    print('Completed:',folder,'F1:',metrics['f1'])


if __name__=='__main__':main()
