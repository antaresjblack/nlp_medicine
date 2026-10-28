"""Fixed-checkpoint tune-only threshold scan. No training, no test access."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
import train as gp
import train_dict as lex


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--run-name',default='threshold_macbert_dict_v1')
    a=p.parse_args()
    if Path(a.run_name).name!=a.run_name or a.run_name in ('.','..'):p.error('Invalid run name')
    thresholds=[-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,2.]
    out=gp.ROOT/'outputs/bert_gp'/a.run_name
    if out.exists():p.error('Output exists; refusing overwrite')
    out.mkdir(parents=True)
    source=gp.ROOT/'data_eval/eval_tune.jsonl'
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    config=checkpoint['args'];records=gp.read(source)
    tokenizer=AutoTokenizer.from_pretrained(config['model'],use_fast=True,local_files_only=True)
    windows,coverage=gp.prepare(records,tokenizer,config['max_length'],config['overlap'])
    matcher=lex.dictionary_from_counts(checkpoint['dictionary'])
    feature_stats=lex.add_features(records,windows,matcher)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=lex.DictionaryGlobalPointer(config['model'],config['feature_dropout']).to(device)
    model.load_state_dict(checkpoint['model']);model.eval()
    manifest=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        tune_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),thresholds=thresholds,device=str(device),
        torch=torch.__version__,coverage=coverage,selection='maximum strict entity micro F1; tie: closest to zero, then higher threshold',
        batch_size=2,precision='float32 eval; same as training script inference',checkpoint_epoch=checkpoint.get('epoch'),dictionary_features=feature_stats,
        dictionary_source='stored training-only counts from checkpoint; no tune gold used for features')
    gp.save(out/'manifest.json',manifest)
    (out/'plan.md').write_text('固定full_v1检查点，完整2500条tune。只改变一个全局logit阈值，候选-2至2、步长0.5；0为原基准。一次模型推理保存最低阈值以上候选，各阈值复用相同分数。窗口合并取相同跨度类别的最大分数；与任一窗口通过阈值的原解码等价。统一baseline/metrics.py评分，报告P/R/F1、预测数和各类结果。按严格实体级micro F1选择，平分时优先接近0，再优先较高阈值。不读取test，不重训，不按test成绩选择。此为AI辅助拟定的调参方案，需小组理解核验并声明辅助范围。',encoding='utf-8')
    candidates=[{} for _ in records];started=time.perf_counter()
    for start in range(0,len(windows),2):
        chunk=windows[start:start+2];ids,att,valid,_=gp.collate(chunk,device)
        scores=model(ids,att,valid,lex.lexical_batch(chunk,device)).cpu()
        for b,w in enumerate(chunk):
            for label,s,e in (scores[b]>min(thresholds)).nonzero().tolist():
                left=w['offsets'][s][0];right=w['offsets'][e][1]-1
                key=left,right,label;bucket=candidates[w['record']]
                bucket[key]=max(bucket.get(key,float('-inf')),float(scores[b,label,s,e]))
        if start%200==0:print(f'windows {min(start+2,len(windows))}/{len(windows)}',flush=True)
    gp.save(out/'timing.json',dict(inference_seconds=time.perf_counter()-started))
    with (out/'span_scores.jsonl').open('w',encoding='utf-8') as f:
        for r,bucket in zip(records,candidates):
            f.write(json.dumps(dict(id=r['id'],scores=[[*k,v] for k,v in bucket.items()]),ensure_ascii=False)+'\n')
    summary=[]
    for threshold in thresholds:
        pred=[]
        for r,bucket in zip(records,candidates):
            entities=[dict(start_idx=s,end_idx=e,type=gp.LABELS[k],entity=r['text'][s:e+1]) for (s,e,k),v in bucket.items() if v>threshold]
            pred.append(dict(id=r['id'],text=r['text'],entities=entities))
        metrics=gp.evaluate(records,pred);metrics.pop('errors',None)
        gp.save(out/f'metrics_{threshold:g}.json',metrics)
        row={k:metrics[k] for k in ['precision','recall','f1','true_positive','predicted_entities']};row['threshold']=threshold;summary.append(row)
        print(row,flush=True)
    gp.save(out/'summary.json',summary)
    best=max(summary,key=lambda x:(x['f1'],-abs(x['threshold']),x['threshold']))
    gp.save(out/'selected_config.json',dict(**best,checkpoint_sha256=manifest['checkpoint_sha256'],selection_split='tune'))
    lines=['| threshold | P | R | F1 | predicted |','|---:|---:|---:|---:|---:|']
    lines += [f"| {x['threshold']:g} | {x['precision']:.6f} | {x['recall']:.6f} | {x['f1']:.6f} | {x['predicted_entities']} |" for x in summary]
    (out/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    baseline=json.loads((a.checkpoint.parent/'metrics.json').read_text(encoding='utf-8'))
    zero=next(x for x in summary if x['threshold']==0)
    gp.save(out/'baseline_check.json',dict(previous_f1=baseline['f1'],rerun_f1=zero['f1'],matches=abs(baseline['f1']-zero['f1'])<1e-10))
    print('Selected:',best,'Output:',out,flush=True)


if __name__=='__main__':main()
