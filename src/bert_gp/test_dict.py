"""Offline synthetic tests, no model downloads, no test split access."""
import json
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
import train as gp
import train_dict as lex


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.config=types.SimpleNamespace(hidden_size=16);self.emb=torch.nn.Embedding(20,16)
    def forward(self,input_ids,attention_mask):
        return types.SimpleNamespace(last_hidden_state=self.emb(input_ids))


class Tests(unittest.TestCase):
    def setUp(self):
        self.matcher=lex.dictionary_from_counts({'肺炎':{'dis':2,'sym':1},'肺':{'bod':1}})

    def test_all_matches(self):
        matches=self.matcher.match_all('肺炎及肺炎')
        self.assertEqual(len(matches),4)
        self.assertEqual([x['start_idx'] for x in matches if x['entity']=='肺炎'],[0,3])
        f,used=lex.character_features('肺炎及肺炎',matches)
        self.assertEqual(used,4);self.assertEqual(f[2].sum(),0)
        self.assertGreater(f[0,0],0);self.assertGreater(f[0,5],0)
        self.assertEqual(f[0,30],1)

    def test_remove_self_text(self):
        counts={'肺炎':{'dis':2,'sym':1},'肺':{'bod':1}}
        f,used=lex.character_features('肺炎',self.matcher.match_all('肺炎'),counts)
        self.assertEqual(used,0);self.assertEqual(f.sum(),0)

    def test_dictionary_roundtrip(self):
        values=json.loads(json.dumps({k:dict(v) for k,v in self.matcher.label_counts.items()}))
        second=lex.dictionary_from_counts(values)
        self.assertEqual(self.matcher.match_all('肺炎'),second.match_all('肺炎'))

    def test_checkpoint_mode_skips_pretrained_weights(self):
        config=types.SimpleNamespace(hidden_size=16)
        with patch.object(gp.AutoConfig,'from_pretrained',return_value=config), \
             patch.object(gp.AutoModel,'from_pretrained',side_effect=AssertionError('base weights requested')), \
             patch.object(gp.AutoModel,'from_config',return_value=Encoder()):
            model=lex.DictionaryGlobalPointer('offline',load_pretrained=False)
        self.assertEqual(model.encoder.config.hidden_size,16)

    def test_alignment(self):
        records=[{'text':'肺炎','entities':[]}]
        windows=[dict(record=0,ids=[1,2,3,4],offsets=[(0,0),(0,1),(1,2),(0,0)],spans=[])]
        lex.add_features(records,windows,self.matcher)
        f=windows[0]['lex'];self.assertEqual(f.shape,(4,31));self.assertEqual(f[0].sum(),0);self.assertEqual(f[3].sum(),0)
        self.assertGreater(f[1,27],0)

    def test_gate_backward_and_loading(self):
        with patch.object(gp.AutoModel,'from_pretrained',side_effect=lambda name:Encoder()):
            model=lex.DictionaryGlobalPointer('offline',feature_dropout=0)
            model.eval();ids=torch.tensor([[1,2,3,4]]);attention=torch.ones_like(ids);valid=torch.tensor([[False,True,True,False]])
            features=torch.ones(1,4,31)
            base=gp.GlobalPointer.forward(model,ids,attention,valid)
            actual=model(ids,attention,valid,features)
            self.assertTrue(torch.equal(base,actual))
            gold=torch.zeros_like(actual);gold[0,0,1,2]=1;gold[0,5,1,1]=1
            loss=gp.loss_fn(actual,gold);self.assertTrue(torch.isfinite(loss));loss.backward()
            self.assertGreater(model.lex_projection.weight.grad.abs().sum().item(),0)
            other=lex.DictionaryGlobalPointer('offline',feature_dropout=0);other.load_state_dict(model.state_dict());other.eval()
            self.assertTrue(torch.equal(actual,other(ids,attention,valid,features)))
            self.assertEqual(len(model.encoder._forward_hooks),0)

    def test_no_dictionary_no_effect(self):
        with patch.object(gp.AutoModel,'from_pretrained',return_value=Encoder()):
            model=lex.DictionaryGlobalPointer('offline',feature_dropout=0);model.eval()
            with torch.no_grad():model.lex_projection.weight.fill_(1)
            ids=torch.tensor([[1,2]]);attention=torch.ones_like(ids);valid=torch.ones_like(ids,dtype=torch.bool)
            self.assertTrue(torch.equal(gp.GlobalPointer.forward(model,ids,attention,valid),model(ids,attention,valid,torch.zeros(1,2,31))))


if __name__=='__main__':unittest.main()
