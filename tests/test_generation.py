import contextlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from beerllm.common import generate


class Batch(dict):
    @property
    def input_ids(self): return SimpleNamespace(shape=(1,2))


class Result:
    def __getitem__(self,key): return [10,20,30,9][key[1]]


class GenerationTests(unittest.TestCase):
    def test_neutral_greedy_settings_override_copy_discouraging_model_defaults(self):
        calls=[]
        def run(**kwargs):
            calls.append(kwargs)
            return Result()
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=100),
                              generation_config=SimpleNamespace(repetition_penalty=1.1),
                              eval=lambda:None,generate=run)
        tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**k:'prompt',pad_token_id=9,eos_token_id=9,
                                  decode=lambda *a,**k:'3 gallons')
        class Tokenizer:
            pad_token_id=9; eos_token_id=9
            apply_chat_template=staticmethod(tokenizer.apply_chat_template)
            decode=staticmethod(tokenizer.decode)
            def __call__(self,*a,**k): return Batch(input_ids='tokens')
        with patch.dict('sys.modules',{'torch':SimpleNamespace(inference_mode=contextlib.nullcontext)}):
            result=generate(model,Tokenizer(),[{'role':'user','content':'Repeat these quantities.'}],2,True)
        self.assertEqual(calls[0]['repetition_penalty'],1.0)
        self.assertFalse(calls[0]['do_sample'])
        self.assertEqual(calls[0]['num_beams'],1)
        self.assertEqual(calls[0]['no_repeat_ngram_size'],0)
        self.assertTrue(calls[0]['use_cache'])
        self.assertEqual(result['text'],'3 gallons')
        self.assertFalse(result['hit_token_limit'])


if __name__=='__main__': unittest.main()
