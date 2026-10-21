import copy
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from beerllm.common import model_revision,write_jsonl
from beerllm.fit_check import PARAPHRASES,QUESTIONS,evaluate,evaluation_rows,prepare,prepare_varied,select_examples
from beerllm.common import read_jsonl,write_json


def rows():
    return [{'task':'followup','split':'train','messages':[{'role':'system','content':'Explain recipes.'},
             {'role':'user','content':q},{'role':'assistant','content':'A source-checked answer.'}]} for q in QUESTIONS]


class FitCheckTests(unittest.TestCase):
    def test_varied_fit_keeps_canonical_evaluation_and_rejects_heldout_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.jsonl';config=root/'refine.json';out=root/'fit'
            examples=rows();facts=[]
            for index,example in enumerate(list(examples)):
                example['recipe_id']=str(index)
                alternative=copy.deepcopy(example);alternative['task']='followup_variant'
                alternative['messages'][1]['content']='Another wording '+str(index)
                alternative['messages'][-1]['content']='Alternate answer '+str(index)
                examples.append(alternative)
                facts.append({'original_question':QUESTIONS[index],'answer':alternative['messages'][-1]['content']})
            write_jsonl(source,examples);write_json(config,{'lessons':[{'facts':facts}]})
            report=prepare_varied(source,config,out)
            self.assertEqual(report['train_examples'],4)
            self.assertEqual(len(read_jsonl(out/'evaluation.jsonl')),2)
            self.assertEqual(len(evaluation_rows(read_jsonl(out/'evaluation.jsonl'),True)),4)
            self.assertEqual(prepare_varied(source,config,out),report)
            examples[-1]['split']='test';write_jsonl(source,examples)
            with self.assertRaisesRegex(ValueError,'training-only'): prepare_varied(source,config,out)

    def test_evaluation_uses_the_requested_pinned_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_jsonl(root/'train.jsonl',rows())
            model_id='Qwen/Qwen2.5-1.5B-Instruct'
            with patch('beerllm.fit_check.load_model',return_value=(object(),object())) as loader, \
                 patch('beerllm.fit_check.generate',return_value={'text':'Answer','generated_tokens':1,'hit_token_limit':False}), \
                 patch('beerllm.fit_check.loss_stats',return_value={'answer_nll':1,'answer_tokens':1,'total_answer_nll':1}), \
                 contextlib.redirect_stdout(io.StringIO()):
                report=evaluate(root,None,root/'result.json',include_paraphrases=True,model_id=model_id)
            loader.assert_called_once_with(model_id,None,revision=None)
            self.assertEqual(report['model_revision'],model_revision(model_id))
            self.assertEqual(len(report['examples']),4)

    def test_rewordings_preserve_answers_and_do_not_change_training_examples(self):
        examples=rows();original=copy.deepcopy(examples)
        variants=evaluation_rows(examples,True)
        self.assertEqual(len(variants),4)
        self.assertEqual([r['messages'][1]['content'] for r in variants[2:]],PARAPHRASES)
        self.assertEqual(variants[0]['messages'][-1],variants[2]['messages'][-1])
        self.assertEqual(examples,original)
        self.assertEqual(len(evaluation_rows(examples,False)),2)

    def test_heldout_or_ambiguous_examples_are_rejected(self):
        examples=rows()
        self.assertEqual(select_examples(examples),examples)
        examples[0]['split']='test'
        with self.assertRaisesRegex(ValueError,'held-out'):
            select_examples(examples)
        examples=rows();examples.append(copy.deepcopy(examples[0]))
        with self.assertRaisesRegex(ValueError,'Expected one'):
            select_examples(examples)

    def test_training_reuse_is_explicit_and_source_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.jsonl';output=root/'fit'
            examples=rows();write_jsonl(source,examples)
            report=prepare(source,output)
            self.assertIn('validation repeats training',report['purpose'])
            self.assertEqual((output/'train.jsonl').read_bytes(),(output/'validation.jsonl').read_bytes())
            self.assertEqual(prepare(source,output),report)
            examples[0]['messages'][-1]['content']='Changed target.';write_jsonl(source,examples)
            with self.assertRaisesRegex(ValueError,'source changed'):
                prepare(source,output)


if __name__=='__main__': unittest.main()
