import copy
from pathlib import Path
import tempfile
import unittest
from beerllm.common import write_jsonl
from beerllm.fit_check import PARAPHRASES,QUESTIONS,evaluation_rows,prepare,select_examples


def rows():
    return [{'task':'followup','split':'train','messages':[{'role':'system','content':'Explain recipes.'},
             {'role':'user','content':q},{'role':'assistant','content':'A source-checked answer.'}]} for q in QUESTIONS]


class FitCheckTests(unittest.TestCase):
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
