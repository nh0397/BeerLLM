import unittest
from beerllm.train import encode_example,replay_short_facts


class TinyTokenizer:
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        text=''.join(f"<{m['role']}>{m['content']}<end>" for m in messages)
        return text + ('<assistant>' if add_generation_prompt else '')
    def __call__(self,text,add_special_tokens=False):
        return {'input_ids':[ord(c) for c in text]}


class TrainingTests(unittest.TestCase):
    def test_short_fact_replay_weights_existing_rows_without_mutation_or_heldout_data(self):
        rows=[{'task':'followup','split':'train','messages':[]},
              {'task':'named_recipe','split':'train','messages':[]},
              {'task':'source_fact','split':'train','messages':[]}]
        replayed=replay_short_facts(rows,2)
        self.assertEqual(len(rows),3)
        self.assertEqual(len(replayed),4)
        self.assertIs(replayed[-1],rows[0])
        self.assertEqual(replay_short_facts(rows,1),rows)
        for repeat in [0,-1,True,1.5]:
            with self.assertRaises(ValueError): replay_short_facts(rows,repeat)
        with self.assertRaisesRegex(ValueError,'held-out'):
            replay_short_facts([{'task':'followup','split':'test'}],2)

    def setUp(self):
        self.row={'messages':[{'role':'user','content':'Question'},{'role':'assistant','content':'Answer'}]}
        self.tok=TinyTokenizer()
    def test_only_answer_is_supervised(self):
        item=encode_example(self.tok,self.row,100)
        actual=''.join(chr(i) for i in item['labels'] if i!=-100)
        self.assertEqual(actual,'Answer<end>')
        self.assertEqual(len(item['labels']),len(item['input_ids']))
    def test_overlength_is_not_truncated(self):
        self.assertIsNone(encode_example(self.tok,self.row,10))


if __name__=='__main__': unittest.main()
