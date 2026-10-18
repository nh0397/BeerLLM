import unittest
from beerllm.train import encode_example


class TinyTokenizer:
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        text=''.join(f"<{m['role']}>{m['content']}<end>" for m in messages)
        return text + ('<assistant>' if add_generation_prompt else '')
    def __call__(self,text,add_special_tokens=False):
        return {'input_ids':[ord(c) for c in text]}


class TrainingTests(unittest.TestCase):
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
