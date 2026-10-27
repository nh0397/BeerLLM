import json
from pathlib import Path
import tempfile
import unittest
from beerllm.common import ROOT,write_json
from beerllm.train import initial_adapter_signature


class AdapterContinuationTests(unittest.TestCase):
    def test_continuation_pins_parent_weights_and_rejects_incompatible_architecture(self):
        config=json.loads((ROOT/'configs/qwen-helles-followups.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            adapter={'base_model_name_or_path':config['model_id'],'peft_type':'LORA','task_type':'CAUSAL_LM',
                     'r':16,'lora_alpha':32,'lora_dropout':0.0,'revision':config['model_revision'],
                     'target_modules':list(reversed(config['target_modules']))}
            write_json(root/'adapter_config.json',adapter);write_json(root/'beerllm.json',{'format_version':1})
            (root/'adapter_model.safetensors').write_bytes(b'parent weights')
            before=initial_adapter_signature(root,config)
            (root/'adapter_model.safetensors').write_bytes(b'changed parent weights')
            self.assertNotEqual(before,initial_adapter_signature(root,config))
            with self.assertRaisesRegex(ValueError,'settings differ'):initial_adapter_signature(root,dict(config,lora_rank=32))
            adapter['revision']='0'*40;write_json(root/'adapter_config.json',adapter)
            with self.assertRaisesRegex(ValueError,'revision differs'):initial_adapter_signature(root,config)
            adapter['revision']=config['model_revision'];adapter['target_modules']=['q_proj'];write_json(root/'adapter_config.json',adapter)
            with self.assertRaisesRegex(ValueError,'target modules differ'):initial_adapter_signature(root,config)


if __name__=='__main__':unittest.main()
