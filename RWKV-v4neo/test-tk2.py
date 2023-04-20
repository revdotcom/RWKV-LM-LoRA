########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, copy, types, gc, sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/../rwkv_pip_package/src')
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file=f'./20B_tokenizer.json')
# tokenizer = PreTrainedTokenizerFast(tokenizer_file=f'./20B_tokenizer.jp.json')

import numpy as np
from prompt_toolkit import prompt

#strategy = 'cuda fp16'
#MODEL_NAME = '/data/workspaces/jp/LLMs/rwkv-4-raven/RWKV-4-Raven-3B-v9-Eng99%-Other1%-20230411-ctx4096.pth'
#model = RWKV(model=MODEL_NAME, strategy=strategy)
#pipeline = PIPELINE(model, f"{current_path}/20B_tokenizer.json")
#dd = pipeline.encode(i)
#xxx = pipeline.decode(model_tokens[out_last:])

# to|2|17.3400|17.4000|||LC|[]|||



mystr = ["ceci est un test", "this is a test", "<|endoftext|>"]
v = tokenizer.encode(list(mystr))
print(f"len(v) = {len(v)}")
arr = np.array(v, dtype='uint16')
print(f"shape : {arr.shape}")
print(arr)

