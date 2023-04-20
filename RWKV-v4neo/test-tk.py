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



while True:
    mystr = input("Enter a string of text: ")
    mystr = mystr.strip()

    if "!read:" in mystr:
        filename = mystr[6:].strip()
        print(f"reading file {filename} ...")
        mystr = open(filename).read()

    #v = pipeline.encode(mystr)
    v = tokenizer.encode(mystr)
    print(f"len input {len(v)} tokens\n")
    if len(v) < 100:
        for i in range(len(v)): 
            #print(f"{i:4d} : [{pipeline.decode([v[i]])}]")
            print(f"{i:4d} : [{tokenizer.decode([v[i]])}]")
    else:
        print("too many tokens to print!")
    print()
