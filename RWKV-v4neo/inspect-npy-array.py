########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, copy, types, gc, sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/../rwkv_pip_package/src')
import sys
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file=f'./20B_tokenizer.json')
# tokenizer = PreTrainedTokenizerFast(tokenizer_file=f'./20B_tokenizer.jp.json')
import tqdm

import numpy as np
from diskarray import DiskArray

#sfrom diskarray import DiskArray
#MODEL_NAME = '/data/workspaces/jp/LLMs/rwkv-4-raven/RWKV-4-Raven-3B-v9-Eng99%-Other1%-20230411-ctx4096.pth'
#model = RWKV(model=MODEL_NAME, strategy=strategy)
#pipeline = PIPELINE(model, f"{current_path}/20B_tokenizer.json")
#dd = pipeline.encode(i)
#xxx = pipeline.decode(model_tokens[out_last:])

# to|2|17.3400|17.4000|||LC|[]|||

npy_file = sys.argv[1]
# going strait to list of int
# array = DiskArray(npy_file, dtype=np.uint16).data.astype("int")
eot_id = tokenizer.encode("<|endoftext|>")
array = DiskArray(npy_file, dtype=np.uint16)

beg_of_doc = 0
end_of_doc = None
i=0
num_docs = 0
# while end_of_doc is None and beg_of_doc+i < len(array):

print("try 1")
unique, counts = np.unique(array, return_counts=True)
print(f"we have {len(unique)} unique tokens, end of text has a count of {counts[eot_id]}")

with tqdm.tqdm("tokens processed", total=len(array)) as pbar:
    # just count the number of end-of-text for now
    while i < len(array):
        if array[i] == eot_id:
            num_docs += 1

        if i % 50000 == 0:
            pbar.update(50000)
            pbar.set_postfix({"num_docs": num_docs})
            # print(f"i: {i} num_docs: {num_docs}")

        # print docs
        # if array[beg_of_doc+i] == eot_id:
        #     end_of_doc = beg_of_doc + i
        #     print(f"doc between {beg_of_doc} and {end_of_doc} inclusively")
        #     print("doc is")
        #     print(tokenizer.decode(array[beg_of_doc:end_of_doc+1]))
        #     print("-"*80)
        #     m = input("find the next one ? y/n").strip().lower()
        #     if m == "n":
        #         break
        #     beg_of_doc = end_of_doc + 1
        #     end_of_doc = None
            
        i += 1
print(f"num_docs: {num_docs}")

# 
# print("len")
# print(len(array))
# 
# for i in range(0, len(array), 100):
#     print(f"{i:4d} : {array[i:i+100]}")
# # v = array[5000:5100]
# # print(v)
# # vstr = tokenizer.decode(v)
# # print("vstr:", vstr[0:100])
# # print()
# 