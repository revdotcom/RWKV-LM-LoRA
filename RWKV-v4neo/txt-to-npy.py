#@title Training Data Options
#@markdown `input_file` should be the path to a single file that contains the text you want to fine-tune with.
#@markdown Either upload a file to this notebook instance or reference a file in your Google drive.

import numpy as np
from transformers import PreTrainedTokenizerFast

repo_dir="."

tokenizer = PreTrainedTokenizerFast(tokenizer_file=f'{repo_dir}/20B_tokenizer.json')

#input_file = "conversation_pt2.txt" #@param {type:"string"}
input_file = "/data/workspaces/jp/LLMs/normalised_chatml_rwkvready/summarisation.txt"
output_file = input_file + ".20B.npy"

print(f'Tokenizing {input_file} (VERY slow. please wait)')

data_raw = open(input_file, encoding="utf-8").read()
print(f'Raw length = {len(data_raw)}')

data_code = tokenizer.encode(data_raw)
print(f'Tokenized length = {len(data_code)}')

out = np.array(data_code, dtype='uint16')
np.save(output_file, out, allow_pickle=False)
