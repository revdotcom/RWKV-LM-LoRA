#@title Training Data Options
#@markdown `input_file` should be the path to a single file that contains the text you want to fine-tune with.
#@markdown Either upload a file to this notebook instance or reference a file in your Google drive.

import numpy as np
from transformers import PreTrainedTokenizerFast
import glob
import argparse
from diskarray import DiskArray
import pathlib
import tqdm
from nlprep.nlp import NLPReader
import sys
from io import StringIO
import random
import multiprocessing as mp
from multiprocessing import Queue

repo_dir="."
tokenizer = PreTrainedTokenizerFast(tokenizer_file=f'{repo_dir}/20B_tokenizer.json')


# using argparse, collect the nlpfolder, output_file and taskname parameters
def get_args():
    parser = argparse.ArgumentParser("process NLP files and prepare task for RWKV finetuning")
    parser.add_argument("--nlpfolder", type=str, default="nlpfiles")
    parser.add_argument("--output_file", type=str, default="data.npy")
    # probably better to have on/off tasks for individual elements
    # parser.add_argument("--taskname", type=str, choices=["punctuation", "casing"], default="punctuation")
    args = parser.parse_args()
    return args


def plain_nlp_to_tokens(input_file, tokenizer):
    data_raw = open(input_file, encoding="utf-8").read()
    data_code = tokenizer.encode(data_raw)
    out = np.array(data_code, dtype='uint16')
    return out

def read_nlp_file(input_file):
    with open(input_file, "r") as f:
        nlpr = list(NLPReader(f))
    return nlpr

def nlp_remove_case_information(nlp):
    for row in nlp:
        row.data['case'] = ''
        row.data['token'] = row.data['token'].lower()
    return nlp

def nlp_remove_punctuation(nlp):
    for row in nlp:
        row.data['punctuation'] = ''
        if 'prepunctuation' in row.data:
            row.data['prepunctuation'] = ''
    return nlp

# remove entirely a field from the nlp rows (including in the header)
def nlp_drop_field(nlp, fieldname):
    for row in nlp:
        if fieldname in row.data:
            del row.data[fieldname]
        else:
            # no need to continue, if the first row doesn't have the field
            # the rest won't either
            break
    return nlp

# set the content of a field, if it exists, to a given value
def nlp_mute_field(nlp, fieldname, mute_value=''):
    for row in nlp:
        if fieldname in row.data:
            row.data[fieldname] = mute_value
        else:
            # no need to continue, if the first row doesn't have the field
            # the rest won't either
            break
    return nlp


def print_nlp(nlp):
    for rowno, row in enumerate(nlp):
        if rowno == 0:
            keys_order = row.data.keys()
            print("|".join(keys_order))
        print("|".join(row.data[key] for key in keys_order))
def nlp_to_single_string(nlp):
    # open a StringIO object and write the nlp to it

    output = StringIO()
    for rowno, row in enumerate(nlp):
        if rowno == 0:
            keys_order = row.data.keys()
            output.write("|".join(keys_order))
        output.write('\n' + "|".join(row.data[key] for key in keys_order))

    new_nlp_str = output.getvalue()
    output.close()
    return new_nlp_str

def nlp_to_single_string_plain(nlp, split_on_speaker_change=False):
    # open a StringIO object and write the nlp to it

    output = StringIO()
    prev_spk = None
    first_word = True
    for rowno, row in enumerate(nlp):
        if rowno == 0:
            pass
        
        spk_id = row.data['speaker']
        pre_punct = row.data['prepunctuation']
        punct = row.data['punctuation']
        token = row.data['token']
        if prev_spk is None:
            prev_spk = spk_id

        if split_on_speaker_change and spk_id != prev_spk:
            output.write('\n')
            first_word = True
            prev_spk = spk_id
        if not first_word:
            output.write(' ')
        output.write(pre_punct + token + punct)
        first_word=False

    new_nlp_str = output.getvalue()
    output.close()
    return new_nlp_str
    
# let's drop the ali_comment column, mute tags as our source nlp file
# then let's use that as the target.  after, we drop punc and case info
# and perhaps the speaker as well
def TASK_get_case_and_punctuate_raw_text(input_file):
    nlp = read_nlp_file(input_file)

    # refenrece "target" nlp to generate
    ref_txt = nlp_to_single_string_plain(nlp)

    nlp = nlp_remove_punctuation(nlp)
    nlp = nlp_remove_case_information(nlp)

    # what one would perhaps generate from a ctm (although confidences are missing, 
    # that's a different task perhaps!
    hyp_txt = nlp_to_single_string_plain(nlp)

    task_string = "\n".join([
        "Here is an Englis text where punctuatation and case information is missing. Rewrite it with the proper punctuation marks and put the words in their proper case.",
        "<|input|>:",
        hyp_txt,
        "<|output|>:",
        ref_txt,
        "<|endoftext|>"
    ])
    return task_string

# 

# let's drop the ali_comment column, mute tags as our source nlp file
# then let's use that as the target.  after, we drop punc and case info
# and perhaps the speaker as well
def TASK_get_case_and_punctuate(input_file, mute_speaker_prob = 0.5, mute_timings_prob = 0.5):
    nlp = read_nlp_file(input_file)

    nlp = nlp_drop_field(nlp, 'ali_comment')
    nlp = nlp_mute_field(nlp, 'tags')

    # We mask these fields before taking the "reference" nlp snapshot
    # as we don't necessarily want to train "generating timings" or "generating speakers"
    # although the diarization aspect could be interesting...
    if random.random() < mute_speaker_prob:
        nlp = nlp_mute_field(nlp, 'speaker')

    if random.random() < mute_timings_prob:
        nlp = nlp_mute_field(nlp, 'ts')
        nlp = nlp_mute_field(nlp, 'endTs')

    # refenrece "target" nlp to generate
    ref_nlp = nlp_to_single_string(nlp)

    nlp = nlp_remove_punctuation(nlp)
    nlp = nlp_remove_case_information(nlp)

    # what one would perhaps generate from a ctm (although confidences are missing, 
    # that's a different task perhaps!
    hyp_nlp = nlp_to_single_string(nlp)

    task_string = "\n".join([
        "here is a normalized nlp file, add casing information and punctuation information in the right places",
        "<|input|>:",
        hyp_nlp,
        "<|output|>:",
        ref_nlp,
        "<|endoftext|>"
    ])
    return task_string

# no real task here, just a view on an nlp file with various fields on/off
def TASK_simple_nlp_presentation(input_file,
            mute_speaker_prob = 0.0,
            mute_timings_prob = 0.5,
            mute_tags_prob = 0.5,
             ):
    nlp = read_nlp_file(input_file)

    # we don't want this
    nlp = nlp_drop_field(nlp, 'ali_comment')

    if random.random() < mute_tags_prob:
        nlp = nlp_mute_field(nlp, 'tags')

    if random.random() < mute_speaker_prob:
        nlp = nlp_mute_field(nlp, 'speaker')

    if random.random() < mute_timings_prob:
        nlp = nlp_mute_field(nlp, 'ts')
        nlp = nlp_mute_field(nlp, 'endTs')

    # refenrece "target" nlp to generate
    ref_nlp = nlp_to_single_string(nlp)

    task_string = "\n".join([
        "here is a simple normalized nlp file:",
        ref_nlp,
        "<|endoftext|>"
    ])
    return task_string


# Will decide how to handle a specific file and return the data code
# TODO: take parameters to define how to split the tasks
def read_nlp_and_prepare_task(f):
    try :
        r = random.random()
        if r < 0.33:
            task_str = TASK_simple_nlp_presentation(f)
        elif r < 0.66:
            task_str = TASK_get_case_and_punctuate(f)
        else:
            task_str = TASK_get_case_and_punctuate_raw_text(f)

        data_code = np.array(tokenizer.encode(task_str), dtype='uint16')
        return data_code
    except:
        return None


## just some quick debugging code
if False and __name__ == "__main__":
    input_file = "50lines.nlp"
    # nlp = read_nlp_file(input_file)
    # plain_nlp = open(input_file, encoding="utf-8").read()

    # print("original nlp")
    # print(plain_nlp)

    # print("original nlp, parsed")
    # print_nlp(nlp)

    # nlp= nlp_drop_field(nlp, 'ali_comment')
    # print("dropping ali_comment")
    # print_nlp(nlp)

    # nlp = nlp_remove_punctuation(nlp)
    # print("removing punct")
    # print_nlp(nlp)

    # nlp = nlp_remove_case_information(nlp)
    # print("removing case")
    # print_nlp(nlp)

    # nlp_str = nlp_to_single_string(nlp)
    # # print("final as single string again")
    # print(nlp_str)

    for i in range(10):
        print()
        print("-"*80)
        print()
        # print(TASK_get_case_and_punctuate(input_file))
        # print(TASK_simple_nlp_presentation(input_file))
        print(TASK_get_case_and_punctuate_raw_text(input_file))
        print()

# Let's try a parallel version of that code
if True and __name__ == "__main__":
# "/data/workspaces/jp/LLMs/normalised_chatml_rwkvready/summarisation.txt"
    args = get_args()

    input_folder = args.nlpfolder
    output_file = args.output_file

    # check if output file already exists
    if glob.glob(output_file):
        print(f'Output file {output_file} already exists, appending')
        arr = DiskArray('disk.array', mode='r+', dtype=np.uint16, growby=256*1024*1024)
    else:
        print(f'Output file {output_file} does not exist, creating')
        # arr = DiskArray(output_file, mode='r+', dtype=np.uint16, shape=(0, 4*1024), growby=256*1024*1024)
        arr = DiskArray(output_file, mode='r+', dtype=np.uint16, growby=256*1024*1024)

    p0 = len(arr)
    print("initial array length = ", p0)

    total_tokens = 0

    files_to_process = list(pathlib.Path(input_folder).rglob("*.nlp")) 
    # Create a queue to hold the processed numpy arrays

    # Create a pool of worker processes
    pbar = tqdm.tqdm("Processing nlp files: ", total = len(files_to_process))
    # on hydra, 8 processes gives about 130 files/sec
    # on hydra, 16 processes gives about 180 files/sec
    failures = 0
    with mp.Pool(processes=16) as pool:
        # Loop over the files in parallel and process each one
        # Use the imap_unordered() method to process the files
        for fno, data_code in enumerate(pool.imap_unordered(read_nlp_and_prepare_task, files_to_process)):
            if data_code is None:
                failures += 1
                continue
            total_tokens += len(data_code)
            arr.extend(data_code)
            if fno % 100 == 0:
                pbar.update(100)
                pbar.set_postfix({"total_tokens": total_tokens, "array_length_GB": f"{len(arr)/1024**3:.2f} GB"})
            # if fno > 500:
            #     break

    arr.flush()
    print(f"total tokens written/added = {total_tokens}")
    print(f"total failures = {failures}")
        # np.save(output_file, out, allow_pickle=False)


# slow single-threaded version
if False and __name__ == "__main__":
# "/data/workspaces/jp/LLMs/normalised_chatml_rwkvready/summarisation.txt"
    args = get_args()

    input_folder = args.nlpfolder
    output_file = args.output_file

    # check if output file already exists
    if glob.glob(output_file):
        print(f'Output file {output_file} already exists, appending')
        arr = DiskArray('disk.array', mode='r+', dtype=np.uint16, growby=256*1024*1024)
    else:
        print(f'Output file {output_file} does not exist, creating')
        arr = DiskArray(output_file, mode='r+', dtype=np.uint16, shape=(64*1024,), growby=256*1024*1024)

    p0 = len(arr)
    print("initial array length = ", p0)

    # using glob, traverse all the folders and subfolders below input_folder for all files with the .nlp extension
    base_txt = "here a regular normalized aligned nlp file:\n"
    base_txt_arr = np.array(tokenizer.encode(base_txt), dtype='uint16')
    end_txt = "\n<|endoftext|>\n\n"
    end_txt_arr =np.array(tokenizer.encode(end_txt), dtype='uint16')
    total_tokens = 0

    file_strings = []
    concat_strings = ""
    with tqdm.tqdm("processing nlp files: ") as pbar:
        for fno, f in enumerate(pathlib.Path(input_folder).rglob("*.nlp")):

            if random.random() < 0.5:
                task_str = TASK_simple_nlp_presentation(f)
            else:
                task_str = TASK_get_case_and_punctuate(f)

            data_code = np.array(tokenizer.encode(task_str), dtype='uint16')
            total_tokens += len(data_code)
            arr.extend(data_code)

            if fno % 500 == 0:
                pbar.update(500)

    print(f"total tokens written/added = {total_tokens}")
        # np.save(output_file, out, allow_pickle=False)



 

# if main
if False and __name__ == "__main__":
# "/data/workspaces/jp/LLMs/normalised_chatml_rwkvready/summarisation.txt"
    args = get_args()

    input_folder = args.nlpfolder
    output_file = args.output_file

    # check if output file already exists
    if glob.glob(output_file):
        print(f'Output file {output_file} already exists, appending')
        arr = DiskArray('disk.array', mode='r+', dtype=np.uint16, growby=256*1024*1024)
    else:
        print(f'Output file {output_file} does not exist, creating')
        arr = DiskArray(output_file, mode='r+', dtype=np.uint16, shape=(64*1024,), growby=256*1024*1024)

    p0 = len(arr)
    print("initial array length = ", p0)

    # using glob, traverse all the folders and subfolders below input_folder for all files with the .nlp extension
    base_txt = "here a regular normalized aligned nlp file:\n"
    base_txt_arr = np.array(tokenizer.encode(base_txt), dtype='uint16')
    end_txt = "\n<|endoftext|>\n\n"
    end_txt_arr =np.array(tokenizer.encode(end_txt), dtype='uint16')
    total_tokens = 0

    file_strings = []
    concat_strings = ""
    with tqdm.tqdm("processing nlp files: ") as pbar:
        for fno, f in enumerate(pathlib.Path(input_folder).rglob("*.nlp")):
            # print(f'Tokenizing {f} (VERY slow. please wait)')

            data_raw = open(f, encoding="utf-8").read()
            # print(f'Raw length = {len(data_raw)}')
            if False:
                file_strings.append(base_txt)
                file_strings.append(data_raw)
                file_strings.append(end_txt)
            else:
                concat_strings += base_txt + data_raw + end_txt

            # data_code = np.array(tokenizer.encode(data_raw), dtype='uint16')
            # arr.extend(base_txt_arr)
            # arr.extend(data_code)
            # arr.extend(end_txt_arr)
            # total_tokens += len(data_code) + len(base_txt_arr) + len(end_txt_arr)
            if fno % 500 == 0:
                # data_code = np.array(tokenizer.encode("".join(file_strings)), dtype='uint16')
                data_code = np.array(tokenizer.encode(concat_strings), dtype='uint16')
                file_strings = []
                concat_strings = ""
                total_tokens += len(data_code)
                arr.extend(data_code)

                pbar.update(500)

        if len(file_strings) > 0:
            # data_code = np.array(tokenizer.encode("".join(file_strings)), dtype='uint16')
            data_code = np.array(tokenizer.encode(concat_strings), dtype='uint16')
            file_strings = []
            concat_strings = ""
            total_tokens += len(data_code)
            arr.extend(data_code)

    print(f"total tokens written/added = {total_tokens}")
        # np.save(output_file, out, allow_pickle=False)



#gpt-generated code :

if False:
    # Create an empty memmap array with a size of 10 and dtype of float32
    arr = np.memmap('data.npy', dtype='float32', mode='w+', shape=(10,))

    # Write values to the memmap array
    arr[0:5] = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Flush changes to disk
    arr.flush()

    # Append values to the memmap array
    arr[5:] = [6.0, 7.0, 8.0, 9.0, 10.0]

    # Flush changes to disk
    arr.flush()

    # Delete the memmap object to free up memory
    del arr

    # Load the memmap array from disk
    arr = np.memmap('data.npy', dtype='float32', mode='r', shape=(10,))

    # Print the values in the memmap array
    print(arr)




