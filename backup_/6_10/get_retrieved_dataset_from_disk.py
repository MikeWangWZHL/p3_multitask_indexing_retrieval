import os
import random
import logging

from datasets import load_from_disk
from transformers import (
    set_seed,
    AutoTokenizer
)

from glob import glob
from tqdm import tqdm
import torch
import json

from retrieval import retrieval_for_disk_dataset

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="visualize output with all templates result json")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="dataset_name",
        required=True,
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="dataset_config_name",
        required=True,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="demonstration number",
        required=True,
    )
    args = parser.parse_args()
    return args





MAX_LENGTH = 1024
N_EXCEED_MAX_LENGTH = 0
N_NOT_ENOUGH_DEMO = 0
N_DISCARD_NOT_ORIGINAL = 0


REMAP_NAMES = {
    "wiki_hop":"wiki_hop_original"
}


def main():

    ### input config ###
    input_dir = "/cephfs/user/mikeeewang/summer_22/workspace/data/p3_subset" # directory storing local dataset on disk
    intermeidate_dir = "/cephfs/user/mikeeewang/summer_22/code/t-zero/evaluation/retrieved_dataset_train_validation/p3_subset_6_6_multichoice_qa_new" # storing dataset with retrieved examples
    shard_names = ['p3_subset_6_6_multichoice_qa_new'] # retrieval_index names

    args = parse_args()
    dataset_name = args.dataset_name
    dataset_config_name = args.dataset_config_name
    EXAMPLE_NUM = args.k # concat how many examples

    if dataset_config_name in ["None","","none"]:
        dataset_config_name = None

    ### output config ### 
    output_dir = '/cephfs/user/mikeeewang/summer_22/workspace/data/p3_finetuning/with_retrieval'
    if dataset_config_name:
        output_name = f'{dataset_name}__{dataset_config_name}__{os.path.basename(input_dir)}__k-{EXAMPLE_NUM}'
        task_full_name = f'{dataset_name}_{dataset_config_name}'
    else:
        output_name = f'{dataset_name}__none__{os.path.basename(input_dir)}__k-{EXAMPLE_NUM}'
        task_full_name = f'{dataset_name}'

    print('full task name:', task_full_name)

    ### load task to templates json for filtering out non-original templates ###
    task_2_templates = json.load(open("/cephfs/user/mikeeewang/summer_22/workspace/data/bigscience_P3_task_2_templates.json"))
    if task_full_name not in task_2_templates:
        print(f"ERROR: unseen task name: {task_full_name}")
        quit()
    task_meta_data = task_2_templates[task_full_name]
    if 'omit_templates' in task_meta_data:
        omit_template_strings = task_meta_data['omit_templates']
    else:
        omit_template_strings = []

    
    ### get the intermidate retrieved dataset ###
    print("task:", task_full_name)
    for p in tqdm(glob(os.path.join(input_dir, f"{task_full_name}*"))):
        
        # check if skip template
        skip = False
        for s in omit_template_strings: 
            if s in os.path.basename(p):
                skip = True
        if skip: 
            print("skip:",p)
            continue
    
        loaded_dataset = load_from_disk(p)
        output_path = os.path.join(intermeidate_dir, os.path.basename(p))
        retrieval_for_disk_dataset(
            loaded_dataset, 
            shard_names, # a list of retrieval database name
            output_path,
            saved_dataset_root = "/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/indexed_datasets",
            saved_index_root = "/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/output_indexing",
            device = torch.device("cuda:7"), # encoder device
            k = 20 # retrieval number
        )


    print("output_dir:",os.path.join(output_dir, output_name))

    ### augmentation -> store augmented dataset ###
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained('bigscience/T0')

    def process(examples):
        global N_EXCEED_MAX_LENGTH
        global N_NOT_ENOUGH_DEMO
        global N_DISCARD_NOT_ORIGINAL

        bs = len(examples['retrieved_examples'])
        # print('batch size:', bs)
        new_inputs_pretokenized = []
        for i in range(bs):
            ### select retrieved examples as demonstration ###
            scores, retrieved_examples = examples["scores"][i], examples['retrieved_examples'][i]
            cands = [
                (
                    scores[ii], 
                    retrieved_examples["inputs_pretokenized"][ii], 
                    retrieved_examples["targets_pretokenized"][ii], 
                    retrieved_examples["dataset_name"][ii],
                    retrieved_examples["template_name"][ii],
                    retrieved_examples["idx"][ii]
                ) 
                for ii in range(len(scores))
            ]
            cands = sorted(cands, key = lambda x: x[0])

            demonstration_pretokenized = []
            
            assert EXAMPLE_NUM <= len(cands)
            ex_i = 0
            chosen_key_set = set()
            while len(demonstration_pretokenized) < EXAMPLE_NUM and ex_i < len(cands):
                cand = cands[ex_i]
                ### check if (dataset_name, idx) is not been chosen
                if (cand[3],cand[5]) not in chosen_key_set:
                    ### check if template is original
                    is_original = True
                    # key = cand[3]
                    # if key in REMAP_NAMES:
                    #     key = REMAP_NAMES[key]
                    # templates_metadata = task_2_templates[key]
                    # if "omit_templates" in templates_metadata:
                    #     for s in templates_metadata['omit_templates']:
                    #         if s == cand[4]:
                    #             is_original = False
                    #             break
                    if is_original:
                        demonstration_pretokenized.append(cand[1].rstrip("\n") + "\n" + cand[2].lstrip("\n"))
                    else:
                        N_DISCARD_NOT_ORIGINAL += 1
                ex_i += 1
            
            if len(demonstration_pretokenized) < EXAMPLE_NUM:
                print(f'not enough demonstrations! got: {len(demonstration_pretokenized)}, expected: {EXAMPLE_NUM}')
                N_NOT_ENOUGH_DEMO += 1
            
            # print(demonstration_pretokenized)
            new_input_pretokenized = '\n\n'.join(demonstration_pretokenized) \
                                     + '\n\n' \
                                     + examples['inputs_pretokenized'][i]
            # print(new_input_pretokenized)
            new_inputs_pretokenized.append(new_input_pretokenized)


        new_inputs = tokenizer(
            new_inputs_pretokenized,
            padding=False,
            max_length=MAX_LENGTH,
            truncation=True,
            add_special_tokens=False
        )
        examples['inputs'] = new_inputs['input_ids']
        examples['inputs_pretokenized'] = new_inputs_pretokenized

        for i in examples['inputs']:
            if len(i) > MAX_LENGTH:
                N_EXCEED_MAX_LENGTH += 1

        # for index in random.sample(range(len(examples)), 3):
        #     logger.info(tokenizer.decode(examples['inputs'][index]) + '\n')

        return examples

    for p in tqdm(glob(os.path.join(intermeidate_dir, f"{task_full_name}*"))):
        # process dataset
        raw_dataset = load_from_disk(p)
        # raw_train_dataset = raw_dataset['train']
        new_dataset = raw_dataset.map(
            process,
            batched=True,
            num_proc=8,
            # remove_columns=column_names
        )
        for index in random.sample(range(len(new_dataset['train'])), 3):
            logger.info(new_dataset['train'][index])
        for index in random.sample(range(len(new_dataset['validation'])), 3):
            logger.info(new_dataset['validation'][index])

        new_dataset.save_to_disk(os.path.join(output_dir, output_name, os.path.basename(p)))

    logger.info(f'{N_EXCEED_MAX_LENGTH}')
    logger.info(f'not enough demo: {N_NOT_ENOUGH_DEMO}')
    logger.info(f'discard not original: {N_DISCARD_NOT_ORIGINAL}')


if __name__ == '__main__':
    main()
