from collections import defaultdict
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

from retrieval import retrieval_for_disk_dataset, retrieve_template_str

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

import argparse


def encode_right_truncated(tokenizer, texts, padding=False, max_length=1024, add_special_tokens=False):
    ret_ids = []
    for text in texts:
        tokenized = tokenizer.tokenize(text, padding=padding, add_special_tokens=add_special_tokens)

        if not add_special_tokens:
            truncated = tokenized[-max_length:]
        else:
            truncated = tokenized[0:1] + tokenized[-(max_length-1):]

        ids = tokenizer.convert_tokens_to_ids(truncated)
        assert len(ids) <= max_length
        ret_ids.append(ids)
    return ret_ids


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
        "--train",
        action="store_true",
        help="augmenting for training tasks",
    )
    parser.add_argument(
        "--no_demo",
        action="store_true",
        help="if original",
    )
    parser.add_argument(
        "--use_original_eval_template",
        action="store_true",
        help="if use original template instead of the retrieved on for evaluation tasks",
    )
    args = parser.parse_args()
    return args





MAX_LENGTH = 1024
N_EXCEED_MAX_LENGTH = 0
# N_NOT_ENOUGH_DEMO = 0
# N_DISCARD_NOT_ORIGINAL = 0


REMAP_NAMES = {
    "wiki_hop":"wiki_hop_original"
}

import re
def load_task_template_name_dict():
    res = defaultdict(dict)
    with open("/cephfs/user/mikeeewang/summer_22/workspace/data/p3_template_hf_dataset/template_dataset_v1.1.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            dataset_name, dataset_config_name = obj["dataset_name"], obj["dataset_config_name"]
            if dataset_config_name:
                task_full_name = f'{dataset_name}_{dataset_config_name}'
            else:
                task_full_name = f'{dataset_name}'
            tn = obj['template_name'].replace('-', '_').replace(' ', '_').replace('/', '_').replace('___', '_')
            tn = re.sub(r"[^\w\d'\s\_]+", '', tn).strip('_')
            res[task_full_name][tn] = obj['template_str']
    return res

def get_training_used_templates(dataset_mixture_config_path, task_template_name_dict):
    dataset_mixture_config = json.load(open(dataset_mixture_config_path))
    ret_dict = {}
    for p in dataset_mixture_config["sampled_list"]:
        task_name = os.path.basename(os.path.dirname(p))
        dataset_name, dataset_config_name = task_name.split("__")[0],task_name.split("__")[1]

        if dataset_config_name in ["None","none"]:
            dataset_config_name = None
        if dataset_config_name:
            task_full_name = f'{dataset_name}_{dataset_config_name}'
        else:
            task_full_name = f'{dataset_name}'

        template_name = os.path.basename(p).replace(f'{task_full_name}_', '').replace('_score_eval', '').strip('_')
        if dataset_name == 'anli':
            template_name = template_name.replace('_r1', '').replace('_r2', '').replace('_r3', '')

        template_str = task_template_name_dict[task_full_name][template_name]
        ret_dict[f"{task_full_name}_{template_name}"] = {
            "task_full_name":task_full_name,
            "template_name":template_name,
            "template_str":template_str
        }
    return ret_dict


def main():
    args = parse_args()

    ### input config ###
    input_dir = "/cephfs/user/mikeeewang/summer_22/workspace/data/p3_subset" # directory storing local dataset on disk

    ### load task tempalte name to template str dict
    task_template_name_dict = load_task_template_name_dict()

    if not args.train:
        training_model_name = "mulcqa_mixture_template_augmented_6-12_n-2"
        output_index_dir = f"/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/output_indexing/{training_model_name}"
        dataset_mixture_config_path = f'/cephfs/user/mikeeewang/summer_22/code/MultitaskGenerativeMoE/output/p3_finetuning/with_template_augmented/t5-base/{training_model_name}/dataset_mixture_config.json'
        candidate_template_dict = get_training_used_templates(dataset_mixture_config_path, task_template_name_dict)
        os.makedirs(output_index_dir, exist_ok=True)


    dataset_name = args.dataset_name
    dataset_config_name = args.dataset_config_name
    if args.no_demo:
        output_name_suffix = "no_demo"
    else:
        if args.use_original_eval_template:
            output_name_suffix = "template_augmented_use_original"
        else:
            output_name_suffix = "template_augmented_use_retrieved"

    if dataset_config_name in ["None","","none"]:
        dataset_config_name = None

    ### output config ###
    output_dir = '/cephfs/user/mikeeewang/summer_22/workspace/data/p3_finetuning/with_template_augmented'
    if dataset_config_name:
        output_name = f'{dataset_name}__{dataset_config_name}__{os.path.basename(input_dir)}__{output_name_suffix}'
        task_full_name = f'{dataset_name}_{dataset_config_name}'
    else:
        output_name = f'{dataset_name}__none__{os.path.basename(input_dir)}__{output_name_suffix}'
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



    print("output_dir:",os.path.join(output_dir, output_name))

    ### augmentation -> store augmented dataset ###
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained('bigscience/T0')

    for p in tqdm(glob(os.path.join(input_dir, f"{task_full_name}*"))):

        # check if skip template
        skip = False
        for s in omit_template_strings:
            if s in os.path.basename(p):
                skip = True
        if skip:
            print("skip:",p)
            continue

        # output_path = os.path.join(intermeidate_dir, os.path.basename(p))
        template_name = os.path.basename(p).replace(f'{task_full_name}_', '').replace('_score_eval', '').strip('_')
        if dataset_name == 'anli':
            template_name = template_name.replace('_r1', '').replace('_r2', '').replace('_r3', '')

        if args.train:
            template_str = task_template_name_dict[task_full_name][template_name]
        else:
            ## use own template
            template_str = task_template_name_dict[task_full_name][template_name]
            if not args.use_original_eval_template:
                ## use retrieved simiar template
                print('-----')
                print(template_str)
                # print(candidate_template_dict)
                scores, retrieved_templates = retrieve_template_str(candidate_template_dict, template_str, output_index_dir)
                template_str = retrieved_templates['template_str'][0]
                retrieved_task_full_name = retrieved_templates['task_full_name'][0]
                retrieved_template = retrieved_templates['template_name'][0]
                print('-----')
                print(template_str)
                print('-----')

        print(template_name)

        def process(examples, idx):

            if args.no_demo:
                bs = len(examples['inputs_pretokenized'])
                chosen_examples = []
                new_inputs_pretokenized = []
                for i in range(bs):
                    new_inputs_pretokenized.append(examples['inputs_pretokenized'][i])
                    chosen_examples.append([])
            else:
                global N_EXCEED_MAX_LENGTH
                bs = len(examples['inputs_pretokenized'])
                # print('batch size:', bs)
                new_inputs_pretokenized = []
                chosen_examples = []
                for i in range(bs):
                    original_inputs_pretokenized = examples['inputs_pretokenized'][i]
                    # print(demonstration_pretokenized)
                    new_input_pretokenized = template_str \
                                            + '\n\n' \
                                            + original_inputs_pretokenized
                    # print(new_input_pretokenized)
                    new_inputs_pretokenized.append(new_input_pretokenized)
                    # chosen_examples.append(template_str)
                    if args.use_original_eval_template:
                        chosen_examples.append({
                            "task_full_name":task_full_name,
                            "template_name":template_name,
                            "template_str":template_str

                        })
                    else:
                        chosen_examples.append({
                            "task_full_name":retrieved_task_full_name,
                            "template_name":retrieved_template,
                            "template_str":template_str

                        })


            ### truncate from beginning ###
            new_input_ids = encode_right_truncated(
                tokenizer,
                new_inputs_pretokenized,
                padding=False,
                max_length=MAX_LENGTH,
                add_special_tokens=False
            )

            examples['inputs'] = new_input_ids
            examples['inputs_pretokenized'] = new_inputs_pretokenized
            # store the chosen examples
            examples['chosen_examples'] = chosen_examples

            for i in examples['inputs']:
                if len(i) == MAX_LENGTH:
                    N_EXCEED_MAX_LENGTH += 1

            return examples


        # process dataset
        raw_dataset = load_from_disk(p)
        # raw_train_dataset = raw_dataset['train']
        new_dataset = raw_dataset.map(
            process,
            batched=True,
            num_proc=8,
            with_indices=True
            # remove_columns=column_names
        )
        for index in random.sample(range(len(new_dataset['train'])), 3):
            logger.info(new_dataset['train'][index])
        for index in random.sample(range(len(new_dataset['validation'])), 3):
            logger.info(new_dataset['validation'][index])

        new_dataset.save_to_disk(os.path.join(output_dir, output_name, os.path.basename(p)))

    logger.info(f'{N_EXCEED_MAX_LENGTH}')


if __name__ == '__main__':
    main()
