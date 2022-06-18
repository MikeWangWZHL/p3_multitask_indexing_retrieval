#!/usr/bin/env python
# coding=utf-8
# Copyright BigScience, The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reproduce the main evaluation in `Multitask Prompted Training Enables Zero-Shot Task Generalization` using PyTorch.

This script is heavily adapted from https://github.com/huggingface/transformers/blob/7533d30acd975027e83a548e4c38e06fa335291b/examples/pytorch/multiple-choice/run_swag_no_trainer.py
"""

import argparse
import logging
import os
import random
import json
import copy

import datasets
import torch
from datasets import load_dataset, load_from_disk, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
)
from promptsource.templates import DatasetTemplates

from t0.data_collator import DataCollatorForMultipleChoice
from t0.model import ModelBase

from template_list import template_list
from retrieval import setup_retriever, retrieve, setup_retriever_shard

logger = logging.getLogger(__name__)

from collections import defaultdict
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce main evaluation in T0.")
    parser.add_argument(
        "--shard_names",
        nargs='+', 
        default=[""],
        help="names of the retrieval database subsets",
        required=True,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
        required=True,
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--template_name",
        type=str,
        default=None,
        help="The template/prompt name",
        required=False,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--target_max_length",
        type=int,
        default=256,
        help="Target max length. Sequences longer than this will be truncated."
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--output_retrieved_dataset_root",
        type=str,
        default="/cephfs/user/mikeeewang/summer_22/code/t-zero/evaluation/retrieved_dataset",
        help="where to store the output dataset with retrieved samples"
    )
    parser.add_argument(
        "--saved_dataset_root",
        type=str,
        default = "/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/indexed_datasets",
        help="where to load retrieval database from"
    )
    parser.add_argument(
        "--saved_index_root",
        type=str,
        default="/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/output_indexing",
        help="where to load retrieval database indexing from"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--use_faiss_gpu",
        action="store_true",
        help="whether use device=-1 in load_faiss_index",
    )
    parser.add_argument(
        "--eval_all_templates",
        action="store_true",
        help=(
            "If passed, ignore template name and evaluate all possible templates listed in tempalte_list.py"
        ),
    )
    parser.add_argument(
        "--retrieve_train",
        action="store_true",
        help=(
            "If passed, retrieve for training samples as well"
        ),
    )
    args = parser.parse_args()

    return args

@torch.no_grad()
def main():
    args = parse_args()

    ### retrieval config ### 
    print('shard names:', args.shard_names)
    
    # TODO: make this configurable
    device = torch.device("cuda:7")
    k = 20 # retrieval number

    if args.dataset_config_name in ["None","","none"]:
        args.dataset_config_name = None

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if args.dataset_name == "anli":
            raw_datasets = load_dataset(args.dataset_name, split=args.dataset_config_name)
        else:
            if not args.retrieve_train:
                print('NOTE: only do retrieval for validation ... ')
                raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, split="validation")
            else:
                print("NOTE: retrieve for all splits")
                if args.dataset_config_name:
                    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
                else:
                    raw_datasets = load_dataset(args.dataset_name)

    print(raw_datasets)

    # Trim a number of evaluation examples
    if args.debug:
        raw_datasets = raw_datasets.select(range(100))

    column_names = raw_datasets.column_names
    if isinstance(column_names, dict):
        column_names = list(column_names.values())[0]
    print(column_names)

    # Get the prompt to apply and the possible targets.
    prompts = DatasetTemplates(
        f"{args.dataset_name}"
        if args.dataset_config_name is None
        else f"{args.dataset_name}/{args.dataset_config_name}"
    )

    assert (args.dataset_name, args.dataset_config_name) in template_list

    if args.eval_all_templates:
        template_names = template_list[(args.dataset_name, args.dataset_config_name)]
        print(f'evaluating all possible templates for original task, total number:{len(template_names)}')
    else:
        assert args.template_name is not None
        template_names = [args.template_name]


    # loop over templates
    for template_name in template_names:
        print(f'INFO: tempalte {template_name} ...')
        template = prompts[template_name]

        # output config
        output_dir = os.path.join(args.output_retrieved_dataset_root,".".join(args.shard_names))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        template_name_in_path = template_name.replace(' ', '-')
        if args.dataset_config_name:
            output_name = args.dataset_name + "__" + args.dataset_config_name + "__" + template_name_in_path
        else:
            output_name = args.dataset_name + "__" + template_name_in_path

        output_disk_path = os.path.join(output_dir, output_name)
        if os.path.exists(output_disk_path):
            print('INFO: skipping already exists dataset:', output_disk_path)
            continue

        # loop over retrieval databases
        processed_dataset = copy.deepcopy(raw_datasets)
        for shard_name in args.shard_names:
            print('INFO: retrieving from:', shard_name)
            loaded_retrieval_dataset, q_encoder, q_tokenizer = setup_retriever_shard(shard_name, args.saved_dataset_root, args.saved_index_root, device, use_faiss_gpu=args.use_faiss_gpu)    
            ### preprocess dataset functions ###
            @torch.no_grad()
            def preprocess_function(examples):
                bs = len(examples[column_names[0]])
                print('batch processing size:', bs)
                scores_batch = []
                retrieved_examples_batch = []
                for i in range(bs):
                    ex = {
                        k: examples[k][i]
                        for k in column_names
                    }
                    input, target = template.apply(ex)
                    # ex_answer_choices = template.get_answer_choices_list(ex)
                    # assert target in ex_answer_choices
                    scores, retrieved_examples = retrieve(loaded_retrieval_dataset, q_encoder, q_tokenizer, input, device, topk=k)
                    scores_batch.append(list(scores))
                    retrieved_examples_batch.append(retrieved_examples)

                if "retrieved_examples" not in examples:
                    examples["retrieved_examples"] = [{k:[] for k in retrieved_examples_batch[0].keys()} for idx in range(bs)]
                    examples["scores"] = [[] for idx in range(bs)]

                examples["scores"] = [
                    examples["scores"][idx] + scores_batch[idx] for idx in range(bs)
                ]
                examples["retrieved_examples"] = [
                    {
                        k: value + retrieved_examples_batch[idx][k] for k,value in examples["retrieved_examples"][idx].items()
                    } for idx in range(bs)
                ]

                return examples

            print('INFO: preparing dataset ...')
            processed_dataset = processed_dataset.map(
                preprocess_function, batched=True
            )
        
        # Log a few random samples from the eval set:
        if "train" in processed_dataset:
            print(f"Sample 0 of the training set: {processed_dataset['train'][0]}.")

        # save to disk
        print('save to disk ...')
        processed_dataset.save_to_disk(output_disk_path)
        
        del loaded_retrieval_dataset, processed_dataset, q_encoder, q_tokenizer

    
if __name__ == "__main__":
    main()



