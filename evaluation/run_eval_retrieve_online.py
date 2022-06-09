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

import datasets
import torch
from datasets import load_dataset, load_metric
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


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce main evaluation in T0.")
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
        required=True,
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default=None,
        help=(
            "name of the prompt version choose from [original, ic_dpr_full_prompt]",
            "original: original zero-shot eval from T0; ic_dpr_full_prompt: adding in-context example retrieved from other tasks using dpr"),
        required=True,
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
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models. The list of T0 variants can be found on `https://huggingface.co/bigscience/T0_3B`",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--parallelize",
        action="store_true",
        help=(
            "If passed, will call `model.parallelize` which splits the model on all GPUs available when applicable (model parallelism). "
            "Note that this feature is still experimental in HF Transformers."
        ),
    )
    parser.add_argument(
        "--eval_all_templates",
        action="store_true",
        help=(
            "If passed, ignore template name and evaluate all possible templates listed in tempalte_list.py"
        ),
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


    # Handle the output directory creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # In distributed evaluation, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if args.dataset_name == "anli":
            raw_datasets = load_dataset(args.dataset_name, split=args.dataset_config_name)
        else:
            raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, split="validation")

    # Trim a number of evaluation examples
    if args.debug:
        raw_datasets = raw_datasets.select(range(100))

    column_names = raw_datasets.column_names


    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "Either `args.config_name` or `args.model_name_or_path` should be provided."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if tokenizer.pad_token is None:
        for token in [tokenizer.eos_token, tokenizer.bos_token, tokenizer.sep_token]:
            if token is not None:
                tokenizer.pad_token = token
        if tokenizer.pad_token is None:
            raise ValueError("Please define a pad token id.")


    model = ModelBase.from_config(
        config=config,
        model_name_or_path=args.model_name_or_path,
        parallelize=args.parallelize
    )
    print('done loading model')

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    
    # Get the prompt to apply and the possible targets.
    prompts = DatasetTemplates(
        f"{args.dataset_name}"
        if args.dataset_config_name is None
        else f"{args.dataset_name}/{args.dataset_config_name}"
    )

    assert (args.dataset_name, args.dataset_config_name) in template_list
    all_results = []
    if args.eval_all_templates:
        template_names = template_list[(args.dataset_name, args.dataset_config_name)]
        print(f'evaluating all possible templates, total number:{len(template_names)}')
    else:
        template_names = [args.template_name]
    
    ### set up retriever ###
    if "dpr" in args.prompt_mode:
        #TODO: make this configurable 
        retriever_device = torch.device("cuda:7")
        retrieve_num = 1
        concat_num = 1
        # saved_index_dir_path="/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/output_indexing/p3_subset_6_1"
        # ds_with_embeddings, q_encoder, q_tokenizer = setup_retriever(
        #     saved_index_dir=saved_index_dir_path,
        #     device=retriever_device
        # )
        shard_name = "p3_subset_6_1"
        saved_index_root = "/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/output_indexing"
        saved_dataset_root = "/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/indexed_datasets"
        saved_index_dir_path = os.path.join(saved_index_root,shard_name)
        ds_with_embeddings, q_encoder, q_tokenizer = setup_retriever_shard(
            shard_name = shard_name, 
            saved_dataset_root = saved_dataset_root, 
            saved_index_root = saved_index_root, 
            device = retriever_device
        )
        print("retriever device:", q_encoder.device)
    else:
        retrieve_num = 0
        concat_num = 0
    if concat_num > 1:
        args.per_device_eval_batch_size = args.per_device_eval_batch_size // 2
        

    # main loop over templates
    retrieval_results = defaultdict(list) # key is the template name, value is the input and the retrieved results
    for template_name in template_names:
        print(f'evaluating tempalte {template_name} ...')
        template = prompts[template_name]
        
        ### preprocess dataset functions ###
        ## original preprocess function ##
        
        def preprocess_function_original(examples):
            bs = len(examples[column_names[0]])

            input_texts = []
            target_texts = []
            answer_choices_texts = []
            for i in range(bs):
                ex = {
                    k: examples[k][i]
                    for k in column_names
                }
                input, target = template.apply(ex)
                ex_answer_choices = template.get_answer_choices_list(ex)
                assert target in ex_answer_choices
                input_texts.append(input)
                target_texts.append(target)
                answer_choices_texts.append(ex_answer_choices)

            tokenized_inputs = tokenizer(
                input_texts,
                padding=padding,
                max_length=args.max_length,
                truncation=True,
                add_special_tokens=False,
            )
            tokenized_targets = [
                tokenizer(
                    ans_choi,
                    padding=True,
                    max_length=args.target_max_length,
                    truncation=True,
                )
                for ans_choi in answer_choices_texts
            ]

            features = {
                k: [
                    [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                    for idx, elem in enumerate(v)
                ]
                for k, v in tokenized_inputs.items()
            }

            features["labels"] = [
                tokenized_targets[idx]["input_ids"]
                for idx in range(bs)
            ]
            features["labels_attention_mask"] = [
                tokenized_targets[idx]["attention_mask"]
                for idx in range(bs)
            ]
            features["targets"] = [
                answer_choices_texts[idx].index(t)
                for idx, t in enumerate(target_texts)
            ]

            return features


        
        @torch.no_grad()
        def preprocess_function_dpr_full_prompt(examples):
            bs = len(examples[column_names[0]])
            # print(bs)
            input_texts = []
            target_texts = []
            answer_choices_texts = []
            for i in range(bs):
                ex = {
                    k: examples[k][i]
                    for k in column_names
                }
                input, target = template.apply(ex)
                ex_answer_choices = template.get_answer_choices_list(ex)
                assert target in ex_answer_choices
                # print('input:', input)
                # print('target:', target)
                
                ## retrieve
                scores, retreived_examples = retrieve(
                    ds_with_embeddings,
                    q_encoder,
                    q_tokenizer,
                    question=input,
                    device=retriever_device,
                    topk=retrieve_num
                )
                
                ## store the retrieval result
                retrieval_results[template_name].append(
                    {
                        "input":input,
                        "target":target,
                        "retrieved_input":retreived_examples["inputs_pretokenized"],
                        "retrieved_target":retreived_examples["targets_pretokenized"],
                        "retrieval_scores":list([str(s) for s in scores])
                    }
                )

                ## use top 1 example to augment:
                prefix = ""
                assert concat_num <= retrieve_num
                for ii in range(concat_num):
                    prefix = prefix + retreived_examples["inputs_pretokenized"][ii] + retreived_examples["targets_pretokenized"][ii]
                    prefix = prefix.rstrip('\n')
                    prefix += '\n\n'
                
                input = prefix + input

                input_texts.append(input)
                target_texts.append(target)
                answer_choices_texts.append(ex_answer_choices)
                # print("input:\n",input)
                # print("target:\n",target)
                # print('\n\n')

            tokenized_inputs = tokenizer(
                input_texts,
                padding=padding,
                max_length=args.max_length,
                truncation=True,
                add_special_tokens=False,
            )
            tokenized_targets = [
                tokenizer(
                    ans_choi,
                    padding=True,
                    max_length=args.target_max_length,
                    truncation=True,
                )
                for ans_choi in answer_choices_texts
            ]

            features = {
                k: [
                    [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                    for idx, elem in enumerate(v)
                ]
                for k, v in tokenized_inputs.items()
            }

            features["labels"] = [
                tokenized_targets[idx]["input_ids"]
                for idx in range(bs)
            ]
            features["labels_attention_mask"] = [
                tokenized_targets[idx]["attention_mask"]
                for idx in range(bs)
            ]
            features["targets"] = [
                answer_choices_texts[idx].index(t)
                for idx, t in enumerate(target_texts)
            ]

            return features



        ### select preprocessing function ###
        if args.prompt_mode == 'original':
            preprocess_function = preprocess_function_original
        elif args.prompt_mode == 'ic_dpr_full_prompt':
            print('using retrieval augmentation: ic_dpr_full_prompt...')
            preprocess_function = preprocess_function_dpr_full_prompt
        else:
            raise NotImplementedError

        ### preprocess dataset ###
        with accelerator.main_process_first():
            print('preparing dataset ...')
            eval_dataset = raw_datasets.map(
                preprocess_function, batched=True, remove_columns=column_names
            )

        # Log a few random samples from the eval set:
        for index in random.sample(range(len(eval_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")

        # DataLoaders creation:
        if args.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            data_collator = DataCollatorForMultipleChoice(
                tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
            )
        
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


        # Use the device given by the `accelerator` object.
        if not args.parallelize:
            model.to(accelerator.device)

        # Prepare everything with our `accelerator`.
        eval_dataloader = accelerator.prepare(eval_dataloader)


        # Metrics
        metric = load_metric("accuracy")

        # Eval!
        total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes

        logger.info("***** Running evaluation *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
        logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_batch_size}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                predictions = model(batch)

            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["targets"]),
            )

            progress_bar.update(1)

        eval_metric = metric.compute()
        accelerator.print(f"Result: {eval_metric}")

        results = {
            "dataset_name": args.dataset_name,
            "dataset_config_name": args.dataset_config_name,
            "template_name": template_name,
            "evaluation": eval_metric,
            "prompt_mode": args.prompt_mode,
            "retrieval_database": None
        }
        if 'dpr' in args.prompt_mode:
            results['retrieval_database'] = saved_index_dir_path
            results['retrieve_num'] = retrieve_num
            results['concat_num'] = concat_num
            
        all_results.append(results)

    ### store retrieval results ###
    if accelerator.is_main_process:
        if args.output_dir is not None:
            output_name = f"retrieval_results.json"
            with open(os.path.join(args.output_dir, output_name), "w") as f:
                json.dump(retrieval_results, f, indent=4)
    print('store retrieval results...')

    if accelerator.is_main_process:
        if args.output_dir is not None:
            output_name = f"results__{args.dataset_name}__{args.dataset_config_name}__{args.prompt_mode}.json"
            if retrieve_num != 0 and concat_num != 0:
               output_name =  f"results__{args.dataset_name}__{args.dataset_config_name}__{args.prompt_mode}__ret-{retrieve_num}_concat-{concat_num}.json"
            output_name = output_name.replace('/','_')
            with open(os.path.join(args.output_dir, output_name), "w") as f:
                json.dump(all_results, f, indent=4)
    
if __name__ == "__main__":
    main()
