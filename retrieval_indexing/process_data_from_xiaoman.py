import os
import sys
import argparse
import logging
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import random
from collections import defaultdict
import copy

import ujson as json
from datasets import (
    load_dataset,
    load_from_disk
)
from datasets.utils.logging import set_verbosity_error
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    default_data_collator,
    DataCollatorForSeq2Seq,
    AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import PaddingStrategy
import promptsource.templates

"""
get_sequence()你里你只用看 input, target = template.apply(data) 就行了, 后面和你的project是无关的
"""


logger = logging.getLogger(__name__)
# TEMPLATES_FOLDER_PATH='/cephfs/user/xiaomanpan/lib/promptsource/promptsource/templates'
# promptsource.templates.TEMPLATES_FOLDER_PATH=TEMPLATES_FOLDER_PATH


def clean_template_name(s):
    return s.replace('/', '')


def parse_args():
    parser = argparse.ArgumentParser(description="")
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
        "-t",
        "--template_name",
        type=str,
        default=None,
        required=True,
        help="The template/prompt name in `promptsource`.",
    )
    parser.add_argument(
        "-st",
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "-tk",
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "-il",
        "--max_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "-tl",
        "--target_max_length",
        type=int,
        default=256,
        help="Target max length. Sequences longer than this will be truncated."
    )
    parser.add_argument(
        "-pml",
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "-ie",
        "--input_eos",
        action="store_true",
        help=(
            "T0 was trained without EOS in its input sequences, which is the default in this script."
            "However, T5 was pretrained with EOS in its input sequences. See README for more info."
        ),
    )
    parser.add_argument(
        "-sd",
        "--seed",
        type=int,
        default=42,
        help="Especially important for few-shot example sampling.",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default=None,
        required=True,
        help="Input arrow dump directory",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Ourput directory",
    )
    parser.add_argument(
        "-np",
        "--num_proc",
        type=int,
        default=1,
        help="Number of processors for data pre-processing"
    )
    parser.add_argument(
        "-k",
        "--knowledge_types",
        type=str,
        nargs='+',
        default=[],
        help=""
    )
    args = parser.parse_args()

    return args


def get_sequence(data, template, knowledge_types):
    input, target = template.apply(data)
    kic = []
    for knwl_type in data['knowledge']:
        if knwl_type not in knowledge_types:
            continue
        if knwl_type == 'lexicon':
            for key in data['knowledge'][knwl_type]['gloss']:
                if not data['knowledge'][knwl_type]['gloss'][key]:
                    continue
                kic.append(
                    '\n'.join(data['knowledge'][knwl_type]['gloss'][key][0]['glosses'][:5]))
        elif knwl_type == 'causal':
            for key in data['knowledge'][knwl_type]:
                for n in range(5):
                    kic.append(data['knowledge'][knwl_type][key][n]['sent'])
        else:
            for key in data['knowledge'][knwl_type]:
                if not data['knowledge'][knwl_type][key]:
                    continue
                kic.append(
                    '\n'.join(json.loads(data['knowledge'][knwl_type][key][0]['sentence_key'])[:1]))
    input = input + '\n' + '[KiC]\n' + '\n'.join(kic) + '\n[KiC]\n'
    return input, target


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()
    set_seed(args.seed)

    if args.dataset_name == 'anli':
        prompts = promptsource.templates.DatasetTemplates('anli', None)
    else:
        if args.dataset_config_name == 'None':
            args.dataset_config_name = None
        prompts = promptsource.templates.DatasetTemplates(
            f"{args.dataset_name}"
            if args.dataset_config_name is None
            else f"{args.dataset_name}/{args.dataset_config_name}"
        )
    template = prompts[args.template_name]

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    padding = "max_length" if args.pad_to_max_length else False

    os.makedirs(args.output_dir, exist_ok=True)

    def preprocess_train(examples):
        bs = len(examples[column_names[0]])

        input_texts = []
        target_texts = []
        answer_choices = []
        for i in range(bs):
            ex = {
                k: examples[k][i]
                for k in column_names
            }

            input, target = get_sequence(ex, template, args.knowledge_types)

            ex_answer_choices = template.get_answer_choices_list(ex)
            if target:
                assert target in ex_answer_choices
            else:
                target = '<NO LABEL>'
            input_texts.append(input)
            target_texts.append(target)
            answer_choices.append(ex_answer_choices)

        # Log a few random samples:
        for index in random.sample(range(len(input_texts)), 3):
            logger.debug(f'Template name: {args.template_name}')
            logger.debug(f"Sample {index} of the input texts:")
            logger.debug(f"\n{input_texts[index]}\n")

        model_inputs = tokenizer(
            input_texts,
            padding=padding,
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=args.input_eos,
        )
        model_inputs['inputs'] = model_inputs['input_ids']
        model_inputs.pop('input_ids')
        model_inputs.pop('attention_mask')

        with tokenizer.as_target_tokenizer():
            tokenized_targets = tokenizer(
                target_texts,
                padding=padding,
                max_length=args.target_max_length,
                truncation=True,
                add_special_tokens=False,
            )
            model_inputs['targets'] = [
                [(t if t != tokenizer.pad_token_id else -100) for t in targets] + [1]
                for targets in tokenized_targets["input_ids"]
            ]
            model_inputs['inputs_pretokenized'] = input_texts
            model_inputs['targets_pretokenized'] = target_texts
            model_inputs['answer_choices'] = answer_choices
        return model_inputs

    logger.info(f'loading {args.input_dir}')
    raw_dataset = load_from_disk(args.input_dir)

    column_names = raw_dataset['train'].column_names
    processed_dataset = raw_dataset.map(
        preprocess_train,
        batched=True,
        remove_columns=column_names,
        num_proc=args.num_proc
    )

    for index in random.sample(range(len(processed_dataset['train'])), 3):
        logger.debug(
            f"Sample {index} of the training set: {processed_dataset['train'][index]}.")

    template_name = clean_template_name(args.template_name)
    output_dir = f"{args.output_dir}/{'_'.join(args.knowledge_types)}_{template_name.replace(' ', '_')}"
    processed_dataset.save_to_disk(f'{output_dir}')


if __name__ == '__main__':
    main()