
import argparse
import logging
import os
import random
import json

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

import sys
ROOT_DIR = "/cephfs/user/mikeeewang/summer_22/code/t-zero"
T0_DIR = os.path.join(ROOT_DIR,'t0')
sys.path.insert(1, T0_DIR)
from data_collator import DataCollatorForMultipleChoice
from model import ModelBase, ModelBase_with_confidence


from template_list import template_list
from retrieval import setup_retriever, retrieve, setup_retriever_shard

logger = logging.getLogger(__name__)

from collections import defaultdict