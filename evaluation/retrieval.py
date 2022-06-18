from datasets import DatasetDict, load_from_disk, concatenate_datasets, load_dataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
import time
from tqdm import tqdm
import os
# from multiprocess import set_start_method
from glob import glob
import json



def set_up_device(gpu_index):
    # single gpu
    if torch.cuda.is_available():
        dev = f"cuda:{gpu_index}"
    else:
        dev = "cpu"
    return gpu_index, torch.device(dev)
    # return gpu_index, torch.device("cpu")

def load_multitask_datasets(dataset_disk_paths):
    print("loading picked datasets x prompt_template number:", len(dataset_disk_paths))
    datasets = []
    print('loading datasets train split ...')
    for p in tqdm(dataset_disk_paths):
        loaded_dataset = load_from_disk(p)
        if "train" in loaded_dataset:
            loaded_dataset_train_split = loaded_dataset['train']
        else:
            print(f'INFO: no train split found, using entire dataset: {p}')
            loaded_dataset_train_split = loaded_dataset

        datasets.append(loaded_dataset_train_split)
    concatenated = concatenate_datasets(datasets)
    print("concatenated dataset:", concatenated)
    return concatenated

@torch.no_grad()
def setup_retriever(saved_index_dir, device):
    # load dataset and index
    index_path = os.path.join(saved_index_dir,'index.faiss')
    config_json = os.path.join(saved_index_dir,'config.json')
    
    config = json.load(open(config_json))
    key_name, dataset_paths = config['key_name'], config['dataset_paths']
    ds_with_embeddings = load_multitask_datasets(dataset_paths)
    # ds_with_embeddings.load_faiss_index('embeddings', index_path)
    print('loading faiss index on gpu...')
    ds_with_embeddings.load_faiss_index('embeddings', index_path, device=-1) # using gpu

    q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_encoder.to(device)
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    return ds_with_embeddings, q_encoder, q_tokenizer

@torch.no_grad()
def setup_retriever_shard(shard_name, saved_dataset_root, saved_index_root, device, use_faiss_gpu = False):

    shard_path = os.path.join(saved_dataset_root, shard_name)
    index_path = os.path.join(saved_index_root, shard_name, "index.faiss")

    # if not os.path.exists(shard_path):
    #     config_json = os.path.join(saved_index_root, shard_name, 'config.json')
    #     config = json.load(open(config_json))
    #     key_name, dataset_paths = config['key_name'], config['dataset_paths']
    #     concatenated = load_multitask_datasets(dataset_paths)
    #     concatenated.save_to_disk(shard_path)

    print("loading:",shard_path)
    reloaded = load_from_disk(shard_path)

    if isinstance(reloaded, DatasetDict):
        reloaded = reloaded["train"]
        print(len(reloaded))

    print('loading index_path:',index_path)
    if use_faiss_gpu:
        reloaded.load_faiss_index('embeddings', index_path, device=-1)
    else:
        reloaded.load_faiss_index('embeddings', index_path)

    print(reloaded)

    q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_encoder.to(device)
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    return reloaded, q_encoder, q_tokenizer

@torch.no_grad()
def retrieve(
        ds_with_embeddings,
        q_encoder,
        q_tokenizer,
        question,
        device,
        topk=3
    ):
    if isinstance(ds_with_embeddings, list):
        question_embedding = q_encoder(**q_tokenizer(question, return_tensors="pt", truncation=True).to(device))[0][0].cpu().numpy()
        cands = []
        for ds in ds_with_embeddings:
            subset_scores, subset_retrieved_examples = ds.get_nearest_examples('embeddings', question_embedding, k=topk)
            for i in range(len(subset_scores)):
                cands.append((subset_retrieved_examples[i],subset_scores[i]))
        cands = sorted(cands, key = lambda x: x[1])
        scores = []
        retrieved_examples = []
        for item in cands:
            retrieved_examples.append(item[0])
            scores.append(item[1])
    else:
        question_embedding = q_encoder(**q_tokenizer(question, return_tensors="pt", truncation=True).to(device))[0][0].cpu().numpy()
        scores, retrieved_examples = ds_with_embeddings.get_nearest_examples('embeddings', question_embedding, k=topk)

    return scores, retrieved_examples

import copy
def retrieval_for_disk_dataset(
    loaded_dataset,
    shard_names, # a list of retrieval database name
    output_path,
    saved_dataset_root = "/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/indexed_datasets",
    saved_index_root = "/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/output_indexing",
    device = torch.device("cuda:7"), # encoder device
    k = 20, # retrieval number
    max_length = 1024
    ):

    if os.path.exists(output_path):
        print('INFO: skipping already exists dataset:', output_path)
        return

    print("working on:", loaded_dataset)

    # loop over retrieval databases
    processed_dataset = loaded_dataset
    for shard_name in shard_names:
        print('INFO: retrieving from:', shard_name)
        loaded_retrieval_dataset, q_encoder, q_tokenizer = setup_retriever_shard(shard_name, saved_dataset_root, saved_index_root, device, use_faiss_gpu=True)    

        ### preprocess dataset functions ###
        @torch.no_grad()
        def preprocess_function(examples):
            bs = len(examples["inputs_pretokenized"])
            print('batch processing size:', bs)
            scores_batch = []
            retrieved_examples_batch = []
            for i in range(bs):
                input = examples["inputs_pretokenized"][i]
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
    print(processed_dataset)
    print('save to disk:', output_path)
    processed_dataset.save_to_disk(output_path)

    del loaded_retrieval_dataset, processed_dataset, q_encoder, q_tokenizer


@torch.no_grad()
def retrieve_template_str(candidate_tempalte_dict, query_str, output_dir, device_index=7, num_proc=1, top_k=1):
    gpu_index, device = set_up_device(device_index)

    saving_intermediate_jsonl = os.path.join(output_dir, "tmp.jsonl")
    saving_index_path = os.path.join(output_dir, "index.faiss")
    saving_config_json_path = os.path.join(output_dir, "config.json")
    # load dataset

    key_name = "template_str"
    lines = [value for key, value in candidate_tempalte_dict.items()]
    with open(saving_intermediate_jsonl, 'w') as out:
        for line in lines:
            out.write(json.dumps(line))
            out.write('\n')

    ds = load_dataset('json', data_files=saving_intermediate_jsonl)
    print(ds)
    ds = ds['train']


    if not os.path.exists(saving_index_path):
        ### load model
        ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        ctx_encoder.to(device)

        def map_function(example):
            tokenized = ctx_tokenizer(example[key_name], return_tensors="pt", truncation=True)
            tokenized.to(device)
            output = {'embeddings': ctx_encoder(**tokenized)[0][0].cpu().numpy()}
            return output
        ds_with_embeddings = ds.map(map_function, num_proc=num_proc)

        print('adding index...')
        # ds_with_embeddings.add_faiss_index(column='embeddings', device=-1)
        ds_with_embeddings.add_faiss_index(column='embeddings')


        print('saving faiss index to ', saving_index_path)
        ds_with_embeddings.save_faiss_index('embeddings', saving_index_path)

        with open(saving_config_json_path, 'w') as out:
            print('saving config to ', saving_config_json_path)
            config = {
                "templates":candidate_tempalte_dict,
                "key_name":key_name
            }
            json.dump(config, out, indent=4)
    else:
        ds_with_embeddings = ds
        ds.load_faiss_index('embeddings', saving_index_path)

    q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_encoder.to(device)
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    question_embedding = q_encoder(**q_tokenizer(query_str, return_tensors="pt", truncation=True).to(device))[0][0].cpu().numpy()
    scores, retrieved_examples = ds_with_embeddings.get_nearest_examples('embeddings', question_embedding, k=top_k)

    return scores, retrieved_examples


if __name__ == "__main__":
    candidate_tempalte_dict = {
        "A":"aaa",
        "B":"bbb",
        "C":"ccc",
    }
    scores, retrieved_examples = retrieve_template_str(candidate_tempalte_dict, "Aaa", output_dir="./tmp")
    print(scores, retrieved_examples)