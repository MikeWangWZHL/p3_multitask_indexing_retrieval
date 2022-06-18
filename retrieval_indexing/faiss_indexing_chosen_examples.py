from datasets import load_from_disk, concatenate_datasets, load_dataset
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
    print("picked datasets x prompt_template number:", len(dataset_disk_paths))
    datasets = []
    print('loading datasets train split ...')
    for p in tqdm(dataset_disk_paths):
        loaded_dataset = load_from_disk(p)
        if "train" in loaded_dataset:
            loaded_dataset_train_split = loaded_dataset['train']
        else:
            print(f'INFO: no train split found, using entire dataset: {p}')
            loaded_dataset_train_split = loaded_dataset

        # ### add dataset name ###
        # def add_dataset_name_column(example):
        #     example['dataset_name'] = os.path.basename(p)
        #     return example
        # loaded_dataset_train_split = loaded_dataset_train_split.map(add_dataset_name_column)

        datasets.append(loaded_dataset_train_split)
    concatenated = concatenate_datasets(datasets)
    print("concatenated dataset:", concatenated)
    return concatenated

def load_multitask_datasets_with_idx(picked_datasets_names, dataset_disk_root, task_2_templates = None):
    datasets = []
    dataset_disk_paths = []
    print('loading datasets train split ...')
    for dataset_name in tqdm(picked_datasets_names):
        for p in glob(os.path.join(dataset_disk_root,f"{dataset_name}*")):
            
            if task_2_templates is not None:
                assert dataset_name in task_2_templates
                if 'original_dataset_name' not in task_2_templates[dataset_name]:
                    print('!!! skip non-original:', dataset_name, os.path.basename(p))
                    continue
                if os.path.basename(p) not in task_2_templates[dataset_name]['original_dataset_name']:
                    print('!!! skip non-original:', dataset_name, os.path.basename(p))
                    continue
            
            dataset_disk_paths.append(p)
            loaded_dataset = load_from_disk(p)
            # template_name = os.path.basename(p).replace(dataset_name,'')
            template_name = os.path.basename(p).removeprefix(dataset_name)
            if "train" in loaded_dataset:
                loaded_dataset_train_split = loaded_dataset['train']
            else:
                print(f'INFO: no train split found, using entire dataset: {p}')
                loaded_dataset_train_split = loaded_dataset
            
            ### add dataset name ###
            def add_instance_info_column(example, idx):
                example['dataset_name'] = dataset_name
                example['template_name'] = template_name
                example['idx'] = idx
                return example
            loaded_dataset_train_split = loaded_dataset_train_split.map(add_instance_info_column, with_indices=True)
            datasets.append(loaded_dataset_train_split)
            # break
        # break
    concatenated = concatenate_datasets(datasets)
    print("concatenated dataset:", concatenated)
    return concatenated, dataset_disk_paths


def load_chosen_examples_as_retrieval_dataset(
    picked_dataset_dir_names,
    dataset_dir_root,
    n,
    output_dataset_dir,
    output_indexing_dir,
    key_name
    ):

    if os.path.exists(output_dataset_dir):
        print("NOTE: reuse chosen example dataset:",output_dataset_dir)
        ds_loaded = load_from_disk(output_dataset_dir)
        return ds_loaded

    lines = []
    config_paths = []
    is_added = set()
    for dataset_dir_name in picked_dataset_dir_names:
        for p in sorted(glob(os.path.join(dataset_dir_root, dataset_dir_name, "*")))[:n]:
            print('working on:',p)
            ds = load_from_disk(p)
            if 'train' in ds:
                ds = ds["train"]
            for item in tqdm(ds):
                for e in item['chosen_examples']:
                    if (e['dataset_name'],e['template_name'],e['idx']) not in is_added:
                        is_added.add((e['dataset_name'],e['template_name'],e['idx']))
                        lines.append(e)
            config_paths.append(p)

    # store jsonl
    with open(os.path.join(output_indexing_dir,'chosen_examples.jsonl'), 'w') as out:
        for line in lines:
            out.write(json.dumps(line))
            out.write("\n")

    # store config
    with open(os.path.join(output_indexing_dir,'config.json'), 'w') as out:
        config_ = {
            "key_name":key_name,
            "dataset_paths":config_paths,
            "indexed_dataset_path":output_dataset_dir
        }
        json.dump(config_, out, indent=4)

    # load from jsonl, and save
    ds_loaded = load_dataset('json', data_files = os.path.join(output_indexing_dir,'chosen_examples.jsonl'))
    ds_loaded.save_to_disk(output_dataset_dir)
    print("save chosen example hf dataset to:", output_dataset_dir)
    return ds_loaded



@torch.no_grad()
def indexing(
        picked_dataset_dir_names,
        dataset_dir_root,
        output_name,
        n, # how many tempaltes to use for each dataset
        key_name = "inputs_pretokenized",
        output_dataset_root = "/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/indexed_datasets",
        output_indexing_root = "/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/output_indexing",
        if_batched = False,
        num_proc = 1,
        device_index = 1,
        task_2_templates = json.load(open("/cephfs/user/mikeeewang/summer_22/workspace/data/bigscience_P3_task_2_templates.json"))
        # dataset_disk_root = "/cephfs/user/jianyiyang/workspace/data/bigscience_P3"
    ):

    output_dataset_dir = os.path.join(output_dataset_root, output_name)
    output_indexing_dir = os.path.join(output_indexing_root, output_name)
    os.makedirs(output_indexing_dir, exist_ok=True)

    saving_index_path = os.path.join(output_indexing_dir, "index.faiss")
    if os.path.exists(saving_index_path):
        print('NOTE: faiss index already exist:',saving_index_path)
        return


    ### load dataset
    print('picked dataset dirs:', picked_dataset_dir_names)

    ds = load_chosen_examples_as_retrieval_dataset(
        picked_dataset_dir_names,
        dataset_dir_root,
        n,
        output_dataset_dir,
        output_indexing_dir,
        key_name
    )
    ds = ds['train']
    print(ds)

    ### set up device
    gpu_index, device = set_up_device(device_index)

    ### load model
    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    ctx_encoder.to(device)

    ### generate embeddings
    start = time.time()
    print('start generating bert embeddings and indexing...')
    print('note: truncation at 512 tokens')

    #####################

    ### not batched
    if not if_batched:
        def map_function(example):
            tokenized = ctx_tokenizer(example[key_name], return_tensors="pt", truncation=True)
            tokenized.to(device)
            output = {'embeddings': ctx_encoder(**tokenized)[0][0].cpu().numpy()}
            return output
        ds_with_embeddings = ds.map(map_function, num_proc=num_proc)
    else:
        ## batched
        def map_function_batched(batch):
            tokenized = ctx_tokenizer(batch[key_name], return_tensors="pt", truncation=True, padding=True)
            tokenized.to(device)
            output = {'embeddings': ctx_encoder(**tokenized)[0].cpu().numpy()}
            return output
        ds_with_embeddings = ds.map(map_function_batched, batched = True, batch_size = 512)

    # ### not batched multi-gpu # Not working
    # def map_function_multigpu(example, rank):
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())
    #     tokenized = ctx_tokenizer(example["text"], return_tensors="pt", truncation=True)
    #     tokenized.to(device)
    #     output = {'embeddings': ctx_encoder(**tokenized)[0][0].cpu().numpy()}
    #     return output
    # ds_with_embeddings = ds.map(map_function_multigpu, with_rank=True)

    #####################

    print('adding index...')
    # ds_with_embeddings.add_faiss_index(column='embeddings', device=-1)
    ds_with_embeddings.add_faiss_index(column='embeddings')

    ds_with_embeddings.save_faiss_index('embeddings', saving_index_path)
    print('saving faiss index to ', saving_index_path)

    end = time.time()
    print('time passed:', end-start)

@torch.no_grad()
def search(
        question,
        saved_index_dir,
        topk=3,
        device_index = 1
    ):
    ### set up device
    gpu_index, device = set_up_device(device_index)

    # load dataset and index
    index_path = os.path.join(saved_index_dir,'index.faiss')
    config_json = os.path.join(saved_index_dir,'config.json')

    config = json.load(open(config_json))
    key_name, indexed_dataset_path = config['key_name'], config['indexed_dataset_path']
    ds_with_embeddings = load_from_disk(indexed_dataset_path)
    ds_with_embeddings = ds_with_embeddings['train']
    ds_with_embeddings.load_faiss_index('embeddings', index_path)
    # ds_with_embeddings.load_faiss_index('embeddings', index_path, device=-1)

    q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_encoder.to(device)
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    question_embedding = q_encoder(**q_tokenizer(question, return_tensors="pt", truncation=True).to(device))[0][0].cpu().numpy()
    scores, retrieved_examples = ds_with_embeddings.get_nearest_examples('embeddings', question_embedding, k=topk)
    print("\n\ntop 1 text:",retrieved_examples[key_name][0])
    return scores, retrieved_examples



if __name__ == "__main__":
    # ### do indexing ###
    dataset_dir_root = '/cephfs/user/mikeeewang/summer_22/workspace/data/p3_finetuning/with_retrieval_add-special-token_false__store_chosen_example'
    picked_dataset_dir_names = [
        "cos_e__v1.11__p3_subset__k-1",
        "cosmos_qa__none__p3_subset__k-1",
        "dream__none__p3_subset__k-1",
        "qasc__none__p3_subset__k-1",
        "quartz__none__p3_subset__k-1",
        "sciq__none__p3_subset__k-1",
        "social_i_qa__none__p3_subset__k-1",
        "wiqa__none__p3_subset__k-1"
    ]
    n = 2
    output_name = f"mulcqa_mixture_k-1_6-10_n-{n}"
    # output_name = f"tmp"

    ### if not None, use original tasks only:
    task_2_templates = None
    # task_2_templates = json.load(open("/cephfs/user/mikeeewang/summer_22/workspace/data/bigscience_P3_task_2_templates.json"))

    device_index = 7
    key_name = "inputs_pretokenized"

    indexing(
        picked_dataset_dir_names,
        dataset_dir_root,
        output_name,
        n,
        key_name = "inputs_pretokenized",
        output_dataset_root = "/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/indexed_datasets",
        output_indexing_root = "/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/output_indexing",
        if_batched = False,
        num_proc = 1,
        device_index = 1,
        task_2_templates = task_2_templates
    )

    ### do search ###
    search(
        question="happy birthday",
        saved_index_dir=os.path.join("/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/output_indexing", output_name)
    )
