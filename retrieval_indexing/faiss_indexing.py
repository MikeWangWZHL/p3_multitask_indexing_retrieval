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
            template_name = os.path.basename(p).lstrip(dataset_name)
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


@torch.no_grad()
def indexing(
        picked_datasets_names,
        output_dir, 
        key_name,
        if_batched = False,
        num_proc = 1,
        dataset_disk_root = "/cephfs/user/mikeeewang/summer_22/workspace/data/p3_subset",
        saved_dataset_root = "/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/indexed_datasets",
        device_index = 1,
        task_2_templates = json.load(open("/cephfs/user/mikeeewang/summer_22/workspace/data/bigscience_P3_task_2_templates.json"))
        # dataset_disk_root = "/cephfs/user/jianyiyang/workspace/data/bigscience_P3"
    ):
    
    ### output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ### set up device
    gpu_index, device = set_up_device(device_index)

    ### load model
    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    ctx_encoder.to(device)

    ### load dataset
    print('picked datasets number:', len(picked_datasets_names))
    # dataset_disk_paths = []
    # for d in picked_datasets_names:
    #     dataset_disk_paths += glob(os.path.join(dataset_disk_root,f"{d}*"))
    # ds = load_multitask_datasets(dataset_disk_paths)

    ds, dataset_disk_paths = load_multitask_datasets_with_idx(picked_datasets_names, dataset_disk_root, task_2_templates) 
    shard_path = os.path.join(saved_dataset_root, os.path.basename(output_dir))
    ds.save_to_disk(shard_path)
    print('saved to disk')

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

    saving_index_path = os.path.join(output_dir, "index.faiss")
    saving_config_json_path = os.path.join(output_dir, "config.json")
    
    print('saving faiss index to ', saving_index_path)
    ds_with_embeddings.save_faiss_index('embeddings', saving_index_path)
    
    with open(saving_config_json_path, 'w') as out:
        print('saving config to ', saving_config_json_path)
        config = {
            "dataset_paths":dataset_disk_paths,
            "key_name":key_name
        }
        json.dump(config, out, indent=4)

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
    key_name, dataset_paths = config['key_name'], config['dataset_paths']
    ds_with_embeddings = load_multitask_datasets(dataset_paths)
    ds_with_embeddings.load_faiss_index('embeddings', index_path)
    # ds_with_embeddings.load_faiss_index('embeddings', index_path, device=-1)

    # for idx, item in enumerate(ds_with_embeddings[:5]['text']):
    #     with open(os.path.join('./dummy_text_folder',f'{idx}.txt'), 'w') as out:
    #         out.write(item)
    #     print(item)
    #     print('\n\n')

    q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    q_encoder.to(device)
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    question_embedding = q_encoder(**q_tokenizer(question, return_tensors="pt", truncation=True).to(device))[0][0].cpu().numpy()
    scores, retrieved_examples = ds_with_embeddings.get_nearest_examples('embeddings', question_embedding, k=topk)
    print("\n\ntop 1 text:",retrieved_examples[key_name][0])
    return scores, retrieved_examples



if __name__ == "__main__":
    # ### do indexing ###

    # output_dir = './output_indexing/subset_6_1'
    # picked_datasets_names = [
    #     "cos_e_v1.11", # multi-choice: commonsense QA
    #     "cosmos_qa", # multi-choice: commonsense-based reading comprehension
    #     "dream", # multi-choice: dialogue based reading comprehension
    #     "qasc", # multi-choice: QA via sentence composition
    #     "adversarial_qa_dbert", # extractive QA: reading comprehension
    #     "adversarial_qa_dbidaf", # extractive QA: reading comprehension
    #     "imdb", # sentiment: movie review
    #     "rotten_tomatoes", # sentiment: movie review
    #     "ag_news", # topic classification
    #     "trec", # topic classification  
    # ]

    # output_dir = './output_indexing/p3_subset_6_3-part-1'
    # picked_datasets_names = [
    #     "glue_mrpc", # paraphrase identification
    #     "glue_qqp", # paraphrase identification
    #     "quail",
    #     "quarel",
    #     "quartz",
    #     "sciq",
    #     "social_i_qa",
    #     "wiki_hop",
    #     "wiki_qa",
    #     "dbpedia_14"
    # ]


    # ## subset 3
    # output_dir = './output_indexing/p3_subset_6_3-part-2'
    # picked_datasets_names = [
    #     "kilt_tasks",
    #     "adversarial_qa",
    #     "duorc",
    #     "ropes",
    #     "quoref",
    #     "tydiqa"
    # ]

    # ## unit test
    # output_dir = './output_indexing/test'
    # picked_datasets_names = [
    #     "trec_trec1", # extractive QA: reading comprehension
    # ]

    ## subset 4
    output_dir = './output_indexing/p3_subset_6_6_multichoice_qa_new_original_only'
    # multi-choice qa training set in t-zero
    picked_datasets_names = [
        "cos_e_v1.11", # multi-choice: commonsense QA
        "cosmos_qa", # multi-choice: commonsense-based reading comprehension
        "dream", # multi-choice: dialogue based reading comprehension
        "qasc", # multi-choice: QA via sentence composition
        "quail", 
        "quarel",
        "quartz", 
        "sciq", 
        "social_i_qa",
        "wiki_hop_original",
        "wiki_qa"
    ]

    ### if not None, use original tasks only:
    task_2_templates = None
    # task_2_templates = json.load(open("/cephfs/user/mikeeewang/summer_22/workspace/data/bigscience_P3_task_2_templates.json"))

    device_index = 7
    key_name = "inputs_pretokenized"
    
    indexing(
        picked_datasets_names,
        output_dir, 
        key_name,
        if_batched = False,
        num_proc = None,
        device_index = device_index,
        task_2_templates = task_2_templates
    )

    ### do search ###
    search(
        question="happy birthday", 
        saved_index_dir = './output_indexing/p3_subset_6_6_multichoice_qa_new',
        device_index = device_index 
    )
