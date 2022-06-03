from datasets import load_from_disk, concatenate_datasets, load_dataset
from numpy import pi
import es_client
import time
from glob import glob
import os
from tqdm import tqdm

### config ###
do_indexing = True # if doing indexing for the first time 
key_name = "inputs_pretokenized" # key name in the dataset for indexing


# es_index_name = "6-1"  # name of the index in ElasticSearch
# print('loading datasets ...')
# picked_datasets_names = [
#     "cos_e_v1.11", # multi-choice: commonsense QA
#     "cosmos_qa", # multi-choice: commonsense-based reading comprehension
#     "dream", # multi-choice: dialogue based reading comprehension
#     "qasc", # multi-choice: QA via sentence composition
#     "adversarial_qa_dbert", # extractive QA: reading comprehension
#     "adversarial_qa_dbidaf", # extractive QA: reading comprehension
#     "adversarial_qa_droberta", # extractive QA: reading comprehension
#     "imdb", # sentiment: movie review
#     "rotten_tomatoes", # sentiment: movie review
#     "glue_mrpc", # paraphrase identification
#     "glue_qqp", # paraphrase identification
#     "ag_news", # topic classification
#     "trec", # topic classification
# ]

es_index_name = "test"  # name of the index in ElasticSearch
print('loading datasets ...')
picked_datasets_names = [
    "cos_e_v1.11", # multi-choice: commonsense QA
]



print('picked datasets number:', len(picked_datasets_names))
dataset_disk_root = "/cephfs/user/jianyiyang/workspace/data/bigscience_P3"
dataset_disk_paths = []
for d in picked_datasets_names:
    dataset_disk_paths += glob(os.path.join(dataset_disk_root,f"{d}*"))

print("picked datasets x prompt_template number:", len(dataset_disk_paths))
datasets = []
print('loading datasets...')
for p in tqdm(dataset_disk_paths):
    loaded_dataset_train_split = load_from_disk(p)["train"]
    datasets.append(loaded_dataset_train_split)
concatenated = concatenate_datasets(datasets)
print("concatenated dataset:", concatenated)

quit()

### es config ###
es_config = {
    "settings": {
        "number_of_shards": 16,
        "analysis": {"analyzer": {"stop_standard": {"type": "standard", " stopwords": "_english_"}}},
    },
    "mappings": {
        "properties": {
            "text": {
                "type": "text",
                "analyzer": "standard",
                "similarity": "BM25"
            },
        }
    },
}


### add indexing ###
if do_indexing:
    print(f"adding index {es_index_name} ...")
    concatenated.add_elasticsearch_index(
        key_name, es_client=es_client.client, es_index_config=es_config, es_index_name=es_index_name)
    print("done adding index for:", concatenated.get_index(key_name).es_index_name)

### load index ###
print("loading search index ... ")
concatenated.load_elasticsearch_index(
    key_name, es_client=es_client.client, es_index_name=es_index_name)
print("done loading index")


### query ###
query = "let's think step by step"
print('query:',query,'\n')
scores, retrieved_examples = concatenated.get_nearest_examples(key_name, query, k=5)
for i, j in zip(scores, retrieved_examples['text']):
    print(i)
    print(j)
    print()