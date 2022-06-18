## Instruction for retrieval and train/eval on a dataset

### Generate indexing
- navigate to `retrieval_indexing/`
- set up the dataset names in `retrieval_indexing/faiss_indexing.py`
- run `python retrieval_indexing/faiss_indexing.py` to generate faiss index, output default dir is `/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/output_indexing`

### Run retrieval and preprocessing from local datasets
script for running retrieval and then process from dataset on disk (e.g., datasets from `/cephfs/user/jianyiyang/workspace/data/bigscience_P3`);
The following script will first generate intermediate datasets (that can be reused) with retrieved examples (default num is 20); then it will process the datasets and into the dataset ready for fine-tuning or evaluation with a specified K (num of demonstrations in front of the examples).

- set up `input_dir`, `intermeidate_dir`, `shard_names`, `output_dir` in `get_retrieved_dataset_from_disk.py`
- an example of adding 1 demonstration: 
```
python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name dream \
    --dataset_config_name None \
    --k 1
echo "done dream"
```
- the resulting dataset will be ready for running MoE finetuning

### [UPDATE Jun 12] Run retrieval and preprocess evaluation tasks only on examples
(1) Run `evaluation/get_retrieved_dataset_from_disk.py` to process training set tasks with augmentation examples attached, for example at `t-zero/` run: `bash scripts/process_multiple_training_split.sh`
(2) Start fine-tuning using scripts such as `/MultitaskGenerativeMoE/training/run_mixture_k-1.sh`
(3) Keep a note on the config of the fine-tuning arguments, especially `DATASET_DIR_NAMES` and `N`
(4) Use the same `DATASET_DIR_NAMES` and `N` and output_name to set up `t-zero/retrieval_indexing/faiss_indexing_chosen_examples.py`, run `faiss_indexing_chosen_examples.py` to get the indexing of the training augmentation examples
(5) Run `evaluation/get_retrieved_dataset_from_disk_for_eval_seen_in_training_only.py` to retrieve and process evaulation tasks, for example at `t-zero/` run: `bash scripts/process_multiple_evaluation_split_seen_in_training_only.sh`



### Run retrieval from Huggingface Dataset
- set up `dataset_name` and `dataset_config_name` and `SHARD_NAMES` (retrieval_index name in `/cephfs/user/mikeeewang/summer_22/code/t-zero/retrieval_indexing/output_indexing`) in `scripts/retrieve_from_hf.sh` then run `bash scripts/retrieve_from_hf.sh`; default output will be at `t-zero/evaluation/retrieved_dataset_train_validation`




<!-- ### If Run Evaluation:
- set up arguments and run one of `eval*.sh` in `scripts`, dataset_name, dataset_config_name and shard_name should be same as the "run retrieval" step;

### If Run Finetuning:
- navigate to `t-zero`
- set up and run `bash scripts/process_multiple.sh`
- the default output root will be `/cephfs/user/mikeeewang/summer_22/workspace/data/p3_finetuning/with_retrieval`
- then set up the arguments in `MultitaskGenerativeMoE/training/run.sh`, especially `INPUT_DATASETS` should be the same as previously outputed one in `/cephfs/user/mikeeewang/summer_22/workspace/data/p3_finetuning/with_retrieval`
- navigate to `MultitaskGenerativeMoE/training` then run `bash run.sh` -->



<!-- 
### set up the valid template names
- navigate to `t-zero/`
- run `evaluation/template_list.py --dataset_name <super_glue> --dataset_config_name <wic>` to get the list of tempalte names; consult [spread_sheet](https://docs.google.com/spreadsheets/d/1iKTeefiznOZ0ZU_gXNcc6EiliIEoY19N6wAVaG3VzGs/edit#gid=25382061) to delete the ones that is NOT origianl task; add the item in the varible `template_list` in `template_list.py` -->