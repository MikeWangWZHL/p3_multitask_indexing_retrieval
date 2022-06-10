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