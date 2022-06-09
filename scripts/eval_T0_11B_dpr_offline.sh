export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
MODEL_NAME="bigscience/T0"
DATASET_NAME="super_glue"
DATASET_CONFIG_NAME="boolq"
TEMPLAETE_NAME="after_reading"
PROMPT_MODE="ic_dpr_full_prompt"
# SHARD_NAME="p3_subset_6_3-part-2.p3_subset_6_1"
SHARD_NAME="p3_subset_6_6_multichoice_qa_new"
OUTPUT_DIR="./output/T0_11B__${DATASET_NAME}__${DATASET_CONFIG_NAME}__${SHARD_NAME}_offline"

### eval all templates ###
accelerate launch evaluation/run_eval_retrieve_offline.py \
    --dataset_name $DATASET_NAME \
    --dataset_config_name $DATASET_CONFIG_NAME \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --parallelize \
    --prompt_mode $PROMPT_MODE \
    --shard_name $SHARD_NAME \
    --template_name $TEMPLAETE_NAME \
    # --eval_all_templates \

# ### eval all templates ###
# python evaluation/run_eval_retrieve_offline.py \
#     --dataset_name $DATASET_NAME \
#     --dataset_config_name $DATASET_CONFIG_NAME \
#     --model_name_or_path $MODEL_NAME \
#     --output_dir $OUTPUT_DIR \
#     --parallelize \
#     --prompt_mode $PROMPT_MODE \
#     --shard_name $SHARD_NAME \
#     --template_name $TEMPLAETE_NAME \
#     # --eval_all_templates \