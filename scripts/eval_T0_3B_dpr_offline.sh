export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
MODEL_NAME="bigscience/T0_3B"
DATASET_NAME="super_glue"
DATASET_CONFIG_NAME="boolq"

PROMPT_MODE="ic_dpr_full_prompt"
# DATASET_NAME="super_glue"
# DATASET_CONFIG_NAME="wic"
# TEMPLAETE_NAME="GPT-3-prompt"
SHARD_NAME="p3_subset_6_6_multichoice_qa"
# SHARD_NAME="p3_subset_6_3-part-2.p3_subset_6_1"
# SHARD_NAME="p3_subset_6_1"
# SHARD_NAME="p3_subset_6_3-part-1"

OUTPUT_DIR="./output/T0_3B__${DATASET_NAME}__${DATASET_CONFIG_NAME}__${SHARD_NAME}_offline"
# OUTPUT_DIR="./output/tmp_offline"

### eval all templates ###
accelerate launch evaluation/run_eval_retrieve_offline.py \
    --dataset_name $DATASET_NAME \
    --dataset_config_name $DATASET_CONFIG_NAME \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --parallelize \
    --eval_all_templates \
    --prompt_mode $PROMPT_MODE \
    --shard_name $SHARD_NAME
    # --template_name $TEMPLAETE_NAME \