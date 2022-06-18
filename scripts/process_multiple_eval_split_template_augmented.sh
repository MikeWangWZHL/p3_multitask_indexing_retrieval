### use retrieved
python evaluation/get_retrieved_dataset_from_disk_based_on_template.py \
    --dataset_name openbookqa \
    --dataset_config_name main
echo "done openbookqa_main"

python evaluation/get_retrieved_dataset_from_disk_based_on_template.py \
    --dataset_name piqa \
    --dataset_config_name None
echo "done piqa"


# ### use original 
# python evaluation/get_retrieved_dataset_from_disk_based_on_template.py \
#     --dataset_name openbookqa \
#     --dataset_config_name main \
#     --use_original_eval_template
# echo "done openbookqa_main"

# python evaluation/get_retrieved_dataset_from_disk_based_on_template.py \
#     --dataset_name piqa \
#     --dataset_config_name None \
#     --use_original_eval_template
# echo "done piqa"