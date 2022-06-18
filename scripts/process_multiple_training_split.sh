K=$1

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name sciq \
    --dataset_config_name None \
    --k $K
echo "done sciq"

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name dream \
    --dataset_config_name None \
    --k $K
echo "done dream"

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name cos_e \
    --dataset_config_name v1.11 \
    --k $K
echo "done cos_e"

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name cosmos_qa \
    --dataset_config_name None \
    --k $K
echo "done cosmos_qa"

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name qasc \
    --dataset_config_name None \
    --k $K
echo "done qasc"

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name quartz \
    --dataset_config_name None \
    --k $K
echo "done quartz"

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name social_i_qa \
    --dataset_config_name None \
    --k $K
echo "done social_i_qa"

python evaluation/get_retrieved_dataset_from_disk.py \
    --dataset_name wiqa \
    --dataset_config_name None \
    --k $K
echo "done wiqa"





# python evaluation/get_retrieved_dataset_from_disk.py \
#     --dataset_name quail \
#     --dataset_config_name None \
#     --k $K
# echo "done quail"