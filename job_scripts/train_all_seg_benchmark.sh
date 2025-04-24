
#!/bin/bash

BACKBONES=('pspnet' 'segformer' 'convnext' 'deeplabv3plus' 'swin' 'mae')
DATASETS=('cityscapes' 'pascal_voc12' 'potsdam' 'loveda' 'ade20k' 'a2i2haze' 'synapse')

# 180 models total (6 * 6 * 5) for segmentation --> 640 GPUs required --> 9 days to run all
AUG_TYPES=('ours' 'none' 'default' 'autoaugment' 'augmix' 'randaugment' 'trivialaugment' 'idbh')

PORT_COUNTER=0

for backbone in ${BACKBONES[*]}; do
    for aug in ${AUG_TYPES[*]} ; do
        for dataset in ${DATASETS[*]}; do
            echo "Starting job for aug: ${aug}, backbone: ${backbone}, dataset: ${dataset}"
            GRES="gpu:rtxa5000:2"
            PORT=$((29800+PORT_COUNTER)) sbatch -J ${aug}_${dataset} --gres ${GRES} --mem 60gb --ntasks 8 job_scripts/train_generic.sh ${aug} ${backbone} ${dataset}
            PORT_COUNTER=$((PORT_COUNTER+10))
        done
    done
done