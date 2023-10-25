BATCH=2
LEARNING_RATE=0.001
KERNEL=15
HIDDEN=64
EMBEDDING=16

TEMPORAL=lstm

THEMEDA_DATA_DIR=/mnt/ecosense/data_wip/
RUN_NAME="themeda-demo-attention_resnet"

poetry run themeda train \
    --input land_cover --input rain --input tmax --input elevation --input land_use \
    --input fire_scar_early  --input fire_scar_late \
    --input soil_ece --input soil_clay --input soil_depth \
    --output land_cover \
    --validation-subset 1 --batch-size $BATCH \
    --learning-rate $LEARNING_RATE --temporal-processor-type $TEMPORAL \
    --kernel-size $KERNEL --embedding-size $EMBEDDING \
    --base-dir $THEMEDA_DATA_DIR \
    --run-name $RUN_NAME --output-dir outputs/$RUN_NAME \
    --max-chiplets 1000
    # --wandb --wandb-entity punim1932
