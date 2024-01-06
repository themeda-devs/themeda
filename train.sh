BATCH=1
LEARNING_RATE=0.001
KERNEL=15
HIDDEN=64
EMBEDDING=16

TEMPORAL=TRANSFORMER

RUN_NAME=themeda-outLC-b$BATCH-max4k-tf-PE

poetry run themeda train \
    --input land_cover \
    --input rain --input tmax --input elevation --input land_use \
    --input soil_ece --input soil_clay --input soil_depth \
    --input fire_scar_early  --input fire_scar_late \
    --output land_cover \
    --validation-subset 1 --batch-size $BATCH \
    --learning-rate $LEARNING_RATE --temporal-processor-type $TEMPORAL \
    --kernel-size $KERNEL --embedding-size $EMBEDDING \
    --base-dir $THEMEDA_DATA_DIR \
    --run-name $RUN_NAME --output-dir outputs/$RUN_NAME \
    --no-emd-loss --no-hierarchical-embedding \
    --max-chiplets 4000 \
    --wandb --wandb-entity punim1932



    # --input rain --input tmax --input elevation --input land_use \
    # --input soil_ece --input soil_clay --input soil_depth \
    # --input fire_scar_early  --input fire_scar_late \
