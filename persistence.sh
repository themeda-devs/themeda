BATCH=2
THEMEDA_DATA_DIR=/mnt/ecosense/data_wip/

poetry run themeda validate \
    --input land_cover --input rain --input tmax --input elevation --input land_use \
    --input fire_scar_early  --input fire_scar_late \
    --input soil_ece --input soil_clay --input soil_depth \
    --validation-subset 1 --batch-size $BATCH \
    --persistence \
    --base-dir $THEMEDA_DATA_DIR \
    --max-chiplets 1000
