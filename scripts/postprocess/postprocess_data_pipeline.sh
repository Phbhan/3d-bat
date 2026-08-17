export DATA_PROCESSED_PATH=/home/hanpb2/workspace/Data/DataOD3D/code/3d-bat/input/hanpb2/20260711_1512_VF6_03_1783757531_1783759331
export OUTPUT_PATH=/home/hanpb2/workspace/Data/DataOD3D/code/3d-bat/output/vf6_03_pnk/20260711_1512_VF6_03_1783757531_1783759331

mkdir -p $OUTPUT_PATH

python scripts/postprocess/postprocess_anns.py \
    --input-root $DATA_PROCESSED_PATH \
    --output-root $OUTPUT_PATH \
    --car-name pnkvf603 \
    --lsize Lsize \
    --dummy-dir scripts/postprocess/dummy \
    --require-clearly-visible


rsync -avP '/home/hanpb2/workspace/Data/DataOD3D/code/3d-bat/output/vf6_03_pnk/20260711_1512_VF6_03_1783757531_1783759331' \
            'superpod:/lustre/scratch/client/vinfast/groups/l4/hanpb2/dataset2/vf6_03_pnk/labeled_vf6_pnk_20260711'
            
