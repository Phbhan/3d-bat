# Merge multiple lidar data into unified dcp file
mkdir -p $DATA_PROCESSED_PATH/point_clouds/LIDAR_TOP
python scripts/preprocess/merge_multi_lidar.py \
    --extr $EXTRINSIC_PATH \
    --lidar_root $DATA_INPUT_PATH/LIDAR \
    --out_dir_pcd $DATA_PROCESSED_PATH/point_clouds/LIDAR_TOP \
    --out_dir_bin $DATA_PROCESSED_PATH/point_clouds/LIDAR_TOP_BIN \
    --out_dir_laz $DATA_PROCESSED_PATH/point_clouds/LIDAR_TOP_LAZ \
    --lidar_origin center \
    --visualize \
    --visualize_only \
    --visualize_ts 1783757599-099982977

mkdir -p $DATA_PROCESSED_PATH/point_clouds_lidar/LIDAR_TOP
python scripts/preprocess/merge_multi_lidar.py \
    --extr $EXTRINSIC_PATH \
    --out_extrinsics $NEW_EXTRINSIC_PATH \
    --lidar_root $DATA_INPUT_PATH/LIDAR \
    --out_dir_pcd $DATA_PROCESSED_PATH/point_clouds_lidar/LIDAR_TOP \
    --out_dir_bin $DATA_PROCESSED_PATH/point_clouds_lidar/LIDAR_TOP_BIN \
    --out_dir_laz $DATA_PROCESSED_PATH/point_clouds_lidar/LIDAR_TOP_LAZ \
    --lidar_origin $LIDAR_ORIGIN \
    --visualize \
    --visualize_only \
    --visualize_ts 1783757599-099982977

mkdir -p $DATA_PROCESSED_PATH/images_raw
python scripts/preprocess/select_camera_data.py \
    --lidar_dir $DATA_PROCESSED_PATH/point_clouds/LIDAR_TOP \
    --camera_root $DATA_INPUT_PATH/CAMERA \
    --out_dir $DATA_PROCESSED_PATH/images_raw

mkdir -p $DATA_PROCESSED_PATH/images_pinhole
python scripts/preprocess/undistort_images.py \
  --images_root $DATA_PROCESSED_PATH/images_raw \
  --out_root $DATA_PROCESSED_PATH/images_pinhole \
  --intr_path $INTRINSIC_PATH \
  --new_intr_path $NEW_INTRINSIC_PATH \
  --alpha 0 --overwrite


python scripts/preprocess/build_calib_json.py \
    --extr_path $NEW_EXTRINSIC_PATH \
    --intr_path $NEW_INTRINSIC_PATH \
    --input_pred_dir "/lustre/scratch/client/vinfast/groups/l4/hanpb2/bevfusion/input_data/20260711_1512_VF6_03_1783757531_1783759331" \
    --raw_root  $DATA_INPUT_PATH \
    --primary_lidar LIDAR_TOP \
    --lidar_origin $LIDAR_ORIGIN \
    --no_camera_offset \
    --global_coord_mode utm \
    --out_path  $DATA_PROCESSED_PATH/input_data.json


rsync -avP '/home/hanpb2/workspace/Data/DataOD3D/code/3d-bat/input/hanpb2/20260711_1512_VF6_03_1783757531_1783759331/point_clouds_lidar/LIDAR_TOP' \
            'superpod:/lustre/scratch/client/vinfast/groups/l4/hanpb2/bevfusion/input_data/20260711_1512_VF6_03_1783757531_1783759331/point_clouds'

rsync -avP '/home/hanpb2/workspace/Data/DataOD3D/code/3d-bat/input/hanpb2/20260711_1512_VF6_03_1783757531_1783759331/images_pinhole' \
            'superpod:/lustre/scratch/client/vinfast/groups/l4/hanpb2/bevfusion/input_data/20260711_1512_VF6_03_1783757531_1783759331/images'

rsync -avP '/home/hanpb2/workspace/Data/DataOD3D/code/3d-bat/input/hanpb2/20260711_1512_VF6_03_1783757531_1783759331/input_data.json' \
            'superpod:/lustre/scratch/client/vinfast/groups/l4/hanpb2/bevfusion/input_data/20260711_1512_VF6_03_1783757531_1783759331'

# infer
rsync -avP 'superpod:/lustre/scratch/client/vinfast/groups/l4/hanpb2/bevfusion/output/20260711_1512_VF6_03_1783757531_1783759331/annotations' \
            '/home/hanpb2/workspace/Data/DataOD3D/code/3d-bat/input/hanpb2/20260711_1512_VF6_03_1783757531_1783759331'

mkdir -p $DATA_PROCESSED_PATH/images
mv $DATA_PROCESSED_PATH/images_raw/CAM_F_F $DATA_PROCESSED_PATH/images/CAM_FRONT 
mv $DATA_PROCESSED_PATH/images_raw/CAM_F_B $DATA_PROCESSED_PATH/images/CAM_BACK 
mv $DATA_PROCESSED_PATH/images_raw/CAM_F_L $DATA_PROCESSED_PATH/images/CAM_FRONT_LEFT 
mv $DATA_PROCESSED_PATH/images_raw/CAM_F_R $DATA_PROCESSED_PATH/images/CAM_FRONT_RIGHT

# mkdir -p $DATA_PROCESSED_PATH/annotations
# python scripts/preprocess/create_empty_annotations.py \
#     --input_folder_path_point_clouds $DATA_PROCESSED_PATH/point_clouds/LIDAR_TOP \
#     --output_folder_path_annotations $DATA_PROCESSED_PATH/annotations

python scripts/preprocess/preprocess_anns.py \
        --annotations_dir $DATA_PROCESSED_PATH/annotations \
        --rename \
        --modify-box \
        --x-offset -1.403

mkdir -p $DATA_PROCESSED_PATH/split
python scripts/preprocess/split_sequences.py \
    --input $DATA_PROCESSED_PATH \
    --output $DATA_PROCESSED_PATH/split \
    --num-data 20

for split_dir in "$DATA_PROCESSED_PATH"/split/*/; do
    echo "Processing: $split_dir"

    python scripts/preprocess/create_file_name_list.py \
        --input_folder_path_drive "$split_dir"
done