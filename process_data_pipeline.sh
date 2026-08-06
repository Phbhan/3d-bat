export DATA_INPUT_PATH=/home/hanpb2/workspace/Data/Data_PNK/500h/20260711_1512_VF6_03_1783757531_1783759331/
export DATA_PROCESSED_PATH=/home/hanpb2/workspace/Data/DataOD3D/code/3d-bat/input/hanpb2/20260711_1512_VF6_03_1783757531_1783759331
export EXTRINSIC_PATH=/home/hanpb2/workspace/Data/Data_PNK/calib/VF6_03/VF6_03_Extrinsics.json 
export INTRINSIC_PATH=/home/hanpb2/workspace/Data/Data_PNK/calib/VF6_03/VF6_03_Intrinsics.json

# Merge multiple lidar data into unified dcp file
mkdir -p $DATA_PROCESSED_PATH/point_clouds/LIDAR_TOP
python scripts/merge_multi_lidar.py \
    --extr $EXTRINSIC_PATH \
    --lidar_root $DATA_INPUT_PATH/LIDAR \
    --out_dir_pcd $DATA_PROCESSED_PATH/point_clouds/LIDAR_TOP \
    --out_dir_bin $DATA_PROCESSED_PATH/point_clouds/LIDAR_TOP_BIN \
    --out_dir_laz $DATA_PROCESSED_PATH/point_clouds/LIDAR_TOP_LAZ \
    --origin center \
    --visualize \
    --visualize_only \
    --visualize_ts 1783757599-099982977

mkdir -p $DATA_PROCESSED_PATH/point_clouds_lidar/LIDAR_TOP
python scripts/merge_multi_lidar.py \
    --extr $EXTRINSIC_PATH \
    --lidar_root $DATA_INPUT_PATH/LIDAR \
    --out_dir_pcd $DATA_PROCESSED_PATH/point_clouds_lidar/LIDAR_TOP \
    --out_dir_bin $DATA_PROCESSED_PATH/point_clouds_lidar/LIDAR_TOP_BIN \
    --out_dir_laz $DATA_PROCESSED_PATH/point_clouds_lidar/LIDAR_TOP_LAZ \
    --origin back \
    --visualize \
    --visualize_only \
    --visualize_ts 1783757599-099982977

mkdir -p $DATA_PROCESSED_PATH/images_raw
python scripts/select_camera_data.py \
    --lidar_dir $DATA_PROCESSED_PATH/point_clouds/LIDAR_TOP \
    --camera_root $DATA_INPUT_PATH/CAMERA \
    --out_dir $DATA_PROCESSED_PATH/images_raw

mkdir -p $DATA_PROCESSED_PATH/images_pinhole
python scripts/undistort_images.py \
  --images_root $DATA_PROCESSED_PATH/images_raw \
  --out_root $DATA_PROCESSED_PATH/images_pinhole \
  --intr_path $INTRINSIC_PATH \
  --extr_path $EXTRINSIC_PATH \
  --alpha 0

python scripts/build_calib_json.py \
    --extr_path $EXTRINSIC_PATH \
    --intr_path $DATA_PROCESSED_PATH/images_pinhole/new_intrinsics.json \
    --input_pred_dir "/lustre/scratch/client/vinfast/groups/l4/hanpb2/bevfusion/input_data/20260711_1512_VF6_03_1783757531_1783759331" \
    --raw_root  $DATA_INPUT_PATH \
    --primary_lidar LIDAR_TOP \
    --no_camera_offset \
    --global_coord_mode utm \
    --out_path  $DATA_PROCESSED_PATH/input_data.json


rsync -avP 'input/hanpb2/20260711_1512_VF6_03_1783757531_1783759331/point_clouds_lidar' \
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
# python scripts/create_empty_annotations.py \
#     --input_folder_path_point_clouds $DATA_PROCESSED_PATH/point_clouds/LIDAR_TOP \
#     --output_folder_path_annotations $DATA_PROCESSED_PATH/annotations

python scripts/preprocess_anns.py \
        --annotations_dir $DATA_PROCESSED_PATH/annotations \
        --rename \
        --modify-box \
        --x-offset -0.493 

python scripts/create_file_name_list.py \
    --input_folder_path_drive $DATA_PROCESSED_PATH
