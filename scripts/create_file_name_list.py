from argparse import ArgumentParser
import os
from utils import *


def create_files(input_folder_path_drive):
    channel_names = os.listdir(os.path.join(input_folder_path_drive, 'images'))
    channel_names = [i for i in channel_names if 'CAM' in i]

    ext = sorted(glob.glob(os.path.join(input_folder_path_drive, 'images', channel_names[0], '*')))[0].split('.')[-1]
    for channel in channel_names:
        txt_file = os.path.join(input_folder_path_drive, f'{channel}_filenames.txt')
        img_filenames = sorted(glob.glob(os.path.join(input_folder_path_drive, 'images', channel, f'*.{ext}')))
        img_filenames = [i.split('/')[-1] for i in img_filenames]
        with open(txt_file, 'w') as img_writer:
            for name in img_filenames:
                img_writer.write(name + '\n')

    lidar_channels = os.listdir(os.path.join(input_folder_path_drive, 'point_clouds'))
    lidar_channels = [i for i in lidar_channels if 'LIDAR' in i]
    print("lidar_channels: ", lidar_channels)
    pcd_txt = os.path.join(input_folder_path_drive, 'point_cloud_filenames.txt')
    if len(lidar_channels) == 0:
        pcd_filenames = sorted(glob.glob(os.path.join(input_folder_path_drive, 'point_clouds', '*.pcd')))
    else:
        pcd_filenames = sorted(glob.glob(os.path.join(input_folder_path_drive, 'point_clouds', "LIDAR_TOP", '*.pcd')))
    pcd_filenames = [i.split('/')[-1] for i in pcd_filenames]
    with open(pcd_txt, 'w') as pcd_writer:
        for name in pcd_filenames:
            pcd_writer.write(name + '\n')

    anno_dir = os.path.join(input_folder_path_drive, 'annotations')
    annotation_file_names = sorted(glob.glob(os.path.join(anno_dir, '*.json')))
    annotation_file_names = [i.split('/')[-1] for i in annotation_file_names]
    annos_txt = os.path.join(input_folder_path_drive, 'annotation_filenames.txt')
    with open(annos_txt, 'w') as annos_writer:
        for annotation_file_name in annotation_file_names:
            annos_writer.write(annotation_file_name + '\n')


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--input_folder_path_drive", type=str, required=True, help="Path to the input folder, e.g. drive_22_north_to_south")
    args = arg_parser.parse_args()
    input_folder_path_drive = args.input_folder_path_drive
    # create file name (.txt files) for drive
    create_files(input_folder_path_drive)


    # create empty annotations for each drive
#     sequences = ['drive_33_north_to_south']
#
#     for seq in sequences:
#         seq_path = os.path.join(input_folder_path_drives, seq)
#         annos_path = os.path.join(seq_path, 'annotations')
#
#         lidar_channels = [i for i in os.listdir(annos_path) if 'lidar' in i]
#         anno_files = sorted(glob.glob(os.path.join(annos_path, lidar_channels[0], '*.json')))
#
#         for anno_file in anno_files:
#             with open(anno_file, 'r') as f:
#                 data = json.load(f)
#
#             frame_idx = list(data['openlabel']['frames'])[0]
#             data['openlabel']['frames'][frame_idx]['objects'] = {}
#
#
#             with open(anno_file, 'w') as f:
#                 json.dump(data, f)
