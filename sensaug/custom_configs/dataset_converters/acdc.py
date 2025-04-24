# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil

from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ACDC dataset to mmsegmentation format')
    parser.add_argument('raw_data', help='the path of raw data')
    parser.add_argument(
        '-o', '--out_dir', help='output path', default='./data/acdc')
    parser.add_argument(
        '--split',
        choices=['fog', 'night', 'rain', 'snow', 'all'],
        default='night')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print('Making directories...')

    suffix = ""

    if args.split != "all":
        suffix = f"_{args.split}"
        
    mkdir_or_exist(args.out_dir)
    
    for subdir in [f'gt{suffix}/test', f'rgb_anno{suffix}/test', f'gt{suffix}/train', f'rgb_anno{suffix}/train']:
        mkdir_or_exist(osp.join(args.out_dir, subdir))

    print('Moving images and annotations...')


    if args.split == 'all':
        anno_str_test = '/val/'
        gt_str_test = '/val/'
        anno_str_train = '/train/'
        gt_str_train = '/train/'
    else:
        anno_str_test = f'rgb_anon/{args.split}/val/'
        gt_str_test = f'gt/{args.split}/val/'
        anno_str_train = f'rgb_anon/{args.split}/train/'
        gt_str_train = f'gt/{args.split}/train/'

    for dirpath, _, filenames in os.walk(args.raw_data):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if "rgb_anno" in full_path:
                continue

            if anno_str_test in full_path:
                if full_path.endswith('_rgb_anon.png'):
                    new_path = os.path.join(args.out_dir, f'rgb_anno{suffix}/test',
                                            filename)
                    if full_path != new_path:
                        shutil.copy(full_path, new_path)
            if gt_str_test in full_path:
                if '_gt_labelTrainIds.png' in full_path:
                    new_path = os.path.join(args.out_dir, f'gt{suffix}/test', filename)
                    if full_path != new_path:
                        print(f"{full_path} ==> {new_path}")
                        shutil.copy(full_path, new_path)

            if anno_str_train in full_path:
                if full_path.endswith('_rgb_anon.png'):
                    new_path = os.path.join(args.out_dir, f'rgb_anno{suffix}/train',
                                            filename)
                    if full_path != new_path:
                        shutil.copy(full_path, new_path)
            if gt_str_train in full_path:
                if full_path.endswith('_gt_labelTrainIds.png'):
                    new_path = os.path.join(args.out_dir, f'gt{suffix}/train', filename)
                    if full_path != new_path:
                        print(f"{full_path} ==> {new_path}")
                        shutil.copy(full_path, new_path)

    print('Done!')


if __name__ == '__main__':
    main()