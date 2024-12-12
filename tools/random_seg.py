"""random crop the dataset into train set and valid set
"""
import os
import random
import shutil


def split_images(source_dir, target_dir, other_dir, open_dir, close_dir, ratio, special_dir):

    drone_type = [_ for _ in os.listdir(source_dir) if _ not in other_dir]

    for drone in drone_type:
        if drone in special_dir:
            for detail_type in os.listdir(source_dir+drone+'/'+target_dir+'/'):
                pics = [_ for _ in os.listdir(source_dir+drone+'/'+target_dir+'/'+detail_type+'/')]
                os.makedirs(open_dir+detail_type, exist_ok=True)
                os.makedirs(close_dir+detail_type, exist_ok=True)
                random.shuffle(pics)
                split_point = int(len(pics) * ratio)
                open_files = pics[:split_point]
                close_files = pics[split_point:]

                for file in open_files:
                    shutil.copy(os.path.join(source_dir+drone+'/'+target_dir+'/'+detail_type, file), os.path.join(open_dir+detail_type, file))

                for file in close_files:
                    shutil.copy(os.path.join(source_dir+drone+'/'+target_dir+'/'+detail_type, file), os.path.join(close_dir+detail_type, file))

        else:
            pics = [_ for _ in os.listdir(source_dir+drone+'/'+target_dir+'/')]
            os.makedirs(open_dir+drone, exist_ok=True)
            os.makedirs(close_dir+drone, exist_ok=True)
            random.shuffle(pics)
            split_point = int(len(pics) * ratio)
            open_files = pics[:split_point]
            close_files = pics[split_point:]

            for file in open_files:
                shutil.copy(os.path.join(source_dir+drone+'/'+target_dir, file), os.path.join(open_dir+drone, file))

            for file in close_files:
                shutil.copy(os.path.join(source_dir+drone+'/'+target_dir, file), os.path.join(close_dir+drone, file))


def main():
    source_directory = ''
    target_dir = 'usable'
    other_dir = ['1.dataset']
    ratio = 0.7
    open_dir = ''
    close_dir = ''

    special_dir = ['FLYSKY', 'FRSKY', 'FUTABA',
                   'JR PROPO', 'JUMPER', 'Radiamaster',
                   'Radiolink', 'SIYI', 'SKYDROID']

    split_images(source_directory, target_dir, other_dir, open_dir, close_dir, ratio, special_dir)


if __name__ == '__main__':
    main()