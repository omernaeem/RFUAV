import os
import shutil


def main():
    or_path = 'E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/benchmark_or/'
    target_path = 'E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/temp/'
    drone_type = os.listdir(or_path)
    for drone in drone_type:
        snr = os.listdir(os.path.join(or_path, drone))
        for snr_value in snr:
            data_path = os.path.join(or_path, drone, snr_value, 'hot')
            stft_ps = os.listdir(data_path)
            for stft_p in stft_ps:
                imgs = os.listdir(os.path.join(data_path, stft_p))
                for img in imgs:
                    ensure_directory_exists(os.path.join(target_path, snr_value, stft_p, drone))
                    shutil.copy(os.path.join(data_path, stft_p, img), os.path.join(target_path, snr_value, stft_p, drone))


def ensure_directory_exists(directory_path):

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"目录已创建: {directory_path}")
    else:
        print(f"目录已存在: {directory_path}")


if __name__ == '__main__':
    main()