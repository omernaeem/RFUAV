import os
import shutil

def main():
    input = 'E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/benchmark/'
    output = 'E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/benchmark_CM/'

    drones = os.listdir(input)
    snrs = os.listdir(os.path.join(input, drones[0]))
    CMs = os.listdir(os.path.join(input, drones[0], snrs[0]))
    stft_p = '1024'

    for drone in drones:
        drone_path = os.path.join(input, drone)
        for snr in snrs:
            snr_path = os.path.join(drone_path, snr)
            for CM in CMs:
                CM_path = os.path.join(snr_path, CM)
                files = os.listdir(os.path.join(CM_path, stft_p))
                for file in files:
                    save_path = os.path.join(output, snr, CM, drone)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    shutil.copy(os.path.join(CM_path, stft_p, file), save_path)
                    print(file + ' Done')
                print(CM + ' Done')
            print(snr + ' Done')
        print(drone + ' Done')


if __name__ == '__main__':
    main()