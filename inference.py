# A sample script to test the model.
from utils.benchmark import Classify_Model


def main():
    source = 'E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/bechmark_test/batch6/'
    test = Classify_Model(cfg='E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/4.dataset-20dB_hsv_usingAG/exp/10.ViT_L_16/config.yaml',
                          weight_path='E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/4.dataset-20dB_hsv_usingAG/exp/10.ViT_L_16/best_model.pth')
    # test.inference(source=source, save_path='./res/')
    test.benchmark(data_path=source)


if __name__ == '__main__':
    main()