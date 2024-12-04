# A sample script to test the model.
from utils.benchmark import Classify_Model


def main():

    test = Classify_Model(cfg='E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/3.dataset-origin+20dB_defaultcolor_usingAG/exp/3.ResNet50//config.yaml',
                          weight_path='E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/3.dataset-origin+20dB_defaultcolor_usingAG/exp/3.ResNet50/ResNet_epoch_7.pth')

    test.inference(source='E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/3.dataset-origin+20dB_defaultcolor_usingAG/forTest/input/',
                   save_path='E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/3.dataset-origin+20dB_defaultcolor_usingAG/forTest/res/')
    # test.benchmark()


if __name__ == '__main__':
    main()