# A sample script to test the model.
from utils.benchmark import Classify_Model


def main():
    source = 'E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/bechmark_test/batch9/'

    test = Classify_Model(cfg='E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/3.dataset-origin+20dB_defaultcolor_usingAG/exp/4.ResNet101/config.yaml',
                          weight_path='E:/Drone_dataset/RFUAV/augmentation_exp1_MethodSelect/3.dataset-origin+20dB_defaultcolor_usingAG/exp/4.ResNet101/best_model.pth')
    # test.inference(source=source, save_path='./res/')
    test.benchmark(data_path=source)


if __name__ == '__main__':
    main()