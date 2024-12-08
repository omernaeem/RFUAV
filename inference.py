# A sample script to test the model.
from utils.benchmark import Classify_Model


def main():

    test = Classify_Model(cfg='./example/classify/ResNet101-stage2.yaml',
                          weight_path='./example/ckpt/stage2.pth')

    test.inference(source='E:/Drone_dataset/RFUAV/exp3/train/FLYSKY_FS_I6X/',
                   save_path='./res/')
    # test.benchmark()


if __name__ == '__main__':
    main()