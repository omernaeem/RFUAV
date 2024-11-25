# A sample script to test the model.
from utils.benchmark import Model


def main():

    test = Model(cfg='E:/Train_log/RFUAV/exp1_test/config.yaml',
                 weight_path='E:/Train_log/RFUAV/exp1_test/ResNet_epoch_29.pth')

    test.inference(source='E:/FowardRes_log/RFUAV/1.code_check/exp1/source',
                   save_path='E:/FowardRes_log/RFUAV/1.code_check/exp1/res')
    # test.benchmark()


if __name__ == '__main__':
    main()