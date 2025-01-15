# A sample script to test the model.
from utils.benchmark import Classify_Model


def main():
    source = ''
    test = Classify_Model(cfg='',
                          weight_path='')
    # test.inference(source=source, save_path='./res/')
    test.benchmark(data_path=source)


if __name__ == '__main__':
    main()