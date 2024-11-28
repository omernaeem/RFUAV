# A train sample code
from utils.trainer import CustomTrainer


def main():
    model = CustomTrainer(cfg='./configs/exp1.1_ResNet18.yaml')
    model.train()


if __name__ == '__main__':
    main()