# A train sample code
from utils.trainer import CustomTrainer


def main():
    model = CustomTrainer(cfg='./configs/exp1.7_mobilenet_v3_l.yaml')
    model.train()


if __name__ == '__main__':
    main()