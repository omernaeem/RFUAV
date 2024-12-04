# A train sample code
from utils.trainer import CustomTrainer


def main():
    model = CustomTrainer(cfg='./configs/exp1.8_vit_b_16.yaml')
    model.train()


if __name__ == '__main__':
    main()