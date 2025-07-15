# A sample script to test the model.
from utils.benchmark import Classify_Model

def main():

    # The models need to be downloaded first and put in models directory.
    # I have adjusted the model configurations to match the downloaded models for following 3 models.
    # 1. MobileNet V3 Small
    # 2. ResNet101
    # 3. ViT-L/16


    # Uncomment the model you want to test.

    # test = Classify_Model(cfg='configs/exp1.6_mobilenet_v3_s.yaml',
    #                       weight_path='models/mobilenet_v3_small.pth')
    
    # test = Classify_Model(cfg='configs/exp1.4_ResNet101.yaml',
    #                    weight_path='models/ResNet101.pth')

    test = Classify_Model(cfg='configs/exp1.10_vit_l_16.yaml',
                       weight_path='models/vit_l_16.pth')
    

    # For inference, either images or raw data can be used.
    # In the following case I am using the validation data from DJI AVTA2 dataset.

    #source = '/home/omer/drone/RFUAV/data/valid/DJI AVTA2/12dB/hsv/1024/'
    
    # Another option is to use raw data
    # In the following case I am using the DJI Mini 3 raw data.

    source = 'data/raw/DJI MINI3/VTSBW=10/'
    
    test.inference(source=source, save_path='results/')


    # For Benchmarking, setup the validation dataset and then uncomment the following lines.

    #Download and extract the validation dataset in this directory
    #For example: Extracting DJI AVTA2 dataset will give you a directory structure like:
    #data/valid/DJI AVTA2/<-2db to 20dB>/<autumn, hot, hsv, parula>/<128, 256, 512, 1024>/
    # I only found the images in 1024 to be correct, so I removed the others.

    # source = 'data/'
    # test.benchmark(data_path=source)


if __name__ == '__main__':
    main()