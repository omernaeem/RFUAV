from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
# from PIL import Image
import os
import numpy
# from ..utils import build


"""
CFG中的内容
"""
# Defalut_CFG = build.build_from_cfg('')  # 之后的解析结果
Defalut_CFG = None
temp = "/home/frank/Dataset/leaf_test"

train_path = "/home/frank/Dataset/leaf_test/train"
val_path = "/home/frank/Dataset/leaf_test/valid"
# 定义数据预处理操作
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

"""DatasetLoader for Spectrum Classification
Version 1.1
需要加数据集与CFG之间的纠错功能
"""
class SpectrumClassifyDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        print("Loading dataset...")
        print(f"Total {len(self.classes)} classes: {self.classes}")
        print(f"Total {len(self.imgs)} images")


class SpectrumClassifyDataLoader(Dataset):
    """
    Special dataset class for loading and processing Spectrum.

    Args:
        root (str): Path to the folder containing train/test/val set.
        imagsize (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        batch_size (int, optional): Size of batches. Defaults to 8.
        class_to_idx (dict): dict of included classes and idx.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        transforms (callable): Image transformation function.
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        testset_noexit: If True, testset does not exist. Defaults to False.
    """
    def __init__(self, 
                 root,
                 augment=False,
                 image_size=640,
                 cache=False,
                 hyp=Defalut_CFG,  # 配置文件********写build cfg的函数
                 batch_size=8,
                 num_worker=4,
                 device='0',  # 默认使用gpu,
                 transforms=Defalut_CFG,  # 数据预处理*********xiebulid cfg函数
                 ):
        super().__init__()
        self.transforms = transforms
        self.datasetinfo = {}
        self.trainset = []
        self.valset = []
        self.testset = []
        self.testset_noexit = False
        self.class_to_idx = {}

        # 可行性检查
        if not os.path.exists(root):
            raise FileNotFoundError(f"{root} does not exist!")
        else:
            if not os.path.exists(root + '/train'):
                raise FileNotFoundError("Trainset is not exit!")
            if not os.path.exists(root + '/valid'):
                raise FileNotFoundError("Valset is not a exit!")
            if not os.path.exists(root + '/test'):
                self.testset_noexit = True
                print("Warning testset is not detected in your dataset!")
                
        self.class_to_idx = self.class2indx(root + '/train')
        self.load_img(root, ['.jpg','.png'])
        
        # 图像reshape和数据增强
        
        train_set = self.build_dataloader(self.trainset, batch_size=batch_size, num_workers=num_worker)
        valid_set = self.build_dataloader(self.trainset, batch_size=batch_size, num_workers=num_worker)
        if not self.testset_noexit:
            test_set = self.build_dataloader(self.trainset, batch_size=batch_size, num_workers=num_worker)

    def __getitem__(self, index):
        return self.datasetinfo['total_data'][index]

    def __len__(self):
        return len(self.datasetinfo['total_data'])
    
    def load_img(self, root, extensions):
        """
        从本地文件夹中读数据
        """
        train_path = os.path.join(root, 'train')
        val_path = os.path.join(root, 'valid')
        test_path = os.path.join(root, 'test')
        
        self.trainset = load_data_TVT(train_path, self.class_to_idx, extensions)
        self.valset = load_data_TVT(val_path, self.class_to_idx, extensions)
        if not self.testset_noexit:
            self.testset = load_data_TVT(test_path, self.class_to_idx, extensions)
        self.datasetinfo['len_trainset'] = len(self.trainset)
        self.datasetinfo['len_valset'] = len(self.valset)
        self.datasetinfo['len_testset'] = len(self.testset) if not self.testset_noexit else 0
        self.datasetinfo['total_data'] = self.trainset + self.valset + self.testset if not self.testset_noexit else self.trainset + self.valset
        
    def build_dataloader(
            self,
            image_path,
            batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True, 
    ):
        return DataLoader(self, image_path, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        
    def class2indx(self, root):
        """
        Returns the index of the class name.
        """
        print("Loading the dataset")
        classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {root}.")
        print(f"Found {len(classes)} classes: {classes}")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return class_to_idx
    
    def augmentation(self, hyp=None):
        """
        Builds the augmentation pipeline.
        """
        
    def reshape_image(self, image_size):
        """
        预处理图像将图像规格化
        """
        
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
    
    
def has_file_allowed_extension(filename, extensions) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def load_data_TVT(root, class_to_idx, extensions):
    
    instance = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(root, target_class)
        if not os.path.isdir(target_dir):
            continue
        for _root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(_root, fname)
                if has_file_allowed_extension(path, extensions):
                    item = path, class_index
                    instance.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    # 检查trainset中是否有未出现的类别
    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)
    
    return instance


def build_dataset(cfg) -> Dataset:
    """
    承接的参数应该是build CFG之后的
    最终接口，接到train里面
    """
    

# 测试用例
dataset = SpectrumClassifyDataLoader(root=temp, transforms=None)
td = DataLoader(dataset.trainset, batch_size=32, shuffle=True, num_workers=4)
vd = DataLoader(dataset.valset, batch_size=32, shuffle=True, num_workers=4)

"""
trainset = SpectrumClassifyDataset(root=train_path, transform=transform)
valset = SpectrumClassifyDataset(root=val_path, transform=transform)
td = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
vd = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
"""

"""Usage
    Dataset = SpectrumClassifyDataLoader(cfg)
    return SpectrumClassifyDataLoader(Dataset)
"""