from torchvision.transforms import Compose, ToTensor, RandomCrop
from dataset import DatasetFromFolderEval, DatasetFromFolder, Train_data

def transform1():
    return Compose([
        RandomCrop((256,256)),
        ToTensor(),
    ])

def transform2():
    return Compose([
        ToTensor(),
    ])

def get_training_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform1())


def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform2())


