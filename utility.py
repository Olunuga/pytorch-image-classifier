from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from classifier_data import ClassifierData
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json

supported_archs = {
    "densenet121": 1024,
    "densenet161": 2208,
    "densenet169": 1664,
    "densnet201": 1920,
    "vgg11": 25088,
    "vgg13": 25088,
    "vgg16": 25088,
    "vgg19": 25088,
    "alexnet": 9216
}

def load_json_file(path):
    with open('cat_to_name.json', 'r') as f:
        loaded_file = json.load(f)
        return loaded_file

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    other_data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_image_datasets = datasets.ImageFolder(train_dir, transform= train_data_transforms)
    vldn_image_datasets = datasets.ImageFolder(valid_dir, transform= other_data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform= other_data_transforms)
    
    train_dataloaders = DataLoader(train_image_datasets, batch_size=64, shuffle=True)
    vld_dataloaders = DataLoader(vldn_image_datasets, batch_size=64, shuffle=True)
    test_dataloaders = DataLoader(test_image_datasets, batch_size=64, shuffle=True)
    classes = train_image_datasets.classes
        
    classifier_data = ClassifierData(train_dataloaders, vld_dataloaders, test_dataloaders, classes)
    return classifier_data

def process_image(image, show=False):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    pil_image = Image.open(image)
    
    if show == True:
        print("Before processing...")
        pil_image.show()
    
    pil_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    
    transformed_img = pil_transform(pil_image)
    
    return transformed_img      