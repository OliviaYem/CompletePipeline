# Code snippet to load the VGG classifier

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import ClassifierFunctions as cf

#Set up data
images = '/home/oliviayem/ClassificationDataset/test/'
transform = transforms.Compose([transforms.Resize((255,255))])
data = cf.CroppedClusterDataset(images,transform=transform)
dataloader = DataLoader(data, batch_size=32, shuffle=False)
classNames = ['0','1','2','3','4']

# Set up model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = '/home/oliviayem/CompletePipeline/Classifiers/VGG.pt'
vgg = models.vgg16(pretrained=True)
fcIndex = len(vgg.classifier)-1
originalFC = vgg.classifier.__getitem__(fcIndex)
numFeatures = originalFC.in_features
newFC = nn.Linear(numFeatures,len(classNames)) 
vgg.classifier.__setitem__(fcIndex,newFC) #replace fully connected layer
vgg.load_state_dict(torch.load(path))
model = vgg.to(device)

# run inferences
SaveFileName = '/home/oliviayem/CompletePipeline/testExample.csv'
cf.run_model(model,dataloader,device,classNames,SaveFileName)