#Code snippet to load AlexNet Classifier

import torch
from torchvision import models
import torch.nn as nn

#Set up data
classNames = ['0','1','2','3','4']

# Set up model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = '/home/oliviayem/CompletePipeline/Classifiers/Alex.pt'
alex = models.alexnet(pretrained=False)
fcIndex = len(alex.classifier)-1
originalFC = alex.classifier.__getitem__(fcIndex)
numFeatures = originalFC.in_features
newFC = nn.Linear(numFeatures,len(classNames)) 
alex.classifier.__setitem__(fcIndex,newFC) #replace fully connected layer
alex.load_state_dict(torch.load(path))
alex = alex.to(device)