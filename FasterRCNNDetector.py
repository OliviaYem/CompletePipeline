# FasterRCNNDetector.py
# Written by Olivia Yem, 4.9.2021
# code snippet to load and run the Faster RCNN detection network

import torch
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
import DetectorFunctions as df

# load data
print("Importing data")
testImageDir = '/home/oliviayem/FasterRCNNPyTorch/test/images/'
testLabelDir = '/home/oliviayem/FasterRCNNPyTorch/test/labels/'

dataTransforms = transforms.Compose([transforms.Resize((600,600))])
testData = df.ClusterDataset(testImageDir,testLabelDir,transform=dataTransforms)
testDataloader = DataLoader(testData, batch_size=1, shuffle=True,collate_fn=df.collate_fn)

# set up model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = '/home/oliviayem/CompletePipeline/Detectors/FRCNN.pt'
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
numClasses = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, numClasses)
model.load_state_dict(torch.load(path))
model.to(device)

# run inferences
df.run_FRCNN_model(model,testDataloader,device, './FRCNNoutput/FRCNNoutput.csv', True, testImageDir)
# print('Name')
# print(names[0])
# print('All preds')
# print(preds[0])
# print('All boxes')
# print(preds[0]['boxes'])
# print('length of boxes tensor')
# print(len(preds[0]['boxes']))
# print('First box')
# print(preds[0]['boxes'][0])
# print('first number')
# print(preds[0]['boxes'][0][0].item())
# print('second number')
# print(preds[0]['boxes'][0][1].item())