# helper functions for detector networks

import os
import numpy as np
import glob
import torch
from torchvision.io import read_image
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import csv
import cv2

class ClusterDataset(Dataset):
    def __init__(self,imgDir, labelDir,transform=None, target_transform=None):
        self.img_dir = imgDir
        self.label_dir = labelDir
        self.all_img = []
        imagePath = imgDir+'/*.jpg'
        for i in glob.glob(imagePath):
            self.all_img.append(i)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.all_img)

    def __getitem__(self,idx):
        image = read_image(self.all_img[idx])
        #create target dictionary for single image
        fileName = self.all_img[idx][len(self.img_dir):-4]
        xml = open(self.label_dir+fileName+'.xml','r')
        lines = xml.readlines()
        num_lines = len(lines)
        boxes = []
        numObjects = 0
        for u in range(13,num_lines-1,12):
            obj = lines[u:u+12]
            xmin = (int(obj[6][9:-8])/850)*600
            ymin = (int(obj[7][9:-8])/850)*600
            xmax = (int(obj[8][9:-8])/850)*600
            ymax = (int(obj[9][9:-8])/850)*600
            boxes.append([xmin,ymin,xmax,ymax])
            numObjects = numObjects+1
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((numObjects,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((numObjects,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        name = [os.listdir(self.img_dir)[idx],]
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        #target["name"] = name
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        #sample = {"image":image, "label":target}
        return image, target, name

def collate_fn(batch): #from pytorch utils
    return tuple(zip(*batch))

def run_FRCNN_model(model, dataloader, device, SaveFileName, toCrop, imgDir):
    model.eval()
    with open(SaveFileName,'w',newline='') as f:
        writer = csv.writer(f)
        header = ['FileName','Prediction']
        writer.writerow(header)
    for images, target, names in dataloader:
        images = (image.float()/255 for image in images)
        images = list(image.to(device) for image in images)
        preds = model(images)
        if toCrop == True:
            for imgNumber in range(len(names)):
                for boxNumber in range(len(preds[imgNumber]['boxes'])):
                    if preds[imgNumber]['scores'][boxNumber] > 0.6:
                        origImg = cv2.imread(imgDir+names[imgNumber][0])
                        #print('Read image')
                        xmin = int(preds[imgNumber]['boxes'][boxNumber][0].item())
                        ymin = int(preds[imgNumber]['boxes'][boxNumber][1].item())
                        xmax = int(preds[imgNumber]['boxes'][boxNumber][2].item())
                        ymax = int(preds[imgNumber]['boxes'][boxNumber][3].item())
                        #print('Have corners')
                        croppedImg = origImg[ymin:ymax,xmin:xmax].copy()
                        #print('made copy')
                        fileName = str(names[imgNumber][0])
                        croppedSavePath = './FRCNNoutput/images/' + fileName[:-4] + str(boxNumber) + '.jpg'
                        cv2.imwrite(croppedSavePath,croppedImg)
        with open(SaveFileName,'a',newline='') as f:
            writer = csv.writer(f)
            for i in range(len(names)):
                row = [names[i],preds[i]]
                writer.writerow(row)
    return