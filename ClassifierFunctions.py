# Helper functions for the classifiers used in the pipeline

from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import torch
import csv


# custom dataset for cropped clusters, no labels
class CroppedClusterDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        dirFiles = os.listdir(self.img_dir)
        return len(dirFiles)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        name = self.img_labels.iloc[idx, 0]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        sample = {"image": image, "name": name}
        return sample

# function to run inferences and output the results
def run_model(model,dataloader,device,classes,SaveFileName):
    model.eval()
    for (idx,batch) in enumerate(dataloader):
        inputs = batch['image']
        names = batch['name']
        inputs = inputs.float()
        inputs = inputs.to(device)
        outputs = model(inputs)
        _,preds = torch.max(outputs,1)

    with open(SaveFileName,'w',newline='') as f:
        writer = csv.writer(f)
        header = ['FileName','Prediction']
        writer.writerow(header)
        for i in range(inputs.size()[0]):
            row = [names[i],classes[preds[i]]]
            writer.writerow(row)
    print('Inferences completed')
    return