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
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from detectron2.evaluation.evaluator import DatasetEvaluator, DatasetEvaluators, inference_context
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
from torch import nn


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

def run_detectron_model(model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], toCrop):
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if toCrop == True:
                for imgNumber in range(len(outputs)):
                    for boxNumber in range(len(outputs[imgNumber]["instances"].pred_boxes)):
                        if outputs[imgNumber]["instances"].scores[boxNumber] > 0.6:
                            origImg = cv2.imread(inputs[0]['file_name'])
                            xmin = int(outputs[imgNumber]['instances'].pred_boxes[boxNumber].tensor[0,0].item())
                            ymin = int(outputs[imgNumber]['instances'].pred_boxes[boxNumber].tensor[0,1].item())
                            xmax = int(outputs[imgNumber]['instances'].pred_boxes[boxNumber].tensor[0,2].item())
                            ymax = int(outputs[imgNumber]['instances'].pred_boxes[boxNumber].tensor[0,3].item())
                            croppedImg = origImg[ymin:ymax,xmin:xmax].copy()
                            croppedSavePath = './DetectronOutput/images/' + inputs[imgNumber]['image_id'] + str(boxNumber) + '.jpg'
                            cv2.imwrite(croppedSavePath,croppedImg)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results