# easiest to run command line detection from YOLOv5PyTorch library/directory

# python detect.py -source /home/oliviayem/DetectionDataset/test/images/ --weights --img 850

# or

import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Images
dir = '/home/oliviayem/DetectionDataset/test/images/'
imgs = [dir + f for f in ('zidane.jpg', 'bus.jpg')]  # batch of images

# Inference
results = model(imgs)
results.print()  # or .show(), .save()