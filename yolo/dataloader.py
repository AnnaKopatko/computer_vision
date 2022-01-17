import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import IOU_width_height
from utils import Non_Max_Supression


ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLO(Dataset):
    def __init__(self, csv_file, image_dir, label_dir, anchors, image_size = 416, scales = [13, 26, 52], num_classes = 20, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transfom = transform
        self.scales = scales
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors [2])
        self.num_anchors = anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors//3
        self.num_classes = num_classes
        self.ignore_iou_threshold = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        #we use np.roll in order to switch the dims, make the class dim the last one
        boxes = np.roll(np.loadtxt(frame = label_path, delimeter = ' ', ndmin = 2).tolist())
        image_path = os.path.join(self.image_dir,self.annotations.iloc[index, 0])
        image = np.array(Image.open(image_path).convert("RGB"))

        if self.transfom:
            augmentations = self.transfom(image = image, bboxes = boxes)
            image = augmentations["image"]
            boxes = augmentations["bboxes"]

        targets = [torch.zeros((self.num_anchors//3, S, S, 6)) for S in self.S]

        # for every bounding box we want to assign an anchor and a cell for each of the three scales
        for box in boxes:

            #we calc the IOU just from the width and the height between one box and all of the anchors
            iou_anchors = IOU(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending = True, dim = 1)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx//self.num_anchors_per_scale #this can be 0, 1, 2(we compute which scale)
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale#this also can be 0, 1, 2(we compute which anchor in this scale0

                this_scale = self.S[scale_idx]

                #whish cell has this anchor
                i, j = int(this_scale * y), int(this_scale*x)   #x = 0.5, scale = 13 --> int(6.5) = 6
                anchor_taken  = targets[scale_idx][anchor_on_scale, i, j, 0]

                #we loop over the anchors because every scale needs to have it's own anchor
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    # we want to rescale the image relative to the cell(where it is relative to the whole image right now)
                    x_cell, y_cell = this_scale*x - j, this_scale*y - i
                    width_cell, height_cell = (width*this_scale, height*this_scale)

                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] =int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_threshold:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 #ignore this prediction

        return image, tuple(targets)










