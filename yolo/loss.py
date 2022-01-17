import torch
import torch.nn as nn

from utils import IOU

class Yolo_v3_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        #constants
        self.lambda_class = 1
        self.lambda_noobj = 10

        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        obj = target[..., 0]==1
        no_obj = target[..., 0] ==0
        ##no object loss
        no_object_loss = self.bce((predictions[..., 0:1][no_obj], (target[..., 0:1][no_obj])))

        ##object loss
        #at the beginning when we send our anchors they are the dim of 3x2 - 3 anchors and for each a width and the height
        anchors = anchors.reshape(1, 3, 1, 1, 2) #p_w * exp(t_w)

        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5])*anchors], dim = -1)

        ious = IOU(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.bce((predictions[..., 0:1][obj]), (ious*target[..., 0:1]))

        ##box coords loss

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log((target[..., 3:5]/anchors + 1e-16))

        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        ##class loss
        class_loss = self.entropy((predictions[..., 5:][obj]), (target[..., 5:][obj]).long())

        return (self.lambda_box * box_loss + self.lambda_obj*object_loss + self.lambda_class*class_loss + self.lambda_noobj*no_object_loss)



