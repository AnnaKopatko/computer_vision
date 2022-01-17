import torch

#area of intersection divided by the area of union
#every box is [x1, y1, x2, y2]

#first in intersection:
#x1_inter  = max(x1_box1, x1_box_2), y1_inter  = max(y1_box1, y1_box_2)
#x2_inter  = min(x2_box1, x2_box_2), y2_inter  = min(y2_box2, y2_box_2)
def IOU(boxes_preds, boxes_labels, box_format = "midpoint"):

    if box_format == "midpoint":
        #we take the x of the middle minus the width/2
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3]/2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2


        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    else:
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]

        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]

        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]


    x1 = torch.max(box1_x1, box2_x1)
    x2 = torch.max(box1_x2, box2_x2)
    y1 = torch.max(box1_y1, box2_y1)
    y2 = torch.max(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0)*(y2 - y1).clamp(0)

    box1_area = abs((box1_x1 - box1_x2)*(box1_y2 - box1_y1))
    box2_area = abs((box2_x1 - box2_x2) * (box2_y2 - box2_y1))

    union = box1_area + box2_area - intersection

    return intersection/(union + 1e-6)



def Non_Max_Supression(boxes, iou_threshold, prob_threshold, box_format = "midpoint"):
    assert type(boxes)==list
    boxes = [box for box in boxes if box[1] > prob_threshold]
    boxes_after_nms = []

    #we sort the bounding boxes with the highest prob at the beginning
    boxes = sorted(boxes, key = lambda x: x[1])

    while boxes:
        chosen_box = boxes.pop[0]

        #so we get rid of the boxes of the same class and only take thoose that have the small iou with the main box(so this can be a different object in the picture)
        boxes = [box for box in boxes if box[0] != chosen_box[0] or IOU(torch.tensor(chosen_box)[2:], torch.tensor(box[2:])) < iou_threshold]

        boxes_after_nms.append(chosen_box)

    return boxes_after_nms




