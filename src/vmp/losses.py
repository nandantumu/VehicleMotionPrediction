import torch
from torch.nn import functional as F
import random
import numpy as np
import shapely.geometry as sg
import shapely.affinity as sa
from .data_utils import CAR_LENGTH, CAR_WIDTH
from joblib import Parallel, delayed

def intersection_over_union(prediction, target):
    """Calculate the intersection over union of two bounding boxes.

    Args:
        prediction (np.ndarray): Single point, x,y,theta
        target (np.ndarray): Single point, x,y,theta

    Returns:
        float: IoU
    """
    pred = sg.box(-CAR_LENGTH/2, -CAR_WIDTH/2, CAR_LENGTH/2, CAR_WIDTH/2)
    pred = sa.rotate(pred, prediction[..., 2], origin=(0, 0), use_radians=True)
    pred = sa.translate(pred, prediction[..., 0], prediction[..., 1])

    targ = sg.box(-CAR_LENGTH/2, -CAR_WIDTH/2, CAR_LENGTH/2, CAR_WIDTH/2)
    targ = sa.rotate(targ, target[..., 2], origin=(0, 0), use_radians=True)
    targ = sa.translate(targ, target[..., 0], target[..., 1])

    return pred.intersection(targ).area / pred.union(targ).area

def iou_single_pred(prediction, target):
    """Predict for single row"""
    ious = torch.zeros(prediction.shape[0])
    for i in range(prediction.shape[0]):
        ious[i] = intersection_over_union(prediction[i], target[i])
    return torch.mean(ious)

def iou_metric(prediction, target, workers=8):
    """This function calculates the IoU for each prediction and target pair. 

    Args:
        prediction (torch.tensor): N,HORIZON,3
        target (torch.tensor): N,HORIZON,3
    """
    ious = torch.zeros(prediction.shape[0], prediction.shape[1])
    jobs = [(prediction[i].cpu().numpy(), target[i].cpu().numpy()) for i in range(prediction.shape[0])]
    results = Parallel(n_jobs=workers)(delayed(iou_single_pred)(i[0], i[1]) for i in jobs)
    return torch.sum(torch.tensor(results))

def custom_loss_func(prediction, target, control_outputs=None):
    loss = F.l1_loss(prediction[..., :2], target[..., :2])
    loss += 4 * F.l1_loss(prediction[..., 2], target[..., 2])
    # loss += 1e-5 * torch.linalg.norm(control_outputs[1:] - control_outputs[:-1])
    return loss

@DeprecationWarning
def custom_loss_func_lstm(prediction, target):
    loss = F.l1_loss(prediction[..., :2], target[..., :2])
    loss += 4 * F.l1_loss(prediction[..., 2], target[..., 2])
    return loss

def standard_loss_func(prediction, target, control_outputs=None):
    loss = F.mse_loss(prediction[..., :2], target[..., :2])
    return loss

def heading_error(prediction, target):
    loss = torch.linalg.norm(prediction[..., 2] - target[..., 2], dim=-1)
    heading_error = torch.mean(loss, dim=-1)
    return torch.sum(heading_error)


def average_displacement_error(prediction, target):
    loss = torch.linalg.norm(prediction[..., :2] - target[..., :2], dim=-1)
    ade = torch.mean(loss, dim=-1)
    return torch.sum(ade)


def final_displacement_error(prediction, target):
    loss = torch.linalg.norm(prediction[..., :2] - target[..., :2], dim=-1)
    fde = loss[..., -1]
    return torch.sum(fde)

def displacement_error(prediction, target):
    loss = torch.linalg.norm(prediction[..., :2] - target[..., :2], dim=-1)
    return loss