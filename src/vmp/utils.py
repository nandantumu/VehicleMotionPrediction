import torch
import random
import numpy as np
from enum import IntEnum, unique
from .data_utils import TraceRelativeDataset, TRAIN_LIST, VAL_LIST, TEST_LIST, RACE_TEST_LIST, map_x, map_y
import matplotlib.pyplot as plt

class Curvature(IntEnum):
    NO_CURVATURE = False
    CURVATURE = True

@unique
class Method(IntEnum):
    PIMP = 0
    LSTM = 1
    CTRV = 2
    CTRA = 3

@unique
class DynamicModel(IntEnum):
    BICYCLE = 0
    UNICYCLE = 1

def set_seed(seed=1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def nearest_point_on_trajectory(point, trajectory):
    '''
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    '''
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    dists = np.linalg.norm(point - projections,axis=1)
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

def get_interpolated_point(index, centerline):
    x = np.interp(index, np.arange(centerline.shape[0]), centerline[:,0])
    y = np.interp(index, np.arange(centerline.shape[0]), centerline[:,1])
    return x,y

def point_to_frenet(point, centerline):
    """
    Args:
        point (np.ndarray): [pointx, pointy]
        centerline (np.ndarray): 2x1000, [x, y]^1000
    """
    _, _, offset, progress = nearest_point_on_trajectory(point, centerline[:, :2])
    progress = progress + offset
    nearest_line_pt = get_interpolated_point(progress, centerline[:, :2])
    line_pt_plus_one = get_interpolated_point(progress+1, centerline[:, :2])
    line_to_pose = np.array(nearest_line_pt)-np.array(point)
    line_direction = np.array(line_pt_plus_one)-np.array(nearest_line_pt)
    # Direction Vector
    offset_vector = np.cross(np.array([line_direction[0], line_direction[1], 0]), np.array([0, 0, 1]))[:2]
    sign = np.sign(np.dot(offset_vector, line_to_pose))
    deflection = np.linalg.norm(line_to_pose)*sign
    
    # deflection = np.linalg.norm(line_to_pose) * np.sign(
    #             np.arctan2(line_direction[1], line_direction[0])
    #             - np.arctan2(line_to_pose[1], line_to_pose[0])
    #         )
    # We need to scale the points to be contiguous
    # diffs = progress%1000
    # if np.any(diffs>0):
    #     diffs_mask = diffs > 0
    #     add_comp = np.zeros_like(progress) + 1000*np.ones_like(diffs)*diffs_mask
    #     progress = progress + add_comp
    return np.array([progress, deflection])

def frenet_to_point(f_point, centerline):
    """
    Args:
        f_point (np.ndarray): [point_prog, point_deflection]
        centerline (np.ndarray): 2x1000, [x, y]^1000
    """
    progress, deflection = f_point[0], f_point[1]
    progress = progress%1000
    center_point = get_interpolated_point(progress, centerline)
    line_pt_plus_one = get_interpolated_point((progress+1)%1000, centerline[:, :2])
    line_direction = np.array([line_pt_plus_one[i]-center_point[i] for i in range(2)])
    line_angle = np.arctan2(line_direction[1], line_direction[0])
    unit_vec = np.array([np.cos(line_angle+(np.pi/2)), np.sin(line_angle+(np.pi/2))])
    scaled_vec = deflection*unit_vec
    return center_point + scaled_vec