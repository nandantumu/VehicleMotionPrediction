import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from os.path import normpath
from importlib.resources import files

from overtaking.planners import TripleRacelineLoader

VAL_LIST = [42537, 6231, 23888, 21645, 27858, 14731, 42980, 7037, 24535]
TRAIN_LIST = [42086, 6219, 25331, 21680, 28354, 13329, 42689, 7588, 25676]
TEST_LIST = [43226, 6908, 24000, 21670, 28247, 14169, 42737, 5876, 25405]
RACE_TEST_LIST = [42378, 18229, 41608, 21551, 38352, 20272, 42039, 17109, 41694]
CAR_LENGTH = 0.58
CAR_WIDTH = 0.31

class TraceRelativeDataset(Dataset):
    def __init__(self, dataframe, curve=False, random_noise=False):
        self.dataframe = dataframe
        self.curve = curve
        self.random_noise = random_noise
        self.std_dev = 0.01

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if self.curve:
            key = "input_vel"
        else:
            key = "input_no_curve"

        try:
            inputs = torch.tensor(
                np.array(self.dataframe.iloc[idx][key].to_list()), dtype=torch.float32
            )
            last_pose = torch.tensor(
                np.array(self.dataframe.iloc[idx]["last_pose"].to_list()),
                dtype=torch.float32,
            )
            target = torch.tensor(
                np.array(self.dataframe.iloc[idx]["target"].to_list()),
                dtype=torch.float32,
            )
        except AttributeError as v:
            inputs = torch.tensor(self.dataframe.iloc[idx][key], dtype=torch.float32)
            last_pose = torch.tensor(
                self.dataframe.iloc[idx]["last_pose"], dtype=torch.float32
            )
            target = torch.tensor(
                self.dataframe.iloc[idx]["target"], dtype=torch.float32
            )
        # Add random noise to the inputs, and last_pose
        if self.random_noise:
            inputs[..., :4] += torch.normal(torch.zeros_like(inputs[..., :4]), torch.ones_like(inputs[..., :4])* self.std_dev)
            last_pose += torch.normal(torch.zeros_like(last_pose), torch.ones_like(last_pose)* self.std_dev)

        return inputs, last_pose, target

def get_map_points(map_path, map_ext):
    yaml_path = map_path.with_suffix('.yaml')
    map_path = map_path.with_suffix(map_ext)
    with open(yaml_path, "r") as yaml_stream:
        try:
            map_metadata = yaml.safe_load(yaml_stream)
            map_resolution = map_metadata["resolution"]
            origin = map_metadata["origin"]
            origin_x = origin[0]
            origin_y = origin[1]
        except yaml.YAMLError as ex:
            print(ex)
    map_img = np.array(
        Image.open(map_path).transpose(Image.FLIP_TOP_BOTTOM)
    ).astype(np.float64)
    map_height = map_img.shape[0]
    map_width = map_img.shape[1]

    # convert map pixels to coordinates
    range_x = np.arange(map_width)
    range_y = np.arange(map_height)
    map_x, map_y = np.meshgrid(range_x, range_y)
    map_x = (map_x * map_resolution + origin_x).flatten()
    map_y = (map_y * map_resolution + origin_y).flatten()
    map_z = np.zeros(map_y.shape)
    map_coords = np.vstack((map_x, map_y, map_z))

    # mask and only leave the obstacle points
    map_mask = map_img == 0.0
    map_mask_flat = map_mask.flatten()
    map_points = map_coords[:, map_mask_flat].T
    return map_points[:, 0], map_points[:, 1]

def get_centerline(map_path):
    """Return the centerline in the track"""
    yaml_path = map_path.joinpath('config.yaml')
    with open(yaml_path, "r") as yaml_stream:
        try:
            map_metadata = yaml.safe_load(yaml_stream)
            raceline_path = map_metadata['wpt_path']
        except yaml.YAMLError as ex:
            print(ex)
    raceline_path = normpath(yaml_path.parent/raceline_path)
    planner = TripleRacelineLoader(raceline_path)
    return planner.center


centerline = get_centerline(files('vmp.data.track_config'))
map_x, map_y = get_map_points(files('vmp.data.track_config').joinpath('Spielberg_map'), '.png')

