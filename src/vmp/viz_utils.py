import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import torch

from .data_utils import (
    RACE_TEST_LIST,
    TEST_LIST,
    TRAIN_LIST,
    VAL_LIST,
    TraceRelativeDataset,
    map_x,
    map_y,
    CAR_LENGTH,
    CAR_WIDTH
)
from .networks import LSTMPredictorPCMP
from .utils import Curvature

sns.set_style()
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "Rubik"
sns.set_context("paper")

def rad_to_deg(radians):
    return radians*57.29578

def create_single_plot(input, target, prediction, ax=None, heading_boxes=True, buffer=0.5, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = None
    assert(isinstance(input, np.ndarray))
    assert(isinstance(target, np.ndarray))
    assert(isinstance(prediction, np.ndarray))
    
    (linput,) = ax.plot(
        input[:, 0],
        input[:, 1],
        marker=".",
        label="Input",
    )
    (ltarget,) = ax.plot(
        target[:, 0],
        target[:, 1],
        marker=".",
        label="Truth",
    )
    (lpred,) = ax.plot(
        prediction[:, 0],
        prediction[:, 1],
        marker=".",
        label="Prediction",
    )

    if heading_boxes:
        center = (prediction[-1, 0], prediction[-1, 1])
        corner = [center[0] - (CAR_LENGTH / 2), center[1] - (CAR_WIDTH / 2)]
        ax.add_patch(
            plt.Rectangle(
                corner,
                CAR_LENGTH,
                CAR_WIDTH,
                edgecolor="green",
                fill=False,
                angle=rad_to_deg(prediction[-1, 2]),
                rotation_point=center,
            )
        )

        center = (target[-1, 0], target[-1, 1])
        corner = [center[0] - (CAR_LENGTH / 2), center[1] - (CAR_WIDTH / 2)]
        ax.add_patch(
            plt.Rectangle(
                corner,
                CAR_LENGTH,
                CAR_WIDTH,
                edgecolor="orange",
                fill=False,
                angle=rad_to_deg(target[-1, 2]),
                rotation_point=center,
            )
        )
        center = (prediction[-30, 0], prediction[-30, 1])
        corner = [center[0] - (CAR_LENGTH / 2), center[1] - (CAR_WIDTH / 2)]
        ax.add_patch(
            plt.Rectangle(
                corner,
                CAR_LENGTH,
                CAR_WIDTH,
                edgecolor="green",
                fill=False,
                angle=rad_to_deg(prediction[-30, 2]),
                rotation_point=center,
            )
        )
        center = (
            target[-30, 0],
            target[-30, 1],
        )
        corner = [center[0] - (CAR_LENGTH / 2), center[1] - (CAR_WIDTH / 2)]
        ax.add_patch(
            plt.Rectangle(
                corner,
                CAR_LENGTH,
                CAR_WIDTH,
                edgecolor="orange",
                fill=False,
                angle=rad_to_deg(target[-30, 2]),
                rotation_point=center,
            )
        )

        center = (
            input[-1, 0],
            input[-1, 1],
        )
        corner = [center[0] - (CAR_LENGTH / 2), center[1] - (CAR_WIDTH / 2)]
        ax.add_patch(
            plt.Rectangle(
                corner,
                CAR_LENGTH,
                CAR_WIDTH,
                edgecolor="blue",
                fill=False,
                angle=rad_to_deg(input[-1, 2]),
                rotation_point=center,
            )
        )

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    max_offset_x = max(np.mean(xlim) - xlim[0], xlim[1] - np.mean(xlim))
    max_offset_y = max(np.mean(ylim) - ylim[0], ylim[1] - np.mean(ylim))
    max_offset = max(max_offset_x, max_offset_y) + buffer
    ax.scatter(map_x, map_y, marker=".", color="black")
    ax.set(
        xlim=(np.mean(xlim) - max_offset, np.mean(xlim) + max_offset),
        ylim=(np.mean(ylim) - max_offset, np.mean(ylim) + max_offset),
        aspect=1.0,
        #adjustable="box",
        yticklabels=[],
        xticklabels=[],
    )
    ax.tick_params(          
        axis='both',
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False
    )
    if fig is None:
        return ax
    else:
        return fig, ax

def create_debug_plot(
    net,
    train_dataset: TraceRelativeDataset,
    test_dataset: TraceRelativeDataset = None,
    val_dataset: TraceRelativeDataset = None,
    curvature: Curvature = Curvature.NO_CURVATURE,
    selection=None,
    DEVICE="cpu",
    full_frame=None,
    dataset_name="train",
):
    if selection is None:
        if dataset_name.lower() == "train":
            selection = TRAIN_LIST
        elif dataset_name.lower() == "test":
            selection = TEST_LIST
        elif dataset_name.lower() == "val":
            selection = VAL_LIST
        elif dataset_name.lower() == "race_test":
            selection = RACE_TEST_LIST
        else:
            selection = np.random.choice(len(full_frame), 9)
    inputs, last_poses, targets = train_dataset[: len(selection)]
    inputs = torch.zeros_like(inputs, dtype=torch.float32, device=DEVICE)
    last_poses = torch.zeros_like(last_poses, dtype=torch.float32, device=DEVICE)
    for i, DATA_IDX in enumerate(selection):
        dframe = full_frame
        if curvature is Curvature.NO_CURVATURE:
            inputs[i] = torch.tensor(
                dframe.loc[DATA_IDX]["input_no_curve"],
                dtype=torch.float32,
                device=DEVICE,
            )
        else:
            inputs[i] = torch.tensor(
                dframe.loc[DATA_IDX]["input_vel"], dtype=torch.float32, device=DEVICE
            )
        last_poses[i] = torch.tensor(
            dframe.loc[DATA_IDX]["last_pose"], dtype=torch.float32, device=DEVICE
        )
        targets[i] = torch.tensor(
            dframe.loc[DATA_IDX]["target"], dtype=torch.float32, device=DEVICE
        )
    if isinstance(net, LSTMPredictorPCMP):
        outputs, _ = net.predict(inputs, last_poses)
        outputs = outputs.detach().cpu().numpy()
    else:
        outputs = net.predict(inputs, last_poses).detach().cpu().numpy()

    fig, axs = plt.subplots(3, 3, figsize=(10, 10), dpi=300)
    for idx, DATA_IDX in enumerate(selection):
        data_in_set = "Train" if DATA_IDX in train_dataset.dataframe.index else False
        if not data_in_set:
            data_in_set = "Test" if DATA_IDX in test_dataset.dataframe.index else "Val"
        dframe = full_frame
        (linput,) = axs[idx // 3, idx % 3].plot(
            dframe.loc[DATA_IDX]["input"][:, 0],
            dframe.loc[DATA_IDX]["input"][:, 1],
            marker=".",
            label="Input",
        )
        (ltarget,) = axs[idx // 3, idx % 3].plot(
            dframe.loc[DATA_IDX]["target"][:, 0],
            dframe.loc[DATA_IDX]["target"][:, 1],
            marker=".",
            label="Target",
        )
        (lpred,) = axs[idx // 3, idx % 3].plot(
            outputs[idx, :, 0],
            outputs[idx, :, 1],
            marker="o",
            mfc="none",
            label="Prediction",
        )
        # Create five boxes, middle, end
        # Last box first
        center = (outputs[idx, -1, 0], outputs[idx, -1, 1])
        corner = [center[0] - (CAR_LENGTH / 2), center[1] - (CAR_WIDTH / 2)]
        axs[idx // 3, idx % 3].add_patch(
            plt.Rectangle(
                corner,
                CAR_LENGTH,
                CAR_WIDTH,
                edgecolor="green",
                fill=False,
                angle=rad_to_deg(outputs[idx, -1, 2]),
                rotation_point=center,
            )
        )
        
        center = (dframe.loc[DATA_IDX]["target"][-1, 0], dframe.loc[DATA_IDX]["target"][-1, 1],)
        corner = [center[0] - (CAR_LENGTH / 2), center[1] - (CAR_WIDTH / 2)]
        axs[idx // 3, idx % 3].add_patch(
            plt.Rectangle(
                corner,
                CAR_LENGTH,
                CAR_WIDTH,
                edgecolor="orange",
                fill=False,
                angle=rad_to_deg(dframe.loc[DATA_IDX]["target"][-1, 2]),
                rotation_point=center,
            )
        )
        center = (outputs[idx, -30, 0], outputs[idx, -30, 1])
        corner = [center[0] - (CAR_LENGTH / 2), center[1] - (CAR_WIDTH / 2)]
        axs[idx // 3, idx % 3].add_patch(
            plt.Rectangle(
                corner,
                CAR_LENGTH,
                CAR_WIDTH,
                edgecolor="green",
                fill=False,
                angle=rad_to_deg(outputs[idx, -30, 2]),
                rotation_point=center,
            )
        )
        center = (
            dframe.loc[DATA_IDX]["target"][-30, 0],
            dframe.loc[DATA_IDX]["target"][-30, 1],
        )
        corner = [center[0] - (CAR_LENGTH / 2), center[1] - (CAR_WIDTH / 2)]
        axs[idx // 3, idx % 3].add_patch(
            plt.Rectangle(
                corner,
                CAR_LENGTH,
                CAR_WIDTH,
                edgecolor="orange",
                fill=False,
                angle=rad_to_deg(dframe.loc[DATA_IDX]["target"][-30, 2]),
                rotation_point=center,
            )
        )

        center = (
            dframe.loc[DATA_IDX]["input"][-1, 0],
            dframe.loc[DATA_IDX]["input"][-1, 1],
        )
        corner = [center[0] - (CAR_LENGTH / 2), center[1] - (CAR_WIDTH / 2)]
        axs[idx // 3, idx % 3].add_patch(
            plt.Rectangle(
                corner,
                CAR_LENGTH,
                CAR_WIDTH,
                edgecolor="blue",
                fill=False,
                angle=rad_to_deg(dframe.loc[DATA_IDX]["input"][-1, 2]),
                rotation_point=center,
            )
        )

        xlim = np.average(axs[idx // 3, idx % 3].get_xlim())
        ylim = np.average(axs[idx // 3, idx % 3].get_ylim())
        axs[idx // 3, idx % 3].scatter(map_x, map_y, marker=".", color="black")
        axs[idx // 3, idx % 3].set(
            xlim=(xlim - 3, xlim + 3),
            ylim=(ylim - 3, ylim + 3),
            aspect=1.0,
            adjustable="box",
            yticklabels=[],
            xticklabels=[],
        )
        axs[idx // 3, idx % 3].set_title(
            "{}|{}|{}".format(
                data_in_set,
                DATA_IDX,
                dframe.loc[DATA_IDX]["selected_lane"],
            )
        )
    fig.suptitle(f"{data_in_set} Manually Selected Traces")
    fig.legend(handles=[linput, ltarget, lpred])
    return fig, axs
