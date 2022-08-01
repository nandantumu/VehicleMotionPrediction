from cgi import test
from socketserver import ThreadingUDPServer
from jinja2 import TemplateRuntimeError
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt
from tqdm.autonotebook import tqdm
from PIL import Image
from typing import List
import yaml
from tensorboardX import SummaryWriter
from enum import Enum, unique
from torch.multiprocessing import freeze_support

freeze_support()


class Curvature(Enum):
    NO_CURVATURE = False
    CURVATURE = True


@unique
class Method(Enum):
    PIMP = 0
    LSTM = 1


DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


def _get_map_points(map_path, map_ext):
    with open(map_path + ".yaml", "r") as yaml_stream:
        try:
            map_metadata = yaml.safe_load(yaml_stream)
            map_resolution = map_metadata["resolution"]
            origin = map_metadata["origin"]
            origin_x = origin[0]
            origin_y = origin[1]
        except yaml.YAMLError as ex:
            print(ex)
    map_img = np.array(
        Image.open(map_path + map_ext).transpose(Image.FLIP_TOP_BOTTOM)
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


map_x, map_y = _get_map_points("../data_generation/track_config/Spielberg_map", ".png")

train_frame = pd.read_pickle("../../data/train_data.pkl")
test_frame = pd.read_pickle("../../data/test_data.pkl")
full_frame = pd.read_pickle("../../data/final_data.pkl")

no_race_train_frame = train_frame[
    train_frame["selected_lane"].apply(
        lambda x: True if x in ["left", "center", "right"] else False
    )
]
no_race_test_frame = test_frame[
    test_frame["selected_lane"].apply(
        lambda x: True if x in ["left", "center", "right"] else False
    )
]
race_test_frame = full_frame[
    full_frame["selected_lane"].apply(lambda x: True if x in ["race"] else False)
]

RACE_SELECTION = [2246, 2329, 2711, 2596, 2465, 2365, 2805, 2554, 2266]


class TraceRelativeDataset(Dataset):
    def __init__(self, dataframe, curve=False):
        self.dataframe = dataframe
        self.curve = curve

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if self.curve:
            key = "input"
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
        return inputs, last_pose, target


train_dataset = TraceRelativeDataset(no_race_train_frame, curve=False)
test_dataset = TraceRelativeDataset(no_race_test_frame, curve=False)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=8)

print(len(train_dataset), " in train")
print(len(test_dataset), " in test")


def bicycle_model_eval(inputs, last_poses):
    # This version takes in an input of dim 5
    BATCHES = inputs.shape[0]
    states = []  # torch.zeros((81, 4))
    L = 0.3302
    TS = 0.1
    X, Y, THETA, V = 0, 1, 2, 3
    state = torch.zeros(
        (
            BATCHES,
            4,
        )
    )
    state[:, X] = last_poses[:, 0]
    state[:, Y] = last_poses[:, 1]
    state[:, THETA] = last_poses[:, 2]
    state[:, V] = inputs[:, 0]
    states.append(state)
    for i in range(1, 81):
        # Advance bicycle model
        state = torch.zeros(
            (
                BATCHES,
                4,
            )
        )
        state[:, X] = states[i - 1][:, X] + (
            TS * states[i - 1][:, V] * torch.cos(states[i - 1][:, THETA])
        )
        state[:, Y] = states[i - 1][:, Y] + (
            TS * states[i - 1][:, V] * torch.sin(states[i - 1][:, THETA])
        )
        state[:, THETA] = states[i - 1][:, THETA] + (
            TS * (states[i - 1][:, V] * torch.tan(inputs[:, 1])) / L
        )
        state[:, V] = states[i - 1][:, V] + TS * inputs[:, 2]
        states.append(state)
    trace = torch.dstack(states).movedim((0, 1, 2), (0, 2, 1))
    trace = trace[:, 1:, :3]
    return trace


def custom_loss_func(prediction, target):
    loss = F.smooth_l1_loss(prediction[:, :, :2], target[:, :, :2])
    loss += 4 * F.smooth_l1_loss(prediction[:, :, 2], target[:, :, 2])
    # loss += 10*output[0]**2 if output[0]<0 else 0
    # loss += 2*torch.linalg.norm(output)**2
    return loss


def average_displacement_error(prediction, target):
    loss = torch.linalg.norm(prediction[:, :, :2] - target[:, :, :2], dim=2)
    ade = torch.mean(loss, dim=0)
    return torch.mean(ade)


def final_displacement_error(prediction, target):
    loss = torch.linalg.norm(prediction[:, -1, :2] - target[:, -1, :2], dim=1)
    return torch.mean(loss)


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, horizon=60, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

        self.hidden2output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            # nn.Linear(hidden_dim//2, hidden_dim//2),
            # nn.ELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 3 * horizon),
        )
        print("WARNING: 2 layer LSTM + 2 layer decoder")

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        output = self.hidden2output(lstm_out)
        output = output[:, -1].reshape((inputs.shape[0], self.horizon, 3))
        return output

    def predict(self, inputs, last_poses, horizon=60):
        residuals = self.forward(inputs)
        last_poses = last_poses.to(DEVICE)
        outputs = torch.tile(
            last_poses[:, :3].reshape(last_poses.shape[0], 1, 3), (1, 60, 1)
        )
        outputs = residuals + outputs
        return outputs


class LSTMPredictorBicycle(nn.Module):
    def __init__(
        self, input_dim=3, hidden_dim=32, control_outputs=1, num_layers=2, horizon=60
    ):
        super(LSTMPredictorBicycle, self).__init__()
        self.hidden_dim = hidden_dim
        self.control_outputs = control_outputs
        self.horizon = horizon

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

        # The linear layer that maps from hidden state space to tag space
        self.hidden2output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            # nn.Linear(hidden_dim//2, hidden_dim//2),
            # nn.ELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, control_outputs * 2 + 1),
        )
        print("WARNING: 2 layer LSTM + 2 layer decoder")

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        output = self.hidden2output(lstm_out)
        scaled_output = list()
        scaled_output.append(F.softplus(output[:, :, 0]))
        for step in range(self.control_outputs):
            scaled_output.append(
                torch.tanh(output[:, :, (step * 2) + 1]) * np.pi
            )  # Steering
            scaled_output.append(output[:, :, (step * 2) + 2])  # Acceleration
        output = torch.dstack(scaled_output)
        return output

    def predict(self, inputs, last_poses):
        # Compute LSTM output
        controls = self.forward(inputs)[:, -1]  # Take last prediction
        last_poses = last_poses.to(DEVICE)
        BATCHES = controls.shape[0]
        states = []  # torch.zeros((81, 4))
        L = 0.3302
        TS = 0.1
        X, Y, THETA, V = 0, 1, 2, 3
        CDIMS = 2
        state = torch.zeros(
            (
                BATCHES,
                4,
            ),
            device=DEVICE,
        )
        state[:, X] = last_poses[:, 0]
        state[:, Y] = last_poses[:, 1]
        state[:, THETA] = last_poses[:, 2]
        state[:, V] = controls[:, 0]
        states.append(state)
        step_length = self.horizon // self.control_outputs
        for i in range(1, self.horizon + 1):
            # Advance bicycle model
            step = min((i - 1) // step_length, self.control_outputs - 1)
            state = torch.zeros(
                (
                    BATCHES,
                    4,
                ),
                device=DEVICE,
            )
            state[:, X] = states[i - 1][:, X] + (
                TS * states[i - 1][:, V] * torch.cos(states[i - 1][:, THETA])
            )
            state[:, Y] = states[i - 1][:, Y] + (
                TS * states[i - 1][:, V] * torch.sin(states[i - 1][:, THETA])
            )
            state[:, THETA] = states[i - 1][:, THETA] + (
                TS
                * (states[i - 1][:, V] * torch.tan(controls[:, (step * CDIMS) + 1]))
                / L
            )
            state[:, V] = states[i - 1][:, V] + TS * controls[:, (step * CDIMS) + 2]
            states.append(state)
        trace = torch.dstack(states).movedim((0, 1, 2), (0, 2, 1))
        trace = trace[:, 1:, :3]
        return trace


def create_debug_plot(
    net,
    train_dataset: TraceRelativeDataset = train_dataset,
    test_dataset: TraceRelativeDataset = test_dataset,
    curvature: Curvature = Curvature.NO_CURVATURE,
    selection: list = [1912, 2465, 533, 905, 277, 1665, 2395, 61, 1054],
):
    if selection is None:
        selection = np.random.choice(len(full_frame), 9)
    inputs, last_poses, targets = train_dataset[: len(selection)]
    inputs = torch.zeros_like(inputs, dtype=torch.float32, device=DEVICE)
    last_poses = torch.zeros_like(last_poses, dtype=torch.float32, device=DEVICE)
    for i, DATA_IDX in enumerate(selection):
        data_in_train = True if DATA_IDX in train_frame.index else False
        dframe = full_frame
        if curvature is Curvature.NO_CURVATURE:
            inputs[i] = torch.tensor(
                dframe.loc[DATA_IDX]["input_no_curve"],
                dtype=torch.float32,
                device=DEVICE,
            )
        else:
            inputs[i] = torch.tensor(
                dframe.loc[DATA_IDX]["input"], dtype=torch.float32, device=DEVICE
            )
        last_poses[i] = torch.tensor(
            dframe.loc[DATA_IDX]["last_pose"], dtype=torch.float32, device=DEVICE
        )
        targets[i] = torch.tensor(
            dframe.loc[DATA_IDX]["target"], dtype=torch.float32, device=DEVICE
        )
    outputs = net.predict(inputs, last_poses).detach().cpu().numpy()

    fig, axs = plt.subplots(3, 3, figsize=(10, 10), dpi=300)
    for idx, DATA_IDX in enumerate(selection):
        data_in_train = True if DATA_IDX in train_frame.index else False
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
        xlim, ylim = np.average(axs[idx // 3, idx % 3].get_xlim()), np.average(
            axs[idx // 3, idx % 3].get_ylim()
        )
        axs[idx // 3, idx % 3].scatter(map_x, map_y, marker=".", color="black")
        axs[idx // 3, idx % 3].set(
            xlim=(xlim - 3.5, xlim + 3.5),
            ylim=(ylim - 3.5, ylim + 3.5),
            aspect=1.0,
            adjustable="box",
            yticklabels=[],
            xticklabels=[],
        )
        axs[idx // 3, idx % 3].set_title(
            "{}:{}:{}".format(
                "Train" if data_in_train else "Test",
                DATA_IDX,
                dframe.loc[DATA_IDX]["selected_lane"],
            )
        )
    fig.suptitle("Manually Selected Traces")
    fig.legend(handles=[linput, ltarget, lpred])
    return fig, axs


def train_PIMP(
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    curvature: Curvature,
    hidden_dim: int,
    epochs: int,
    control_outputs: int,
    prefix: str,
    curriculum: bool = True,
    eps_per_input: int = 3,
    aux_test_dataloader: DataLoader = None,
    aux_selection: List = RACE_SELECTION,
    horizon: int = 60,
):
    torch.autograd.set_detect_anomaly(True)
    net = LSTMPredictorBicycle(
        input_dim=9 if curvature is Curvature.CURVATURE else 3,
        hidden_dim=hidden_dim,
        control_outputs=control_outputs,
        horizon=horizon,
    )
    net.to(DEVICE)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    directory = f"{prefix}/PIMP-{control_outputs}-{curvature.name}-{hidden_dim}-P{pytorch_total_params}/"
    writer = SummaryWriter(directory)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    train_losses = list()
    test_losses = list()
    train_ades, test_ades, race_ades = list(), list(), list()
    train_fdes, test_fdes, race_fdes = list(), list(), list()

    step_size = 60 // control_outputs
    curriculum_steps = eps_per_input * control_outputs

    with tqdm(total=epochs, unit="epochs") as progbar:
        for epoch in range(epochs):

            progbar.set_description("TRAINING")
            cum_train_loss = 0.0
            net.train()
            fde, ade = 0.0, 0.0
            for input_data, last_pose, target_data in train_dataloader:
                net.zero_grad()
                input_data = input_data.to(DEVICE)
                last_pose = last_pose.to(DEVICE)
                outp = net.predict(input_data, last_pose)
                target_data = target_data.to(DEVICE)

                if epoch >= curriculum_steps or not curriculum:
                    loss = custom_loss_func(outp, target_data)
                    end_index = horizon
                else:
                    current_step = epoch // eps_per_input
                    end_index = min(step_size * (current_step + 1), horizon)
                    loss = custom_loss_func(
                        outp[:, :end_index], target_data[:, :end_index]
                    )
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    fde += final_displacement_error(outp, target_data).cpu().numpy()
                    ade += average_displacement_error(outp, target_data).cpu().numpy()
                cum_train_loss += loss.item()
            if curriculum and epoch <= curriculum_steps:
                print(
                    f"Epoch {epoch}, Current Step {current_step}, End Index {end_index}, Step Size {step_size}"
                )
            writer.add_scalar("end_index", end_index, epoch)

            train_fig, train_ax = create_debug_plot(
                net, train_dataloader.dataset, test_dataloader.dataset, curvature
            )
            cum_train_loss /= len(train_dataloader.dataset)
            cum_train_loss *= horizon / end_index
            train_losses.append(cum_train_loss)
            ade, fde = ade / len(train_dataloader.dataset), fde / len(
                train_dataloader.dataset
            )
            writer.add_scalar("ADE/train", ade, epoch)
            writer.add_scalar("FDE/train", fde, epoch)
            train_ades.append(ade)
            train_fdes.append(fde)
            writer.add_figure("train/example_fig", train_fig, epoch)
            writer.add_scalar("loss/train", cum_train_loss, epoch)

            cum_test_loss = 0.0
            fde, ade = 0.0, 0.0
            net.eval()
            progbar.set_description("TESTING")
            with torch.no_grad():
                for input_data, last_pose, target_data in test_dataloader:
                    input_data = input_data.to(DEVICE)
                    last_pose = last_pose.to(DEVICE)
                    outp = net.predict(input_data, last_pose)
                    target_data = target_data.to(DEVICE)
                    loss = custom_loss_func(outp, target_data)
                    cum_test_loss += loss.item()
                    fde += final_displacement_error(outp, target_data).cpu().numpy()
                    ade += average_displacement_error(outp, target_data).cpu().numpy()
            cum_test_loss /= len(test_dataloader.dataset)
            test_losses.append(cum_test_loss)
            ade, fde = ade / len(test_dataloader.dataset), fde / len(
                test_dataloader.dataset
            )
            writer.add_scalar("ADE/test", ade, epoch)
            writer.add_scalar("FDE/test", fde, epoch)
            test_ades.append(ade)
            test_fdes.append(fde)
            writer.add_scalar("loss/test", cum_test_loss, epoch)

            if aux_test_dataloader is not None:
                cum_race_test_loss = 0.0
                fde, ade = 0.0, 0.0
                net.eval()
                progbar.set_description("AUX TESTING")
                with torch.no_grad():
                    for input_data, last_pose, target_data in aux_test_dataloader:
                        input_data = input_data.to(DEVICE)
                        last_pose = last_pose.to(DEVICE)
                        outp = net.predict(input_data, last_pose)
                        target_data = target_data.to(DEVICE)
                        loss = custom_loss_func(outp, target_data)
                        cum_test_loss += loss.item()
                        fde += final_displacement_error(outp, target_data).cpu().numpy()
                        ade += (
                            average_displacement_error(outp, target_data).cpu().numpy()
                        )
                    race_fig, race_ax = create_debug_plot(
                        net,
                        train_dataloader.dataset,
                        test_dataloader.dataset,
                        curvature,
                        selection=aux_selection,
                    )
                cum_race_test_loss /= len(aux_test_dataloader.dataset)
                ade, fde = ade / len(aux_test_dataloader.dataset), fde / len(
                    aux_test_dataloader.dataset
                )
                writer.add_scalar("ADE/race_test", ade, epoch)
                writer.add_scalar("FDE/race_test", fde, epoch)
                race_ades.append(ade)
                race_fdes.append(fde)
                writer.add_figure("race_test/example_fig", race_fig, epoch)
                writer.add_scalar("loss/race", cum_race_test_loss, epoch)

            if cum_test_loss <= min(test_losses):
                torch.save(net.state_dict(), f"{directory}/best_model.pt")
            tqdm.write(
                f"Epoch {epoch} | Train Loss: {cum_train_loss} | Test Loss: {cum_test_loss}"
            )
            progbar.update()

    return {
        "curvature": curvature,
        "hidden_dims": hidden_dim,
        "training_loss": cum_train_loss,
        "test_loss": cum_test_loss,
        "ade": ade,
        "fde": fde,
        "params": pytorch_total_params,
    }


def train_LSTM(
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    curvature: Curvature,
    hidden_dim: int,
    epochs: int,
    prefix: str,
    aux_test_dataloader: DataLoader = None,
    aux_selection: List = RACE_SELECTION,
):
    torch.autograd.set_detect_anomaly(True)
    net = LSTMPredictor(
        input_dim=9 if curvature is Curvature.CURVATURE else 3,
        hidden_dim=hidden_dim,
    )
    net.to(DEVICE)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    directory = f"{prefix}/LSTM-{curvature.name}-{hidden_dim}-P{pytorch_total_params}/"
    writer = SummaryWriter(directory)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    train_losses = list()
    test_losses = list()
    train_ades, test_ades, race_ades = list(), list(), list()
    train_fdes, test_fdes, race_fdes = list(), list(), list()

    with tqdm(total=epochs, unit="epochs") as progbar:
        for epoch in range(epochs):
            i = 0

            progbar.set_description("TRAINING")
            cum_train_loss = 0.0
            net.train()
            fde, ade = 0.0, 0.0
            for input_data, last_pose, target_data in train_dataloader:
                net.zero_grad()
                input_data = input_data.to(DEVICE)
                last_pose = last_pose.to(DEVICE)
                outp = net.predict(input_data, last_pose)
                target_data = target_data.to(DEVICE)
                loss = custom_loss_func(outp, target_data)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    fde += final_displacement_error(outp, target_data).cpu().numpy()
                    ade += average_displacement_error(outp, target_data).cpu().numpy()
                cum_train_loss += loss.item()

            train_fig, train_ax = create_debug_plot(
                net, train_dataloader.dataset, test_dataloader.dataset, curvature
            )
            cum_train_loss /= len(train_dataloader.dataset)
            train_losses.append(cum_train_loss)
            ade, fde = ade / len(train_dataloader.dataset), fde / len(
                train_dataloader.dataset
            )
            writer.add_scalar("ADE/train", ade, epoch)
            writer.add_scalar("FDE/train", fde, epoch)
            train_ades.append(ade)
            train_fdes.append(fde)
            writer.add_figure("train/example_fig", train_fig, epoch)
            writer.add_scalar("loss/train", cum_train_loss, epoch)

            cum_test_loss = 0.0
            fde, ade = 0.0, 0.0
            net.eval()
            progbar.set_description("TESTING")
            with torch.no_grad():
                for input_data, last_pose, target_data in test_dataloader:
                    input_data = input_data.to(DEVICE)
                    last_pose = last_pose.to(DEVICE)
                    outp = net.predict(input_data, last_pose)
                    target_data = target_data.to(DEVICE)
                    loss = custom_loss_func(outp, target_data)
                    cum_test_loss += loss.item()
                    fde += final_displacement_error(outp, target_data).cpu().numpy()
                    ade += average_displacement_error(outp, target_data).cpu().numpy()
            cum_test_loss /= len(test_dataloader.dataset)
            test_losses.append(cum_test_loss)
            ade, fde = ade / len(test_dataloader.dataset), fde / len(
                test_dataloader.dataset
            )
            writer.add_scalar("ADE/test", ade, epoch)
            writer.add_scalar("FDE/test", fde, epoch)
            test_ades.append(ade)
            test_fdes.append(fde)
            writer.add_scalar("loss/test", cum_test_loss, epoch)

            if aux_test_dataloader is not None:
                cum_race_test_loss = 0.0
                fde, ade = 0.0, 0.0
                net.eval()
                progbar.set_description("AUX TESTING")
                with torch.no_grad():
                    for input_data, last_pose, target_data in aux_test_dataloader:
                        input_data = input_data.to(DEVICE)
                        last_pose = last_pose.to(DEVICE)
                        outp = net.predict(input_data, last_pose)
                        target_data = target_data.to(DEVICE)
                        loss = custom_loss_func(outp, target_data)
                        cum_test_loss += loss.item()
                        fde += final_displacement_error(outp, target_data).cpu().numpy()
                        ade += (
                            average_displacement_error(outp, target_data).cpu().numpy()
                        )
                    race_fig, race_ax = create_debug_plot(
                        net,
                        train_dataloader.dataset,
                        test_dataloader.dataset,
                        curvature,
                        selection=aux_selection,
                    )
                cum_race_test_loss /= len(aux_test_dataloader.dataset)
                ade, fde = ade / len(aux_test_dataloader.dataset), fde / len(
                    aux_test_dataloader.dataset
                )
                writer.add_scalar("ADE/race_test", ade, epoch)
                writer.add_scalar("FDE/race_test", fde, epoch)
                race_ades.append(ade)
                race_fdes.append(fde)
                writer.add_figure("race_test/example_fig", race_fig, epoch)
                writer.add_scalar("loss/race", cum_race_test_loss, epoch)

            if cum_test_loss <= min(test_losses):
                torch.save(net.state_dict(), f"{directory}/best_model.pt")
            tqdm.write(
                f"Epoch {epoch} | Train Loss: {cum_train_loss} | Test Loss: {cum_test_loss}"
            )
            progbar.update()

    return {
        "curvature": curvature,
        "hidden_dims": hidden_dim,
        "training_loss": cum_train_loss,
        "test_loss": cum_test_loss,
        "ade": ade,
        "fde": fde,
        "params": pytorch_total_params,
    }


def train(
    method: Method,
    curvature: Curvature,
    hidden_dim: int = 32,
    epochs=100,
    control_outputs=10,
    prefix="runs/toy/robustness-curr",
):
    train_dataset = TraceRelativeDataset(
        no_race_train_frame, curve=True if curvature is Curvature.CURVATURE else False
    )
    test_dataset = TraceRelativeDataset(
        no_race_test_frame, curve=True if curvature is Curvature.CURVATURE else False
    )
    race_dataset = TraceRelativeDataset(
        race_test_frame, curve=True if curvature is Curvature.CURVATURE else False
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    race_dataloader = DataLoader(race_dataset, batch_size=32, shuffle=True)

    if method is Method.PIMP:
        return train_PIMP(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            curvature=curvature,
            hidden_dim=hidden_dim,
            epochs=epochs,
            control_outputs=control_outputs,
            prefix=prefix,
            curriculum=True,
            eps_per_input=2,
            aux_test_dataloader=race_dataloader,
        )
    elif method is Method.LSTM:
        return train_LSTM(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            curvature=curvature,
            hidden_dim=hidden_dim,
            epochs=epochs,
            prefix=prefix,
            aux_test_dataloader=race_dataloader,
        )


def train_catch(*args, **kwargs):
    import traceback

    try:
        train(*args, **kwargs)
    except Exception as e:
        print("Exception Occurred:")
        print(e)
        traceback.print_exc()
        print("--------------------")


import threading
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Process, Pool

if __name__ == "__main__":
    freeze_support()
    jobs = [
        (Method.PIMP, Curvature.CURVATURE, 4, 200, 10),
        (Method.PIMP, Curvature.CURVATURE, 8, 200, 10),
        (Method.PIMP, Curvature.CURVATURE, 16, 200, 10),
        (Method.LSTM, Curvature.CURVATURE, 4, 200),
        (Method.LSTM, Curvature.CURVATURE, 8, 200),
        (Method.LSTM, Curvature.CURVATURE, 16, 200),
        (Method.PIMP, Curvature.CURVATURE, 4, 200, 15),
        (Method.PIMP, Curvature.CURVATURE, 8, 200, 15),
        (Method.PIMP, Curvature.CURVATURE, 16, 200, 15),
        (Method.PIMP, Curvature.CURVATURE, 4, 200, 30),
        (Method.PIMP, Curvature.CURVATURE, 8, 200, 30),
        (Method.PIMP, Curvature.CURVATURE, 16, 200, 30),
    ]

    for j in jobs:
        print(j)

    with multiprocessing.get_context("spawn").Pool(processes=5) as pool:
        res = pool.starmap(train, jobs)

    print(res)
