import numpy as np
import torch
import torch.nn as nn

from pit.dynamics.kinematic_bicycle import Bicycle
from pit.dynamics.unicycle import Unicycle
from pit.integration import Euler, RK4

from .utils import Method, Curvature, DynamicModel


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Constructor for Neural Network.
        """
        super().__init__()

    @property
    def device(self):
        return next(self.parameters()).device


class ResidualNet(Model):
    def __init__(self, input_dim, output_dim, epsilon=0.1, hidden_dim=32):
        super(ResidualNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.activation = nn.Tanh()
        self.epsilon = epsilon

    def forward(self, X):
        Y = self.linear(X)
        Y = self.activation(Y)
        Y = Y * self.epsilon
        return Y


class LSTMPredictor(Model):
    def __init__(self, input_dim=3, hidden_dim=32, horizon=60, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

        self.hidden2output = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 3 * horizon),
        )

    def forward(self, inputs):
        _, (lstm_out, _) = self.lstm(inputs)
        lstm_out = torch.flatten(lstm_out.permute(1, 0, 2), start_dim=1)
        output = self.hidden2output(lstm_out)
        output = output.reshape((-1, self.horizon, 3))
        return output

    def predict(self, inputs, last_poses, bias_term=None):
        residuals = self.forward(inputs)
        # last_poses = last_poses.to(DEVICE)
        outputs = torch.tile(
            last_poses[:, :3].reshape(last_poses.shape[0], 1, 3), (1, self.horizon, 1)
        )
        outputs = residuals + outputs
        if bias_term is not None:
            outputs += bias_term
        return outputs

class CTRAPredictor(Model):
    def __init__(self, horizon, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        # No real parameters here
        self.horizon = horizon
        self.ts = 0.01
        self.dynamics = Unicycle()
        self.integrator = Euler(self.dynamics, self.ts)
    
    def forward(self, inputs):
        # Dimensions are (batch_size, input_length, N), where the first 4 of N are x, y, theta, velocity.
        # In this method, we take the Unicycle model, and set the yaw rate to the last known yaw rate, and the acceleration to 0.
        # This is a very simple model, requiring no training.
        yaw_rates = (inputs[..., -1, 2] - inputs[..., -2, 2])/self.ts
        accel = (inputs[..., -1, 3] - inputs[..., -2, 3])/self.ts
        inputs = torch.tile(torch.stack([yaw_rates, accel], -1).reshape(-1, 1, 2), [1, 60, 1])
        return inputs

    def predict(self, inputs, last_poses, bias_term=None):
        # Get the yaw rate and velocity
        inputs = self.forward(inputs)
        last_poses = last_poses[..., :4]
        outputs = self.integrator(last_poses, inputs)
        return outputs


class CTRVPredictor(Model):
    def __init__(self, horizon, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        # No real parameters here
        self.horizon = horizon
        self.ts = 0.01
        self.dynamics = Unicycle()
        self.integrator = Euler(self.dynamics, self.ts)
    
    def forward(self, inputs):
        # Dimensions are (batch_size, input_length, N), where the first 4 of N are x, y, theta, velocity.
        # In this method, we take the Unicycle model, and set the yaw rate to the last known yaw rate, and the acceleration to 0.
        # This is a very simple model, requiring no training.
        yaw_rates = (inputs[..., -1, 2] - inputs[..., -2, 2])/self.ts
        inputs = torch.tile(torch.stack([yaw_rates, torch.zeros_like(yaw_rates)], -1).reshape(-1, 1, 2), [1, 60, 1])
        return inputs

    def predict(self, inputs, last_poses, bias_term=None):
        # Get the yaw rate and velocity
        inputs = self.forward(inputs)
        last_poses = last_poses[..., :4]
        outputs = self.integrator(last_poses, inputs)
        return outputs


class LSTMPredictorPCMP(Model):
    def __init__(
        self,
        input_dim=3,
        hidden_dim=32,
        control_outputs=2,
        num_layers=2,
        horizon=100,
        wheelbase=0.3302,
        residual=True,
        model=DynamicModel.BICYCLE,
    ):
        super(LSTMPredictorPCMP, self).__init__()
        self.hidden_dim = hidden_dim
        self.control_outputs = control_outputs
        self.horizon = horizon
        self.residuals_flag = residual

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

        # The linear layer that maps from hidden state space to tag space
        self.hidden2output = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, (horizon * 2) + 0),
        )

        if model is DynamicModel.BICYCLE:
            self.dynamics = Bicycle(wheelbase)
            self.dynamics.wb.requires_grad = False
        elif model is DynamicModel.UNICYCLE:
            self.dynamics = Unicycle()
        self.integrator = Euler(self.dynamics, timestep=0.01)
        if self.residuals_flag:
            self.residual = ResidualNet(horizon * 4, horizon * 2, hidden_dim=hidden_dim)

    def _activation_function(self, raw_control_inputs: torch.Tensor):
        """Raw control inputs in the shape [ACTORS, PREDS, 2]"""
        steer = raw_control_inputs[..., 0]
        speed = raw_control_inputs[..., 1]
        steer = (7 * np.pi / 16) * torch.tanh(steer / 2)
        speed = 20 * torch.tanh(speed / 2)
        control_inputs = torch.cat([steer.unsqueeze(-1), speed.unsqueeze(-1)], -1)
        return control_inputs

    def forward(self, inputs):
        _, (lstm_out, _) = self.lstm(inputs)
        lstm_out = torch.flatten(lstm_out.permute(1, 0, 2), start_dim=1)
        output = self.hidden2output(lstm_out)
        # accel = output[..., 0]
        output = output[..., 0:].reshape((-1, self.horizon, 2))
        scaled_outputs = self._activation_function(output)
        # return accel, scaled_outputs
        return scaled_outputs

    def predict(self, inputs, last_poses, control_input_bias=None, output_bias=None):
        # Compute LSTM output
        # accel, controls = self.forward(inputs)
        controls = self.forward(inputs)
        # last_poses = last_poses.to(DEVICE)
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
            device=self.device,
        )
        state[:, X] = last_poses[:, 0]
        state[:, Y] = last_poses[:, 1]
        state[:, THETA] = last_poses[:, 2]
        state[:, V] = last_poses[:, 3]  # accel

        real_controls = torch.zeros((BATCHES, self.horizon, CDIMS), device=self.device)
        step_length = self.horizon // self.control_outputs
        for i in range(1, self.horizon):
            step = min((i) // step_length, self.control_outputs - 1)
            real_controls[:, i] = controls[:, step]

        if control_input_bias is not None:
            real_controls += control_input_bias

        trace = self.integrator(state, real_controls)
        if self.residuals_flag:
            _, (hd, _) = self.lstm(inputs)
            hd = hd.permute(1, 0, 2)
            resid_input = torch.flatten(
                trace.clone().detach(), 1
            )  # Do not backprop this gradient. This is simply for error correction. We do not want any off-target effects here.
            residuals = self.residual(resid_input)
            residuals = residuals.reshape((BATCHES, self.horizon, 2))
            trace[:, :, :2] = trace[:, :, :2] + residuals
        
        if output_bias is not None:
            trace += output_bias

        return trace, real_controls


def get_model(config_dict):
    """
    Inflates config dictionary to return a model, optimizer, directory pair.
    """
    if config_dict["method"] is Method.PIMP:
        net = LSTMPredictorPCMP(
            input_dim=10 if config_dict["curvature"] is Curvature.CURVATURE else 3,
            hidden_dim=config_dict["hidden_dim"],
            control_outputs=config_dict["control_outputs"],
            horizon=config_dict["horizon"],
            wheelbase=config_dict["wheelbase"],
            residual=config_dict["residual"],
            model=config_dict["model"],
        )
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=config_dict["lr"],
            momentum=config_dict["momentum"],
            weight_decay=config_dict["wd"],
        )
        pytorch_total_params = sum(p.numel() for p in net.parameters())
        eps_per_input = config_dict["eps_per_input"]
        directory = f"{config_dict['prefix']}/PIMP-{config_dict['control_outputs']}-{config_dict['hidden_dim']}-P{pytorch_total_params}-{config_dict['lr']}-{config_dict['model'].name}-{f'CURR-{eps_per_input}' if config_dict['curriculum'] else 'NOCURR'}-WB{config_dict['wheelbase']}-{'RESID' if config_dict['residual'] else 'NORESID'}-{'DISPLOSS' if config_dict['loss_func']=='disp' else 'CUSTOMLOSS'}"
    elif config_dict["method"] is Method.LSTM:
        net = LSTMPredictor(
            input_dim=10 if config_dict["curvature"] is Curvature.CURVATURE else 3,
            hidden_dim=config_dict["hidden_dim"],
            horizon=config_dict['horizon']
        )
        optimizer = torch.optim.Adam(
            net.parameters(), lr=config_dict["lr"], weight_decay=config_dict["wd"]
        )
        pytorch_total_params = sum(p.numel() for p in net.parameters())
        eps_per_input = config_dict["eps_per_input"]
        directory = f"{config_dict['prefix']}/LSTM-{config_dict['curvature'].name}-{config_dict['hidden_dim']}-P{pytorch_total_params}-{f'CURR-{eps_per_input}' if config_dict['curriculum'] else 'NOCURR'}-{'DISPLOSS' if config_dict['loss_func']=='disp' else 'CUSTOMLOSS'}"
    elif config_dict["method"] is Method.CTRV:
        # There's no optimizer for CTRV, no training either
        config_dict["epochs"] = 1
        net = CTRVPredictor(config_dict["horizon"])
        optimizer = None
        pytorch_total_params = sum(p.numel() for p in net.parameters())
        directory = f"{config_dict['prefix']}/CTRV-{config_dict['horizon']}-P{pytorch_total_params}-{config_dict['lr']}"
    elif config_dict["method"] is Method.CTRA:
        # There's no optimizer for CTRA, no training either
        config_dict["epochs"] = 1
        net = CTRAPredictor(config_dict["horizon"])
        optimizer = None
        pytorch_total_params = sum(p.numel() for p in net.parameters())
        directory = f"{config_dict['prefix']}/CTRA-{config_dict['horizon']}-P{pytorch_total_params}-{config_dict['lr']}"
    else:
        raise ValueError(f"Method type {config_dict['method']} is not valid")
    if "job_id" in config_dict:
        directory += f"-{config_dict['job_id']}/"
    config_dict["pytorch_total_params"] = pytorch_total_params
    return net, optimizer, directory, config_dict
