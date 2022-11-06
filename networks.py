import copy
from dataclasses import dataclass

import torch
from torch import nn

import autograd_hacks


class Mlp(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list,
        mid_act: nn.Module = nn.Tanh,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.mid_act = mid_act
        self.layer_norm = layer_norm
        self.layers = self.make_layers()

    def make_layers(self):
        input_layer = []
        input_layer.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        if self.layer_norm:
            input_layer.append(nn.LayerNorm(self.hidden_dims[0]))
        input_layer.append(self.mid_act())

        hidden_layers = []
        for i in range(len(self.hidden_dims) - 1):
            h1, h2 = self.hidden_dims[i], self.hidden_dims[i + 1]
            hidden_layers.append(nn.Linear(h1, h2))
            if self.layer_norm:
                hidden_layers.append(nn.LayerNorm(h2))
            hidden_layers.append(self.mid_act())

        output_layer = [nn.Linear(self.hidden_dims[-1], self.output_dim)]

        layers = input_layer + hidden_layers + output_layer
        layers = nn.Sequential(*layers)
        return layers

    def forward(self, x):
        return self.layers(x)


class MlpActor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        mid_act: nn.Module = nn.ReLU,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.mid_act = mid_act
        self.layer_norm = layer_norm
        self.mid_pos = (action_low + action_high) / 2
        self.scale_factor = (action_high - action_low) / 2
        self.BackBone = Mlp(
            input_dim,
            output_dim,
            hidden_dims,
            mid_act=self.mid_act,
            layer_norm=self.layer_norm,
        )

    def forward(self, x):
        x_device = x.device
        self.scale_factor = self.scale_factor.to(x_device)
        self.mid_pos = self.mid_pos.to(x_device)
        h = self.BackBone(x)
        return torch.tanh(h) * self.scale_factor + self.mid_pos


class MlpCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list,
        mid_act: nn.Module = nn.ReLU,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.mid_act = mid_act
        self.layer_norm = layer_norm
        self.BackBone = Mlp(
            input_dim,
            output_dim,
            hidden_dims,
            mid_act=self.mid_act,
            layer_norm=self.layer_norm,
        )

    def forward(self, x):
        return self.BackBone(x)
    
@dataclass
class ActorPack:
    actor: torch.nn.Module = None
    target_actor: torch.nn.Module = None
    optimizer: torch.optim.Optimizer = None


@dataclass
class CriticPack:
    critic1: torch.nn.Module = None
    critic2: torch.nn.Module = None
    target_critic1: torch.nn.Module = None
    target_critic2: torch.nn.Module = None
    optimizer: torch.optim.Optimizer = None


def target_network_update(source_net, target_net, gamma: float):
    with torch.no_grad():
        for src_param, tar_param in zip(
            source_net.parameters(), target_net.parameters()
        ):
            tar_param.data.copy_(
                gamma * src_param.data + (1.0 - gamma) * tar_param.data
            )


def calc_model_param_sensitivity(data_batch, model_ins):
    inference_net = copy.deepcopy(model_ins)
    autograd_hacks.add_hooks(inference_net)

    inf_res = inference_net(data_batch)
    B, N = inf_res.shape
    grad_ps = []
    for i in range(N):
        inference_net.zero_grad()
        grad_out = torch.zeros(size=[B, N]).to(data_batch.device)
        grad_out[:, i] = 0.1
        inf_res.backward(grad_out, retain_graph=True)
        autograd_hacks.compute_grad1(inference_net)
        grad_net = copy.deepcopy(model_ins)
        with torch.no_grad():
            for ip, gp in zip(inference_net.parameters(), grad_net.parameters()):
                gp.copy_(torch.sum(torch.abs(ip.grad1), dim=0))
        g_p = net_to_vec(grad_net)
        grad_ps.append(g_p ** 2)
        autograd_hacks.clear_backprops(inference_net)
    grad_P = torch.stack(grad_ps)
    sensitivity = torch.sqrt(torch.sum(grad_P, dim=0))
    return {"inf_res": inf_res, "sens": sensitivity}


def net_to_vec(network_ins: nn.Module) -> torch.Tensor:
    # * Convert a network's parameters to a flat vector
    return nn.utils.parameters_to_vector(network_ins.parameters())


def net_from_vec(net_template: nn.Module, param_vec: torch.Tensor) -> nn.Module:
    # * Replace a network's parameters from a flat vector
    out_net = copy.deepcopy(net_template)
    nn.utils.vector_to_parameters(param_vec, out_net.parameters())
    return out_net


def sample_behav_net(target_net, r, sens, z):
    target_vec = net_to_vec(target_net)
    with torch.no_grad():
        applied_noise = z * r / (sens + 1)
        behav_vec = target_vec + applied_noise
        behav_net = net_from_vec(target_net, behav_vec)
    return behav_net
