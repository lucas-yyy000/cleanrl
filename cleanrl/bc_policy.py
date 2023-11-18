import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict
from torch.distributions.normal import Normal
import sys
sys.path.append('../cleanrl')
from radar_maps.env.radar_map_double_integrator import RadarMap_DoubleIntegrator
import planning_on_voronoi
import visualization

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class NatureCNN(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        super().__init__()
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    
class FeatureExtractor(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False
    ):
        super().__init__()
        extractors: Dict[str, nn.Module] = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == 'img':
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
            else:
                print("Linear module.")
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += gym.spaces.utils.flatdim(subspace)

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)

class Agent(nn.Module):
    def __init__(self, env, features_dim=64):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(features_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(features_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, np.prod(env.action_space.shape)), std=0.01)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(env.action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x)

    def get_action_mean(self, x):
        return self.actor_mean(x)
    
import copy
def obs_as_tensor(obs, device):
    if isinstance(obs, np.ndarray):
        return torch.as_tensor(obs, device=device)
    elif isinstance(obs, dict):
        dict_obs = {}
        for (key, _obs) in obs.items():
            if key == 'img':
                dict_obs[key] = torch.as_tensor(_obs, device=device, dtype=torch.float32)
            else:
                dict_obs[key] = torch.as_tensor(_obs, device=device)
        return dict_obs
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")


def maybe_transpose(observation: np.ndarray, observation_space: spaces.Space) -> np.ndarray:

    # Avoid circular import
    from stable_baselines3.common.vec_env import VecTransposeImage

    if not (observation.shape == observation_space.shape or observation.shape[1:] == observation_space.shape):
        # Try to re-order the channels
        transpose_obs = VecTransposeImage.transpose_image(observation)
        if transpose_obs.shape == observation_space.shape or transpose_obs.shape[1:] == observation_space.shape:
            observation = transpose_obs
    return observation

def obs_to_tensor(observation, observation_space, device):
    observation = copy.deepcopy(observation)
    for key, obs in observation.items():
        obs_space = observation_space.spaces[key]
        if key == 'img':
            obs_ = maybe_transpose(obs, obs_space)
        else:
            obs_ = np.array(obs)
        # Add batch dimension if needed
        observation[key] = obs_.reshape((-1, *observation_space[key].shape))

    observation = obs_as_tensor(observation, device)
    # print("Tensor: ", observation)
    return observation


if __name__ == "__main__":
    size_of_map = 1000
    detection_range = 300
    grid_size = 5
    env = RadarMap_DoubleIntegrator(size_of_map, [size_of_map, size_of_map], detection_range, grid_size, dist_between_radars=size_of_map/5.0, num_radars=10)
    encoder = FeatureExtractor(env.observation_space, 64)
    agent = Agent(env, features_dim=68)
    # print(policy)
    # for name, param in policy.named_parameters():
    #         print(name)

    # print()

    # for name, param in agent.named_parameters():
    #         print(name)
    # print()
    # policy_params = policy.state_dict()
    # with torch.no_grad():
    #     for name, param in encoder.extractors['img'].named_parameters():
    #         param.data.copy_(policy_params['features_extractor.extractors.img.' + name])
    #         print(name)
    #     #     print(policy_params['features_extractor.extractors.img.' + name])
    #     #     print(param)

    #     for name, param in agent.actor_mean_mlp_extratcor.named_parameters():
    #         param.data.copy_(policy_params['mlp_extractor.policy_net.' + name])
    #         print(name)
    #     #     print(name, param.data)
    #     #     print('mlp_extractor.policy_net.' + name, policy_params['mlp_extractor.policy_net.' + name])

    #     for name, param in agent.action_net.named_parameters():
    #         param.data.copy_(policy_params['action_net.' + name])
    #         print(name)
            # print(name, param.data)
            # print('action_net.' + name, policy_params['action_net.' + name])

    obs, _ = env.reset()
    radar_config = env.radar_locs
    print("Initial state: ", obs['state'])
    trajectory = []
    trajectory.append(obs['state'])
    # print(trajectory)
    for i in range(1500):
        # print("Arry: ", obs)
        action = agent.get_action(encoder.forward(obs_to_tensor(obs, env.observation_space, device='cpu')))
        action = action.cpu().detach().numpy()
        # print(action[0])
        obs, reward, done, _, _ = env.step(action[0])
        # vec_env.render()
        # print("Action: ", action)
        trajectory.append(env.state['state'])
        if done:
            # print("Action: ", action)
            print("State: ", env.state['state'])
            break

    radar_locs, voronoi_diagram, path = planning_on_voronoi.get_baseline_path_with_vertices(radar_config, size_of_map)
    visualization.visualiza_traj(trajectory, radar_locs, voronoi_diagram, path, save=True)