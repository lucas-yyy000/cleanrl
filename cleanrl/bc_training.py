import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)
import numpy as np
import pickle
import gymnasium as gym
from gymnasium import spaces

from imitation.algorithms import bc
from imitation.data.types import Trajectory, DictObs
from imitation.util.util import save_policy

from stable_baselines3.common import policies
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from radar_maps.env.radar_map_double_integrator import RadarMap_DoubleIntegrator


data_path = "/home/lucas/workspace/cleanrl/cleanrl/data_multimodal/"
data_num = 1
num_mode = 3


def get_path_chunk(path, num_points=3):
    len_path = len(path)
    if len_path >= num_points:
        return list(np.hstack(path[:num_points]))
    
    path_patched = list(np.hstack(path))
    for _ in range(num_points-len_path):
        path_patched.extend(list(path[-1]))
    # print("Patched path: ", path_patched)
    return path_patched


def get_radar_heat_map(state, radar_locs, img_size, radar_detection_range, grid_size):
    radars_encoding = np.zeros((img_size, img_size))
    theta = np.arctan2(state[3], state[1])
    # theta = 0.0
    loc_to_glob = np.array([[np.cos(theta), -np.sin(theta), state[0]],
                            [np.sin(theta), np.cos(theta), state[2]],
                            [0., 0., 1.]])
    glob_to_loc = np.linalg.inv(loc_to_glob)
    # print(glob_to_loc)
    for radar_loc in radar_locs:
        if abs(state[0] - radar_loc[0]) < radar_detection_range or abs(state[2] - radar_loc[1]) < radar_detection_range:
            # print("Radar global: ", radar_loc)
            glob_loc_hom = np.array([radar_loc[0], radar_loc[1], 1])
            local_loc_hom = np.dot(glob_to_loc, glob_loc_hom)
            radars_loc_coord = local_loc_hom[:2]
            # print('Global: ', radar_loc)
            # print("Local: ", radars_loc_coord)
            # print("Radars local coord: ", radars_loc_coord)
            y_grid = np.rint((radars_loc_coord[1]) / grid_size) 
            x_grid = np.rint((radars_loc_coord[0]) / grid_size) 
            # print("Grid index: ", [x_grid, y_grid])
            # print()
            for i in range(-int(img_size/2), int(img_size/2)):
                for j in range(-int(img_size/2), int(img_size/2)):
                    radars_encoding[int(i + img_size/2), int(j + img_size/2)] += np.exp((-(x_grid - i)**2 - (y_grid - j)**2)/2.0)*1e6

    plt.imsave('heat_map.jpg', radars_encoding.T, cmap='hot', origin='lower')
    heat_map_img = plt.imread('heat_map.jpg')

    return heat_map_img

def generate_training_data(traj, ctr, path_mm, radars, detection_range, grid_size, v_lim, u_lim):
    observations = []
    actions = []
    for i in range(len(traj)):
        x_cur = traj[i]

        heat_map_img = get_radar_heat_map(x_cur, radars, 2*int(detection_range/grid_size), detection_range, grid_size)
        # print(heat_map_img.shape)
        x_cur_normalized = np.array([x_cur[0]/1200.0, x_cur[1]/v_lim, x_cur[2]/1200.0, x_cur[3]/v_lim])

        observation = DictObs({"state": x_cur_normalized, "img": heat_map_img})
        observations.append(observation)
        if i < len(traj) - 1:
            action_prediction = []
            for m in range(num_mode):
                action_prediction.extend(ctr[i, 2*m:2*(m+1)]/u_lim)
                path_tmp = path_mm[num_mode*i + m]
                path_tmp = [x / 1200.0 for x in path_tmp]
                # print(path_tmp)
                action_prediction.extend(get_path_chunk(path_tmp))
            actions.append(action_prediction)
    
    return np.array(observations), np.array(actions)

def process_data(detection_range, grid_size, v_lim, u_lim):
    bc_data = []
    for i in range(data_num):
        print("Processing data: ", i)
        traj = np.load(data_path + f'state_traj_{i}.npy')
        control = np.load(data_path + f'control_traj_{i}.npy')
        radar_config = np.load(data_path + f'radar_config_{i}.npy')

        with open(data_path+ f'nominal_path_multimodal_{i}.pkl', 'rb') as handle:
            path_mm = pickle.load(handle)

        obs, acts = generate_training_data(traj, control, path_mm, radar_config, detection_range, grid_size, v_lim, u_lim)
        bc_traj = Trajectory(obs, acts, infos=None, terminal=True)
        bc_data.append(bc_traj)
    return bc_data

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
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[-1]
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

class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_output_dim: int = 64,
        normalized_image: bool = False
    ):
        super(FeatureExtractor, self).__init__(observation_space, cnn_output_dim)
        extractors: Dict[str, nn.Module] = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == 'img':
                print("CNN")
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
            else:
                print("Linear module.")
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += gym.spaces.utils.flatdim(subspace)

        self.extractors = nn.ModuleDict(extractors)
        self.features_dim = total_concat_size

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)
    
class BCAgent(nn.Module):
    def __init__(self, action_dim, features_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(features_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(features_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action = self.actor_mean(x)
        return action, self.critic(x)
    

def bc_train():
    rng = np.random.default_rng(0)
    size_of_map = 1000
    detection_range = 300
    grid_size = 5

    v_lim = 20.0
    u_lim = 2.0
    env = RadarMap_DoubleIntegrator(size_of_map, [size_of_map, size_of_map], detection_range, grid_size, dist_between_radars=size_of_map/5.0, num_radars=10)

    transitions = process_data(detection_range, grid_size, v_lim, u_lim)
    # print(type(transitions))
    
    policy_ac = policies.MultiInputActorCriticPolicy(observation_space=env.observation_space,
                                                    action_space=env.action_space,
                                                    lr_schedule=lambda _: 1e-3,
                                                     net_arch=[dict(pi=[64, 64],
                                                          vf=[64, 64])],
                                                    activation_fn=torch.nn.ReLU,
                                                    features_extractor_class=FeatureExtractor)
    bc_trainer = bc.BC(
        rng = rng,
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        policy=policy_ac,
        device='cuda'
    )
    
    bc_trainer.train(n_epochs=50)
    save_policy(bc_trainer.policy, 'bc_policy.zip')

if __name__ == "__main__":
    bc_train()