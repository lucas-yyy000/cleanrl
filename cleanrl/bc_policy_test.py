import torch
import numpy as np
from radar_maps.env.radar_map_double_integrator import RadarMap_DoubleIntegrator

policy  = torch.load('cleanrl/bc_policy.zip')
print(policy)