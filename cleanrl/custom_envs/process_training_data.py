
import numpy as np
import matplotlib.pyplot as plt
from utils import get_baseline_path
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Rectangle
map_size = 400
img_size = 80
radar_detection_range = 200
grid_size = 5

def get_radar_heat_map(state, radar_locs):
    radars_encoding = np.zeros((img_size, img_size))
    theta = np.arctan2(state[3], state[1])
    # theta = 0.0
    loc_to_glob = np.array([[np.cos(theta), -np.sin(theta), state[0]],
                            [np.sin(theta), np.cos(theta), state[2]],
                            [0., 0., 1.]])
    glob_to_loc = np.linalg.inv(loc_to_glob)
    print(glob_to_loc)
    for radar_loc in radar_locs:
        if abs(state[0] - radar_loc[0]) < radar_detection_range or abs(state[2] - radar_loc[1]) < radar_detection_range:
            # print("Radar global: ", radar_loc)
            glob_loc_hom = np.array([radar_loc[0], radar_loc[1], 1])
            local_loc_hom = np.dot(glob_to_loc, glob_loc_hom)
            radars_loc_coord = local_loc_hom[:2]
            print('Global: ', radar_loc)
            print("Local: ", radars_loc_coord)
            # print("Radars local coord: ", radars_loc_coord)
            y_grid = np.rint((radars_loc_coord[1]) / grid_size) 
            x_grid = np.rint((radars_loc_coord[0]) / grid_size) 
            print("Grid index: ", [x_grid, y_grid])
            print()
            for i in range(-int(img_size/2), int(img_size/2)):
                for j in range(-int(img_size/2), int(img_size/2)):
                    radars_encoding[int(i + img_size/2), int(j + img_size/2)] += np.exp((-(x_grid - i)**2 - (y_grid - j)**2)/2.0)*1e6

    if np.max(radars_encoding) > 0:
        formatted = (radars_encoding * 255 / np.max(radars_encoding)).astype('uint8')
        # formatted = (radars_encoding / np.max(radars_encoding)).astype('uint8')
    else:
        formatted = radars_encoding.astype('uint8')

    formatted = formatted[np.newaxis, :, :]
    # print("Radar encoding shape ", radars_encoding.shape)
    # print(formatted.shape)
    fig = plt.figure()
    plt.imshow(radars_encoding.T, cmap='hot', interpolation='nearest', origin='lower')
    plt.imsave('heat_map.png', radars_encoding.T, cmap='hot', origin='lower')
    plt.colorbar()
    fig.savefig("radar_heat_map")


    return formatted