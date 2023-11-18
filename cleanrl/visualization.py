import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import Voronoi, voronoi_plot_2d
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def func(num, dataSet, dotsTraj, headingTraj):
    dotsTraj.set_data([dataSet[num][0], dataSet[num][2]])
    # headingTraj.set_data([dataSet[num][0], dataSet[num][0] + 10*np.cos(dataSet[num][2])], [dataSet[num][1], dataSet[num][1] + 10*np.sin(dataSet[num][2])])
    return dotsTraj


def visualiza_traj(traj, radar_config, voronoi_diagram, path, save=False):
    fig = voronoi_plot_2d(voronoi_diagram)
    # plt.plot(voronoi_diagram.vertices[path][:, 0], voronoi_diagram.vertices[path][:, 1], 'bo')
    for i in range(len(np.array(path))-1):
        if i == 0:
            plt.plot(voronoi_diagram.vertices[path][i:i+2, 0], voronoi_diagram.vertices[path][i:i+2, 1], 'bo-', label="Shortest Path")
        else:
            plt.plot(voronoi_diagram.vertices[path][i:i+2, 0], voronoi_diagram.vertices[path][i:i+2, 1], 'bo-')

    numDataPoints = len(traj)
    # GET SOME MATPLOTLIB OBJECTS
    dotsTraj = plt.plot(traj[0][0], traj[0][2], 'go', label="Agent")[0] # For scatter plot
    x=np.linspace(0, 1000, numDataPoints)
    radar = plt.plot(radar_config[:, 0], radar_config[:, 1], 'ro', label="Radar")[0]
    # plt.set_xlim([0, 1000])
    # plt.set_ylim([0, 1000])
    # plt.set_xlabel('X')
    # plt.set_ylabel('Y')
    
    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(traj, dotsTraj, None), interval=10)
    plt.legend()
    if save:
        line_ani.save('bc_test.gif')
    # plt.legend()
    plt.show()

def visualiza_traj_no_radar(traj):
    numDataPoints = len(traj)
    # GET SOME MATPLOTLIB OBJECTS
    fig = plt.figure()
    ax = plt.axes()
    dotsTraj = plt.plot(traj[0][0], traj[0][1], 'go')[0]
    # Creating the Animation object
    ax.set_xlim([-150, 150])
    ax.set_ylim([-150, 150])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    line_ani = animation.FuncAnimation(fig, func_simplified, frames=numDataPoints, fargs=(traj, dotsTraj, None), interval=100)
    # line_ani.save('test_ddpg.gif')
    plt.show()