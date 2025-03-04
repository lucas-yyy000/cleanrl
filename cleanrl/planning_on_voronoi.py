import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
from scipy.sparse import csgraph
from scipy.stats import qmc


def get_baseline_path(size, dist_between_radars, num_radar):
    '''
    Sample radars and generate optimal path based on distance.
    '''
    # Sample with Poisson disk
    rng = np.random.default_rng()
    radius = dist_between_radars/size
    engine = qmc.PoissonDisk(d=2, radius=radius, seed=rng)
    sample_tmp = size*engine.random(num_radar)

    # Randomly sample
    # sample_tmp = size*np.random.rand(num_radar, 2)
    sample = []
    # print(sample.shape)
    if sample_tmp.shape[0] != num_radar:
        raise Exception("Not enough radar locations generated. Try to decrease the distance between radars or increase the map size.")
    for i in range(num_radar):
        # if np.linalg.norm(sample_tmp[i]) < 0.5 or np.linalg.norm(sample_tmp[i] - np.array([size, size])) < 0.5:
        #     continue
        sample.append(sample_tmp[i])
        
    sample = np.append(sample, [[0, 0]], axis=0)
    sample = np.append(sample, [[size, size]], axis=0)
    # print(sample.shape)

    voronoi_diagram = Voronoi(sample)

    kd_tree = KDTree(voronoi_diagram.vertices)
    dd, ii = kd_tree.query([[0., 0.], [size, size]], k=1)

    # Number of vertices of the Voronois diagram
    N = voronoi_diagram.vertices.shape[0]
    graph =  np.zeros((N, N))

    for i, j in voronoi_diagram.ridge_vertices:
        if i < 0 or j < 0:
            continue
        graph[i, j] = np.linalg.norm(voronoi_diagram.vertices[i] - voronoi_diagram.vertices[j])
        graph[j, i] = graph[i, j]

    dist_matrix, predecessors = csgraph.dijkstra(graph, directed=False, indices=[ii[0]], return_predecessors=True)
    # print(dist_matrix)
    # print(predecessors)

    # Extract the shortest path.
    path = []
    path.append(ii[1])
    index = ii[1]
    while True:
        index = predecessors[0][index]
        if index < 0:
            break
        path.append(index)
    # print(path)

    # fig = voronoi_plot_2d(voronoi_diagram)
    # plt.plot(voronoi_diagram.vertices[path][:, 0], voronoi_diagram.vertices[path][:, 1], 'ro')
    # for i in range(len(np.array(path))-1):
    #     plt.plot(voronoi_diagram.vertices[path][i:i+2, 0], voronoi_diagram.vertices[path][i:i+2, 1], 'ro-')

    # plt.show()

    return sample, voronoi_diagram, path[::-1]

def get_baseline_path_with_vertices(radar_locs, size):
    sample = radar_locs
    sample = np.append(sample, [[0, 0]], axis=0)
    sample = np.append(sample, [[size, size]], axis=0)
    # print(sample.shape)

    voronoi_diagram = Voronoi(sample)

    kd_tree = KDTree(voronoi_diagram.vertices)
    dd, ii = kd_tree.query([[0., 0.], [size, size]], k=1)

    # Number of vertices of the Voronois diagram
    N = voronoi_diagram.vertices.shape[0]
    graph =  np.zeros((N, N))

    for i, j in voronoi_diagram.ridge_vertices:
        if i < 0 or j < 0:
            continue
        graph[i, j] = np.linalg.norm(voronoi_diagram.vertices[i] - voronoi_diagram.vertices[j])
        graph[j, i] = graph[i, j]

    dist_matrix, predecessors = csgraph.dijkstra(graph, directed=False, indices=[ii[0]], return_predecessors=True)
    # print(dist_matrix)
    # print(predecessors)

    # Extract the shortest path.
    path = []
    path.append(ii[1])
    index = ii[1]
    while True:
        index = predecessors[0][index]
        if index < 0:
            break
        path.append(index)
    # print(path)

    # fig = voronoi_plot_2d(voronoi_diagram)
    # plt.plot(voronoi_diagram.vertices[path][:, 0], voronoi_diagram.vertices[path][:, 1], 'ro')
    # for i in range(len(np.array(path))-1):
    #     plt.plot(voronoi_diagram.vertices[path][i:i+2, 0], voronoi_diagram.vertices[path][i:i+2, 1], 'ro-')

    # plt.show()

    return sample, voronoi_diagram, path[::-1]