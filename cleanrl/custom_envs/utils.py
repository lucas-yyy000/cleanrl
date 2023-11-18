import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
from scipy.sparse import csgraph
from scipy.stats import qmc

def get_baseline_path(map_size, dist_between_radars, num_radar):
    '''
    Sample radars and generate optimal path based on distance.
    '''
    # Sample with Poisson disk
    rng = np.random.default_rng()
    radius = dist_between_radars/map_size
    engine = qmc.PoissonDisk(d=2, radius=radius, seed=rng)
    sample_tmp = map_size*engine.random(num_radar)

    sample = []
    # print(sample.shape)
    if sample_tmp.shape[0] != num_radar:
        raise Exception("Not enough radar locations generated. Try to decrease the distance between radars or increase the map size.")
    for i in range(num_radar):
        sample.append(sample_tmp[i])
        
    sample = np.append(sample, [[0, 0]], axis=0)
    sample = np.append(sample, [[map_size, map_size]], axis=0)
    # print(sample.shape)

    voronoi_diagram = Voronoi(sample)

    kd_tree = KDTree(voronoi_diagram.vertices)
    dd, ii = kd_tree.query([[0., 0.], [map_size, map_size]], k=1)

    # Number of vertices of the Voronois diagram
    N = voronoi_diagram.vertices.shape[0]
    graph =  np.zeros((N, N))

    for i, j in voronoi_diagram.ridge_vertices:
        if i < 0 or j < 0:
            continue
        graph[i, j] = np.linalg.norm(voronoi_diagram.vertices[i] - voronoi_diagram.vertices[j])
        graph[j, i] = graph[i, j]

    dist_matrix, predecessors = csgraph.dijkstra(graph, directed=False, indices=[ii[0]], return_predecessors=True)


    # Extract the shortest path.
    path = []
    path.append(ii[1])
    index = ii[1]
    while True:
        index = predecessors[0][index]
        if index < 0:
            break
        path.append(index)

    return sample, voronoi_diagram, path[::-1]