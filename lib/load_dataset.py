import os
import numpy as np
def load_population_dataset():
    base_dir = os.path.join(os.getcwd(), 'data', 'python')
    path = os.path.join(base_dir, 'population_matrix_time.npy')
    data = np.load(path)
    print(f"[population] shape={data.shape}, dtype={data.dtype}")
    return data
def load_distance_dataset():
    base_dir = os.path.join(os.getcwd(), 'data', 'python')
    path = os.path.join(base_dir, 'distance_matrix_time.npy')
    data = np.load(path)
    print(f"[distance]   shape={data.shape}, dtype={data.dtype}")
    return data
def load_topology_distance():
    base_dir = os.path.join(os.getcwd(), 'data', 'python')
    path = os.path.join(base_dir, 'topology_matrix_time.npy')
    data = np.load(path)
    print(f"[topology]   shape={data.shape}, dtype={data.dtype}")
    return data
def load_traffic_dataset(dataset):
    base_dir = os.path.join(os.getcwd(), 'data', 'python')
    names = [
        'totaltraffic',
        'servicetraffic_video',
        'servicetraffic_IoT',
        'servicetraffic_data'
    ]
    files = [
        'neighbor_traffic_time.npy',
        'video_traffic_ratio.npy',
        'IoT_traffic_ratio.npy',
        'data_traffic_ratio.npy'
    ]
    data = []
    for name, fname in zip(names, files):
        arr = np.load(os.path.join(base_dir, fname))
        data.append(arr)
    return data


