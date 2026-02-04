import numpy as np
import torch.utils.data
import torch
from torch.utils.data import Dataset
from lib.load_dataset import load_traffic_dataset,load_distance_dataset,load_topology_distance,load_population_dataset
from lib.normalization_traffic import MinMax01Scaler, Max01Scaler
from lib.norm_feature import rowwise_minmax_normalize,minmax_norm_po_local,minmax_norm_po
from lib.add_windows import Add_Window_Horizon
def split_data_by_ratio(*data, val_ratio, test_ratio):
    data_len = data[0].shape[0]
    test_start = int(data_len * (1 - test_ratio))
    val_start = int(data_len * (1 - test_ratio - val_ratio))
    train_data = tuple(arr[:val_start] for arr in data)
    val_data = tuple(arr[val_start:test_start] for arr in data)
    test_data = tuple(arr[test_start:] for arr in data)
    return train_data, val_data, test_data

def normalize_dataset(data, normalizer):
    if normalizer == 'minmax01':
        minimum = data.min(axis=0, keepdims=True)
        maximum = data.max(axis=0, keepdims=True)
        scaler = MinMax01Scaler(minimum, maximum)
        norm_data = scaler.transform(data)
    elif normalizer == 'max01':
        maximum = data.max(axis=(1,2),keepdims=True)#
        scaler = Max01Scaler(maximum)
        norm_data = scaler.transform(data)
    else:
        raise ValueError
    return norm_data,scaler


class MultiInputDataset(Dataset):
    def __init__(self,
                 x_traffic, y_traffic,
                 x_service1, y_service1,
                 x_service2, y_service2,
                 x_service3, y_service3,
                 x_distance,
                 x_population,
                 x_topology):
        self.x_traffic = torch.tensor(x_traffic, dtype=torch.float32)
        self.y_traffic = torch.tensor(y_traffic, dtype=torch.float32)

        self.x_service1 = torch.tensor(x_service1, dtype=torch.float32)
        self.y_service1 = torch.tensor(y_service1, dtype=torch.float32)

        self.x_service2 = torch.tensor(x_service2, dtype=torch.float32)
        self.y_service2 = torch.tensor(y_service2, dtype=torch.float32)

        self.x_service3 = torch.tensor(x_service3, dtype=torch.float32)
        self.y_service3 = torch.tensor(y_service3, dtype=torch.float32)

        self.x_distance = torch.tensor(x_distance, dtype=torch.float32)
        self.x_population = torch.tensor(x_population, dtype=torch.float32)
        self.x_topology = torch.tensor(x_topology, dtype=torch.float32)

        assert len(self.x_traffic) == len(self.x_distance)

    def __len__(self):
        return len(self.x_traffic)

    def __getitem__(self, idx):
        return {
            'traffic_x': self.x_traffic[idx],
            'traffic_y': self.y_traffic[idx],

            'service1_x': self.x_service1[idx],
            'service1_y': self.y_service1[idx],

            'service2_x': self.x_service2[idx],
            'service2_y': self.y_service2[idx],

            'service3_x': self.x_service3[idx],
            'service3_y': self.y_service3[idx],

            'distance_x': self.x_distance[idx],
            'population_x': self.x_population[idx],
            'topology_x': self.x_topology[idx],
        }

def get_first_n_timesteps(data, n: int):
    if isinstance(data, list):
        return [get_first_n_timesteps(arr, n) for arr in data]
    if not isinstance(data, np.ndarray):
        raise ValueError(f"data 必须�?np.ndarray �?list，当前类�?{type(data)}")
    if data.ndim < 1:
        raise ValueError(f"data 必须至少是一维时序数据，当前 ndim={data.ndim}")
    T = data.shape[0]
    if n > T:
        raise ValueError(f"n={n} 超出 data �?维长�?T={T}")
    return data[:n, ...]


def get_dataloader(args):
    data = load_traffic_dataset('all_traffic')
    data = get_first_n_timesteps(data, args.ntime)
    traffic_names = [
        'totaltraffic',
        'servicetraffic_video',
        'servicetraffic_IoT',
        'servicetraffic_data'
    ]
    print("=== Traffic datasets after slicing to n={} ===".format(args.ntime))
    for name, arr in zip(traffic_names, data):
        print(f"  {name:24s} sliced shape: {arr.shape}")
    distance_matrix_time = load_distance_dataset()
    distance_matrix_time = get_first_n_timesteps(distance_matrix_time, args.ntime)
    print(f"\nSliced distance_matrix_time shape:   {distance_matrix_time.shape}")

    population_vector_time = load_population_dataset()
    population_vector_time = get_first_n_timesteps(population_vector_time, args.ntime)
    print(f"Sliced population_vector_time shape: {population_vector_time.shape}")

    topology_matrix_time = load_topology_distance()
    topology_matrix_time = get_first_n_timesteps(topology_matrix_time, args.ntime)
    print(f"Sliced topology_matrix_time shape:   {topology_matrix_time.shape}")

    data[0],scaler=normalize_dataset(data[0],'max01')
    norm_distance_matrix_time=rowwise_minmax_normalize(distance_matrix_time)
    norm_population_vector_time=minmax_norm_po(population_vector_time)
    x_traffic,  y_traffic  = Add_Window_Horizon(data[0], window=args.seq_len, horizon=args.horizon, single=args.single)
    x_service1, y_service1 = Add_Window_Horizon(data[1], window=args.seq_len, horizon=args.horizon, single=args.single)
    x_service2, y_service2 = Add_Window_Horizon(data[2], window=args.seq_len, horizon=args.horizon, single=args.single)
    x_service3, y_service3 = Add_Window_Horizon(data[3], window=args.seq_len, horizon=args.horizon, single=args.single)
    x_distance, _ = Add_Window_Horizon(norm_distance_matrix_time, window=args.seq_len, horizon=args.horizon, single=args.single)
    x_population, _ = Add_Window_Horizon(norm_population_vector_time, window=args.seq_len, horizon=args.horizon, single=args.single)
    x_topology, _ = Add_Window_Horizon(topology_matrix_time, window=args.seq_len, horizon=args.horizon, single=args.single)
    (train_x_traffic, train_y_traffic,
     train_x_service1, train_y_service1,
     train_x_service2, train_y_service2,
     train_x_service3, train_y_service3,
     train_x_distance,
     train_x_population,
     train_x_topology), \
        (val_x_traffic, val_y_traffic,
         val_x_service1, val_y_service1,
         val_x_service2, val_y_service2,
         val_x_service3, val_y_service3,
         val_x_distance,
         val_x_population,
         val_x_topology), \
        (test_x_traffic, test_y_traffic,
         test_x_service1, test_y_service1,
         test_x_service2, test_y_service2,
         test_x_service3, test_y_service3,
         test_x_distance,
         test_x_population,
         test_x_topology) = split_data_by_ratio(
        x_traffic, y_traffic,
        x_service1, y_service1,
        x_service2, y_service2,
        x_service3, y_service3,
        x_distance,
        x_population,
        x_topology,
        test_ratio=args.val_ratio,
        val_ratio=args.test_ratio
    )
    train_dataset = MultiInputDataset(
        train_x_traffic, train_y_traffic,
        train_x_service1, train_y_service1,
        train_x_service2, train_y_service2,
        train_x_service3, train_y_service3,
        train_x_distance,
        train_x_population,
        train_x_topology
    )

    val_dataset = MultiInputDataset(
        val_x_traffic, val_y_traffic,
        val_x_service1, val_y_service1,
        val_x_service2, val_y_service2,
        val_x_service3, val_y_service3,
        val_x_distance,
        val_x_population,
        val_x_topology
    )

    test_dataset = MultiInputDataset(
        test_x_traffic, test_y_traffic,
        test_x_service1, test_y_service1,
        test_x_service2, test_y_service2,
        test_x_service3, test_y_service3,
        test_x_distance,
        test_x_population,
        test_x_topology
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, scaler
if __name__ == '__main__':
    get_dataloader()
