import sys
import torch
import torch_geometric
from torch_geometric.datasets import FakeDataset
from torch_geometric.nn import GraphConv
import torch.nn.functional as F
from execution import runner
import os
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T

criterion = torch.nn.CrossEntropyLoss()
torch_geometric.seed.seed_everything(42)
data = FakeDataset(avg_num_nodes=10000).generate_data()
num_classes = torch.numel(torch.unique(data.y))
h_size = 32
batch_size=1024
print(data)
def optim_func(params) :
    return torch.optim.SGD(params, lr=0.01)

def input_func(steps, dtype, device):
    num_work = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_work = len(os.sched_getaffinity(0)) / 2
        except Exception:
            pass
    if num_work is None:
        num_work = os.cpu_count() / 2
    num_work = int(num_work)
    loader = NeighborLoader(
        data,
        num_neighbors=[50, 50],
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        input_nodes=("v0", None),
        num_workers=num_work,
        replace=True,
        transform=T.ToDevice(device),
    )

    for _ in range(steps):
        data_list.append(next(iter(loader)))

class TestModule(torch.nn.Module) :
    def __init__(self) :
        super(TestModule, self).__init__()
        self.conv1 = GraphConv(data.x.size()[-1], h_size)
        self.conv2 = GraphConv(h_size, num_classes)

    def forward(self, x, edge_index, y):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return [criterion(x[:batch_size], y[:batch_size])]

if __name__ == "__main__" :
    runner.run(sys.argv, 'Homogenous_GNN_Conv_dynamic', TestModule(), optim_func, input_func, None) 
