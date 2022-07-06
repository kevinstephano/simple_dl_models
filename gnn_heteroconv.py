import sys
import torch
from torch_geometric.datasets import FakeDataset
from torch_geometric.nn import GraphConv
import torch.nn.functional as F
from execution import runner

def optim_func(params) :
    return torch.optim.SGD(params, lr=0.01)

def input_func(steps, dtype, device) :
    return FakeDataset(avg_num_nodes=20000).generate_data() 

class TestModule(torch.nn.Module) :
    def __init__(self) :
        super(TestModule, self).__init__()
        self.conv1 = GraphConv(data.x.size()[-1], 16).jittable()
        self.conv2 = GraphConv(16, torch.numel(torch.unique(data.y))).jittable()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

if __name__ == "__main__" :
    runner.run(sys.argv, 'Heterogenous_GNN_Conv', TestModule(), optim_func, input_func, None) 
