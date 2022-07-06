import sys
import torch
import torch_geometric
from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.nn import GraphConv, HeteroConv
import torch.nn.functional as F
from execution import runner

criterion = torch.nn.CrossEntropyLoss()
torch_geometric.seed.seed_everything(42)
frozen_data = FakeHeteroDataset(avg_num_nodes=20000).generate_data()
print(frozen_data)
def optim_func(params) :
    return torch.optim.SGD(params, lr=0.01)

def input_func(steps, dtype, device):
    return [frozen_data.to(device) for _ in range(steps)]

class TestModule(torch.nn.Module) :
    def __init__(self) :
        super(TestModule, self).__init__()
        in_feat = {node_type:frozen_data[node_type].x.shape[-1] for node_type in frozen_data.node_types}
        self.conv1 = HeteroConv(
            {
                rel: GraphConv((in_feat[rel[0]], in_feat[rel[-1]]), out_feat)
                for rel in frozen_data.edge_types
            }
        ).jittable()
        self.conv2 = HeteroConv(
            {
                rel: GraphConv((in_feat[rel[0]], in_feat[rel[-1]]), out_feat)
                for rel in frozen_data.edge_types
            }
        ).jittable()

    def forward(self, data):
        x_dict = data.collect('x')
        edge_index_dict = data.collect('edge_index')
        x_dict = F.relu(self.conv1(x_dict, edge_index_dict))
        x_dict = self.conv2(x_dict, edge_index_dict)
        y_dict = data.collect('y') 
        labeled_node_type = y_dict.keys()[0] # should only be one labeled node type
        return [criterion(x_dict[labeled_node_type], y_dict[labeled_node_type])]

if __name__ == "__main__" :
    runner.run(sys.argv, 'Heterogenous_GNN_Conv', TestModule(), optim_func, input_func, None) 

