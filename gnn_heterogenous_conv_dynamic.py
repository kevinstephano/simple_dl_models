import sys
import torch
import torch_geometric
from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.nn import GraphConv, HeteroConv
import torch.nn.functional as F
from execution import runner

criterion = torch.nn.CrossEntropyLoss()
torch_geometric.seed.seed_everything(42)
dataset = FakeHeteroDataset(avg_num_nodes=20000)
num_channels = dataset.num_channels
data = dataset.generate_data()
labeled_node_type = list(data.collect('y').keys())[0] # should only be one labeled node type
num_classes = torch.numel(torch.unique(data[labeled_node_type].y))
data.labeled_node_type = labeled_node_type
h_size = 32
print(data)
def optim_func(params) :
    return torch.optim.SGD(params, lr=0.01)

def input_func(steps, dtype, device):
    data_list = []
    for _ in range(steps):
        dynamic_data = FakeHeteroDataset(avg_num_nodes=20000)
        dynamic_data.num_channels = num_channels
        dynamic_data = dynamic_data.generate_data()
        dynamic_data.labeled_node_type = labeled_node_type
        data_list.append(dynamic_data)
    return data_list

class TestModule(torch.nn.Module) :
    def __init__(self) :
        super(TestModule, self).__init__()
        in_feat = {node_type:data[node_type].x.shape[-1] for node_type in data.node_types}
        self.conv1 = HeteroConv(
            {
                rel: GraphConv((in_feat[rel[0]], in_feat[rel[-1]]), h_size)
                for rel in data.edge_types
            }
        )
        self.conv2 = HeteroConv(
            {
                rel: GraphConv(h_size, num_classes)
                for rel in data.edge_types
            }
        )

    def forward(self, x_dict, edge_index_dict, y):
        x_dict = (self.conv1(x_dict, edge_index_dict))
        for key in x_dict.keys():
            x_dict[key] = F.relu(x_dict[key])
        x_dict = self.conv2(x_dict, edge_index_dict)
        return [criterion(x_dict[labeled_node_type], y)]

if __name__ == "__main__" :
    runner.run(sys.argv, 'Heterogenous_GNN_Conv_dynamic', TestModule(), optim_func, input_func, None) 

