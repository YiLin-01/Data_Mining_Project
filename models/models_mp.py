import torch
from torch.nn import functional as F


class IBGNN(torch.nn.Module):
    def __init__(self, gnn, mlp, discriminator=lambda x, y: x @ y.t(), pooling='concat'):
        super(IBGNN, self).__init__()
        self.gnn = gnn
        self.mlp = mlp
        self.pooling = pooling
        self.discriminator = discriminator

    def forward(self, data):
        x, edge_index, edge_attr, batch, edge_flag = data.x, data.edge_index, data.edge_attr, data.batch, data.edge_flag
        # x( 2112, 132 )   2112 个点，132的特征维度
        # tensor([[ 1.9269,  1.4873,  0.9007,  ...,  1.0119, -1.4364, -1.1299],
        #         [-0.1360,  1.6354,  0.6547,  ...,  0.4323, -0.1250,  0.7821],
        #         [-1.5988, -0.1091,  0.7152,  ..., -2.4885, -0.3313,  0.8436],
        #         ...,
        #         [ 1.9073, -1.7992, -0.9697,  ..., -1.8730, -0.0885, -0.0985],
        #         [ 0.1348,  0.3386,  1.6941,  ..., -0.5428, -0.2269, -0.2656],
        #         [-0.1778,  0.5877, -0.4171,  ...,  1.0588,  0.4356,  0.6828]])

        # edge_index ( 2, 43236 )   43236 多条边
        #   tensor([[   0,    0,    0,  ..., 2111, 2111, 2111],
        #            [   1,    2,    4,  ..., 2108, 2109, 2110]])

        # edge_attr (43236)   每条边的 attribute vector 只有一个数字
        #   tensor([1.0718e-01, 3.8745e-03, 3.2074e-01, ..., 1.6995e-04, 1.2791e-03,
        #   2.2336e-01]) why \n here

        # batch: ( 2112, ) tensor([ 0,  0,  0,  ..., 15, 15, 15])

        # edge_flag = ( 278784, )    tensor( [False,  True,  True,  ...,  True,  True, False] )  # 278784 = 2112 * 132


        g = self.gnn(x, edge_index, edge_attr, edge_flag, batch)   # gnn 就是 IBGConv，相当于只过了个 IBGConv  train_batch=1: x: Tensor(132,132), edge_index: (2,1952), edge_attr(1952,), edge_flag:(17424,), 132*132, batch: 132且全=1
        if self.pooling == 'concat':
            _, g = self.mlp(g)
            log_logits = F.log_softmax(g, dim=-1)
            return log_logits
        return g
