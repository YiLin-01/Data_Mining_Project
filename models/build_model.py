import torch
from models.models_common import IBGConv, MLP
from models.models_mp import IBGNN


def build_model(args, device, num_features):
    if args.att.lower() == 'gender':
        model = IBGNN(IBGConv(num_features, args, num_classes=2),
                    MLP(args.hidden_dim, args.hidden_dim, args.n_MLP_layers, torch.nn.ReLU, n_classes=2),   # hidden_dim = 16, n_MLP_layers = 1
                    pooling=args.pooling).to(device)
    elif args.att.lower() == 'race':
        model = IBGNN(IBGConv(num_features, args, num_classes=3),
                    MLP(args.hidden_dim, args.hidden_dim, args.n_MLP_layers, torch.nn.Softmax, n_classes=3),   # hidden_dim = 16, n_MLP_layers = 1
                    pooling=args.pooling).to(device)
    print("model=", model)
    return model
