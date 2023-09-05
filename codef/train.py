class Embedding(Module):
    def __init__(self, in_channels, N_freqs, logscale=True, identity=True):
        super().__init__()
        self.N_freqs = N_freqs
        self.annealed = False
        self.identity = identity
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = 


class ImplicitVideoSystem(Module):
    def __init__(self, hparams):

        self.models_to_train = []
        self.embedding_xyz = Embedding(2, 8)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class TranslationField(nn.Module):
    def __init__(self, D=6, W=128, in_channels_w=8, in_channels_xyz=34, skip=[4]):
        super().__init__()
        self.D = D
        self.W = W
        
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz+self.in_channels_w, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz+self.in_channels_w, W)
            else:
                layer = nn.Linear(W, W)
            init_weights(layer)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"warping_field_xyz_encoding_{i+1}", layer)

        out_layer = nn.Linear(W, 2)
        nn.init.zeros_(out_layer.bias)
        nn.init.uniform_(out_layer.weight, -1e-4, 1e-4)
        self.output = nn.Sequential(out_layer)

        
    def forward(self, x):
        """
        
        """