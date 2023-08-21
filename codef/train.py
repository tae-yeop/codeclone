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