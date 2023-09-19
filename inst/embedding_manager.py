import torch



class EmbeddingManager(nn.Module):
    def __init__(self,
                 embedder,
                 ):
        super().__init__()

        self.string_to_token_dict = {}
        self.init = True




        self.attention = Attentions()

    def forward(self,
                tokenized_text,
                embedded_text,
                image_embeds,):
        b, n, device = *tokenized_text.shape, tokenized_text.device
        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            placeholder_embedding = self.attention(image_embeds.view(b, 1, 768).to(device),
                                                   image_embeds.view(b, 1, 768).to(device)).view(1,768)

            if self.max_vectors_per_token == 1:
                




class Attentions(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None,
                 gated_ff=True, checkpoint=True):
        super().__init__()
        # SA
        self.attn1 = 

        # CA

        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim))

    def forward(self, x, context=None):
        x_1 = self.attn1(x)
        x_2 = self.attn2(x_1, x)
        x_3 = self.net(x_2)
        return x_3