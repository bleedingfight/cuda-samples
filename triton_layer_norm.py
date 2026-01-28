import torch
from torch import nn

torch.manual_seed(1)
# NLP Example
batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim)
# Activate module
torch_out = layer_norm(embedding)
# print(embedding.shape, out.shape)
# # Image Example
N, C, H, W = 4, 5, 8, 6
input = torch.randn(N, C, H, W)
# layer_norm = nn.LayerNorm([C, H, W])
layer_norm = nn.LayerNorm([H, W])
output = layer_norm(input)
print(input.shape, output.shape)


def layer_norm_2d(x, dims=10, eps=1e-5):
    axis = x.size().index(dims)
    return (x - (x.mean(axis=axis)[:, None])) / (
        x.var(axis=1, correction=0)[:, None] + eps
    ).sqrt()


def layer_norm_3d(x, dims=[], axis=1e-5):
    mean = (
        x.reshape(x.shape[0], x.shape[1], -1).mean(axis=2).reshape(-1, x.shape[1], 1, 1)
    )
    var = (
        x.reshape(x.shape[0], x.shape[1], -1)
        .var(axis=2, correction=0)
        .reshape(-1, x.shape[1], 1, 1)
    )
    out2 = (x - mean) / ((var + 1e-5) ** 0.5)
    return out2


# output = layer_norm_c(embedding)
x = torch.FloatTensor([[1, 2, 4, 1], [6, 3, 2, 4], [2, 4, 6, 1]])
torch_out = nn.LayerNorm(4)(x)
out = layer_norm_2d(x, 4)
torch.testing.assert_close(out, torch_out)
print(out, torch_out)
out = layer_norm_3d(input, [8, 6])

print(out.shape, output.shape)
torch.testing.assert_close(out, output)
