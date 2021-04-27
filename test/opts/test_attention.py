# import torch
# from ocrstack.opts.attention import AdditiveAttention, MultiHeadAttention, ScaleDotProductAttention


# def test_additive_attention():
#     B, T, E = 2, 5, 128
#     S = 10
#     attention = AdditiveAttention(128)
#     queries = torch.rand(B, T, E)
#     keys = torch.rand(B, S, E)
#     values = torch.rand(B, S, E)
#     outputs, weights = attention.forward(queries, keys, values, output_weights=False)
#     assert outputs.shape == torch.Size([B, T, E])
#     assert weights is None

#     outputs, weights = attention.forward(queries, keys, values, output_weights=True)
#     assert outputs.shape == torch.Size([B, T, E])
#     assert weights.shape == torch.Size([B, T, S])

#     attention = MultiHeadAttention(attention, 2)
#     outputs, weights = attention.forward(queries, keys, values, output_weights=False)
#     assert outputs.shape == torch.Size([B, T, E])
#     assert weights is None

#     outputs, weights = attention.forward(queries, keys, values, output_weights=True)
#     assert outputs.shape == torch.Size([B, T, E])
#     assert weights.shape == torch.Size([B, T, S])


# def test_scale_dot_product_attention():
#     B, T, E = 2, 5, 128
#     S = 10
#     attention = ScaleDotProductAttention(128)
#     queries = torch.rand(B, T, E)
#     keys = torch.rand(B, S, E)
#     values = torch.rand(B, S, E)
#     outputs, weights = attention.forward(queries, keys, values, output_weights=False)
#     assert outputs.shape == torch.Size([B, T, E])
#     assert weights is None

#     outputs, weights = attention.forward(queries, keys, values, output_weights=True)
#     assert outputs.shape == torch.Size([B, T, E])
#     assert weights.shape == torch.Size([B, T, S])

#     attention = MultiHeadAttention(attention, 2)
#     outputs, weights = attention.forward(queries, keys, values, output_weights=False)
#     assert outputs.shape == torch.Size([B, T, E])
#     assert weights is None

#     outputs, weights = attention.forward(queries, keys, values, output_weights=True)
#     assert outputs.shape == torch.Size([B, T, E])
#     assert weights.shape == torch.Size([B, T, S])
