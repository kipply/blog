+++
title = "LLM Parameter Counting"
date = 2022-03-30
weight = 1
path = "transformer-param-count"

[extra]
show_toc = true
katex = true
+++

Each weight or parameter is a float that was tuned during training and is usually two bytes as most training is done half-precision now([bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)). Not everything is trained/served bfloat16, but it's at least half-precision (at least since [the GPT-3 Paper](https://arxiv.org/pdf/2005.14165.pdf) in 2020) which gets us the two bytes.

The weights loosely consist of the following, per each block (where one block a decoder unit that consists of a self-attention layer and a feedforward layer, though I'll refer to blocks as layers):

 - \\( W_q,W_k,W_v \\) matrices, which are each \\(d_\text{model} \cdot n_\text{heads}\cdot d_\text{head} \\) and project the input into the query, key, and value used in self-attention.
 - A \\( W_o \\) matrix, which is also \\(d_\text{model}\cdot n_\text{heads}\cdot d_\text{head} \\) and used on the output of self-attention, before the MLP layer (the feed-foward neural network that's stacked on the self-attention layer).
 - MLP weights, which are two matrices each of \\({d_\text{model}}^2 \cdot 4\\). You might also see this referred to by feedforward or linear layers.


The four in the MLP weights calculation is based on architecture, but basically every transformer since the [original from 2017](https://arxiv.org/pdf/1706.03762.pdf) has gone with that ratio — where the MLP is 4 four times the size of the model embedding dimension. In a vast majority of transformer architectures, \\(n_\text{heads}\cdot d_\text{head} = d_\text{model}\\). You can see this in [all the GPT models](https://arxiv.org/pdf/2005.14165.pdf) at Table 2.1 (the 13B model is off by 20, but might just be a... typo?), in the [Gopher models](https://arxiv.org/pdf/2112.11446.pdf) in Table 1 (where what I called \\(d_\text{head}\\), they called "Key/Value Size"). This is not necessarily the case, but can be assumed.

So then we have a handy equation to calculate the number of parameters!

{% katex(block=true) %}P = 12 \cdot n_\text{layers} \cdot {d_\text{model}}^2 {% end %}

With these, we can practice seeing how the factor of four in the MLP layers and the relationship of \\(n_\text{heads}\cdot d_\text{head} = d_\text{model}\\) holds true with the dimensions in the [inaugural Anthropic paper](https://arxiv.org/pdf/2112.00861.pdf) in Table 1, where only \\(n_\text{layers}\\), \\(d_\text{model}\\) and \\(P\\) are supplied.

{% katex(block=true) %}
P = 12 * n_\text{layers} \cdot {d_\text{model}}^2\\
= 12 \cdot 64 \cdot 8192^2\\
= 51,539,607,552
{% end %}

This is not *quite* 52B. It's probably cheating to round up by half a billion parameters, but we can account for them! The equation above is most of the parameters, but we're missing token embeddings. Anthropic uses a 65536 vocab size, so we get \\(n_\text{tokens} * d_\text{model} = 536,870,912 \\). Adding \\(536,870,912 + 51,539,607,552 = 52,076,478,464\\). We actually have that half a billion params twice for the unembeddings, which leads us to about 52.5B tokens.

We're also missing biases that are attached to all the weights, as well as layernorm. Biases should be approximately zero, and layernorm are \\(d_\text{model}\\) (though they exist per block), but otherwise known as zero. Transformers also have positional encoding mechanisms, which for GPT-2 and the original transformer is \\(n_\text{ctx}\cdot d_\text{model}\\) (aka, zero) but Gopher 280B there's 21.5B weights spent on the relative positional encoding method presented in the [Transformer XL paper](https://arxiv.org/abs/1901.02860).
