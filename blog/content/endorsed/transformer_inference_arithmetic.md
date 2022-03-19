+++
title = "Transformer Inference Performance Arithmetic"
date = 2020-04-20
weight = 1
path = "transformer-inference-arithmetic"

[extra]
show_toc = true
+++

This post assumes some prior knowledge about transformers, say at having understood most of [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) but not having internalised all of it. This is entirely focused on decoder-only architectures but can be extrapolated to encoder-decoder or encoder-only architectures. The post is long and verbose, but has many sections of calculations that are trivial to derive or do on your own!

> Would like to extend extremely large amount of credit to [James Bradbury](https://twitter.com/jekbradbury) for his help in teaching me about performance concepts and reviewing the post. Also big thanks to Horace He and Mo Bavarian for iterating with me (i prematurely put names here pls halp), and to Jim Wu for teaching me how to write math notation.

### parameter counting
Each weight or parameter is a float that was tuned during training and is usually two bytes as most training is done half-precision now([bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)). Not everything is trained/served bfloat16, but it's at least half-precision (at least since [the GPT-3 Paper](https://arxiv.org/pdf/2005.14165.pdf) in 2020) which gets us the two bytes.

It's useful to break down what these weights are, so that we can understand how the components are used for inferencing later.

The weights loosely consist of the following, per each block (where one block a decoder unit that consists of a self-attention layer and a feedforward layer, though I'll refer to blocks as layers):

 - \\( W_q,W_k,W_v \\) matrices, which are each \\(d_\text{model} \cdot n_\text{heads}\cdot d_\text{head} \\) and project the input into the query, key, and value used in self-attention.
 - A, \\( W_o \\) matrix, which is also \\(d_\text{model}\cdot n_\text{heads}\cdot d_\text{head} \\) and used on the output of self-attention, before the MLP layer (the feed-foward neural network that's stacked on the self-attention layer).
 - MLP weights, which are two matrices each of \\({d_\text{model}}^2 \cdot 4\\)


> TODO: diagram here?

The four in the MLP weights calculation is based on architecture, but basically every transformer since the [original from 2017](https://arxiv.org/pdf/1706.03762.pdf) has gone with that ratio — where the MLP is 4 four times the size of the model embedding dimension. In a vast majority of transformer architectures, \\(n_\text{heads}\cdot d_\text{head} = d_\text{model}\\). You can see this in [all the GPT models](https://arxiv.org/pdf/2005.14165.pdf) at Table 2.1 (the 13B model is off by 20, but might just be a... typo?), in the [Gopher Models](https://arxiv.org/pdf/2112.11446.pdf) in Table 1 (where what I called \\(d_\text{head}\\), they called "Key/Value Size"). This is not necessarily the case, but can be assumed.

So then we have a handy equation to calculate the number of parameters!

{% katex(block=true) %}P = 12 \cdot n_\text{layers} \cdot {d_\text{model}}^2 {% end %}

With these, we can practice seeing how the factor of four in the MLP layers and the relationship of \\(n_\text{heads}\cdot d_\text{head} = d_\text{model}\\) holds true with the dimensions in the [inaugural Anthropic paper](https://arxiv.org/pdf/2112.00861.pdf) in Table 1, where only \\(n_\text{layers}\\),  \\(d_\text{model}\\) and  \\(P\\) are supplied.

{% katex(block=true) %}
P = 12 * n_\text{layers} \cdot {d_\text{model}}^2\\
= 12 \cdot 64 \cdot 8192^2\\
= 51,539,607,552
{% end %}

This is not *quite* 52B. It's probably cheating to round up by half a billion parameters, but we can account for them! The equation above is most of the parameters, but we're missing token embeddings. Anthropic uses a 65536 vocab size, so we get \\(n_\text{tokens} * d_\text{model} = 536,870,912 \\). Adding \\(536,870,912 + 51,539,607,552 = 52,076,478,464\\). We acutally have that half a billion tokens twice for the unembeddings, which leads us to about 52.5B tokens.

We're also missing biases that are attached to all the weights, as well as layernorm. Biases should be approximately zero, and layer norm is \\(2\cdot d_\text{model}\\), otherwise known as zero. Transformers also have positional encoding mechanisms, which for GPT-2 and the original transformer is \\(n_\text{ctx}\cdot d_\text{model}\\) (aka, zero) but Gopher 280B there's some 20B weights spent on the relative positional encoding method presented in the [Transformer XL paper](https://arxiv.org/abs/1901.02860).

### kv cache
For the cases we are considering, transformer inference consists of processing a provided prompt/context (which can happen in parallel), and then sampling additional tokens one by one. The sampling process needs to refer to the context from the prompt and previously sampled tokens for the key and value components of its self-attention layers. This context is provided in matrices known as the kv cache, aka past cache (the open source GPT-2 implementation called it `past`).

The purpose of this is that it would be inefficient to recalculate those values every time we wanted to generate a new token. With the computed \\(k, v \\) values, we can save quite a bit of computation. Per token, the number of bytes we store is

{% katex(block=true) %} 2\cdot n_\text{layers} \cdot n_\text{heads} \cdot d_\text{head} \cdot 2 {% end %}

Where we have 2 for \\(k\\) and \\(v\\), then we store that per each layer, and each of those values is a \\( n_\text{heads}\times d_\text{head}\\) matrix. Then multiply by two again for the number of bytes.

Our weights that we multiply by the token embeddings are \\(W_k, W_v \in \mathbb{R}^{d_\text{model}\times d_\text{model}}\\) and then each token embedding is \\(t_e\in \mathbb{R}^{1\times d_\text{model}}\\). So then the computation to compute \\(k\\) and \\(v\\) for all our layers is

{% katex(block=true) %}
n_\text{layers} \cdot 2 \cdot {d_\text{model}}^2\cdot 2
{% end %}

To calculate compute for just \\(k\\) we multiply \\(t_e\\) by \\(W_k\\), which takes \\(2 \cdot {d_\text{model}}^2\\) flops, as the computation for a matrix-vector multiplication is \\(2mn\\) given \\(A \in \mathbb{R}^{m\times n}, b \in \mathbb{R}^{n}\\). We have another factor of two as we do \\(k\\) and \\(v\\) and then a factor of \\(n_\text{layers}\\).

This means for a 52B parameter model (taking Anthropic's, where \\(d_\text{model} = 8192\\) and \\(n_\text{layers} = 64\\)). The flops are;
{% katex(block=true) %}
64 \cdot 2 \cdot 8192^2\cdot 2 = 17,179,869,184
{% end %}


We'll need to read all the kv weights once. So that would be \\(2 \cdot 2 \cdot n_\text{layers} \cdot {d_\text{model}}^2\\) bytes. Say we have an [A100 GPU](](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)), where we do \\(312\text{e}12\\) flops per second and \\(1.5\text{e}12\\) bytes per second of memory bandwidth.

{% katex(block=true) %}
\text{memory} = 2 \cdot 2 \cdot n_\text{layers} \cdot {d_\text{model}}^2 \div 1.5\text{e}12\\
\text{compute} = n_\text{layers} \cdot 2 \cdot {d_\text{model}}^2\cdot 2 \div 312\text{e}12\\
{% end %}

None of the model architecture matters anymore — we get a distinct ratio here of 208 given this hardware specification. This means that if we're going to compute kv for one token, it'll take the same amount of time to compute for up to 208 tokens! For 52B this is 11.4 milliseconds (in practice, we'd use four GPUs in parallel so it would actually be under 3 milliseconds, more in following sections). In a following section, we'll learn to calculate that to generate a single token on a 52B on 4 GPUs we need 20ms.

> TODO: diagram about which operations happen to which parts of the inferencing.

This forwards pass is run on any context tokens, which as we've now discovered is very low relative to the decoding steps. In general, inference is a lot slower than processing these forwards pass (which is similar to the operations we do at training time). Calculating for a kv cache token is exactly 1/6th of the compute of doing a decoding step, but is also divided by a large factor (up to 208) for the parallelism.

This is not the whole story (given parallelism or overheads and [tradeoffs](https://twitter.com/pommedeterre33/status/1491314217370398725?s=21) associated with storing this cache). If we're serving small requests we may be memory bandwidth bound rather than flops, in which case we don't want to try saving flop time, rather we want to use that time.

### capacity
Given the number of parameters, and the amount of kv cache we need to store, we should start worrying about how much space is on our accelerators to do that. Now is a good time to put up a table of our accelerators -- for this blog post we'll only work with Nvidia A100 GPUs (which are generally speaking, the best GPUs we can get for inference).

|                          | A100 40GB SXM | A100 80GB SXM |
|--------------------------|---------------|---------------|
| BFLOAT16 Flops           | 312TF         | 312TF         |
| Capacity                 | 40GB          | 80GB          |
| GPU Memory Bandwidth     | 1555GB/s      | 2039GB/s      |
| Communication Bandwidth  | 300GB/s       | 300GB/s       |

For now we only need to pay attention to capacity. The numbers are pulled from [this doc](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf), but we’ve marked the communication bandwidth as 300GB/s instead of 600GB/s because Nvidia is adding up 300 GB/s into each chip and 300 GB/s out of each chip rather than presenting a bidirectional number.

Given the parameter count, we can multiply by two to get bytes. So to calculate the size of the weights for a 52B model.
{% katex(block=true) %}
52,076,478,464 \cdot 2 = 104,152,956,928 \text{bytes} \approx 104\text{GB}\\
{% end %}

Oh no! This doesn't fit in one GPU! In practice, people can't really get access to the 80GB GPUs yet (see [GCP](https://web.archive.org/web/20220316203021/https://cloud.google.com/compute/docs/gpus) and [AWS](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) GPU offerings). So for the 40GB GPUs, we'd need at least three GPUs just to have all the weights loaded in (will discuss how to use the GPUs together in the next section). That leaves us \\(120-104 = 16GB\\) left for our kv cache. Is that enough? Back to our equation for kv cache cost per token, again with a 52B model;

{% katex(block=true) %}
2\cdot n_\text{layers} \cdot n_\text{heads} \cdot d_\text{head}  \cdot 2
 = 2\cdot 64 \cdot 8192 * 2
 = 2,097,152 \text{bytes}
\approx 0.002 GB
{% end %}
And then we'd do \\(16/0.002 \approx 8000\\) tokens can fit into our kv cache with this GPU set up, or that we could do up to a batch size 40 where each item is 200 tokens. For four GPUs, we'd get \\(56/0.002 \approx 23000\\). In practice (I'll say this a lot) we'd definitely go for the four GPUs, since it's painful to divide models made out of powers of two into three.

But the capacity goes a bit elsewhere. Though we already account for the weights, there are some other intermediate steps that take capacity, but they should be relatively negligible.

Maybe we want more GPUs! If we're going to pay the capacity cost of storing weights, we might as well get more processing in. Tradeoffs get tricky here as increasing the batch size isn't free for latency so we'd have to weigh between getting a good speed to finish all our inferencing together and each inference getting completed at the fastest rate possible.

> TODO: can i avoid three paragraphs in a row lmao

At some point, we'll also get closer to flop bound, in that adding another item to the batch won't be faster than sending them in separate batches (though this is hard to get in practice because of capacity, and giving small models too many GPUs causes the communication costs to go up quickly).

### model parallelism
I'm not going to build up full understanding of model parallelism and all the implementation details, because [HuggingFace has done a great job](https://huggingface.co/docs/transformers/parallelism), but here's a description of it that's particularly relevant for inferencing.

We have all these computations we'd like to do. All these matmuls on big matrices are expensive! Our model has limited memory bandwidth, and we have to pass all our weights through to complete a single decoding step! Wouldn't it be nice to insert more GPUs to start dividing that compute cost by an arbitrarily large number (which in practice, is 16 GPUs attached to one machine, but maybe AWS can only do 8 [I haven't found evidence that they offer 16]).

> TODO: diagram here?

The answer, is that it's tricky — how do we split up our weights? We have to pay some latency to communicate between the models — when does it occur, how often and how much?

We will assume Tensor Parallel (also sometimes called model parallel) where we will split down the middle of the model. In pipeline parallel we would split the layers across GPUs, in tensor parallel, we vertically cut through the layer stack. So that means that for some operations one accelerator can run the computations for the shard it has, and the communication doesn't have to occur until a computation needs to be done with all the previous steps.

> TODO: explain why we don't pipeline parallel, doesn't use bw, explain how multiplies can be slpit

For our inferencing this happens at;
- once for our qkv, as we can compute the qkv on respective hosts, then do the qk multiplication + division and then communicate to do the softmax.
- once after computing the output projection, to do a layer norm.
- once after the layer norm
- once after the MLP layer to do the softmax

> TODO: reason through why and why not all the comms

There are some other optimisations that can go on and nontrivial implementation details that go into tensor parallelism which are not in scope!

### latency calculations
![](../../img/arithmetic_transformers/batchsize.png)
This is a useful little guide for thinking about transformer inference. We've discussed the capacity fairly thoroughly, mapped out comms in the model parallelism section and discussed general compute steps.

For memory bandwidth, the cost we pay is the time it takes to run all our weights through the bandwidth as we do compute on them. If we run our computations slower than we can load the weights, then we are flops bound and the memory bandwidth isn't a factor in latency calculations. If we have a small number of multiplies to do per parameter, then maybe we'll be throttled by memory bandwidth. How many flops we spend per parameter is then equal to the batch size.

> TODO: can i avoid three paragraphs in a row lmao

For comms, it's not about boundedness, but rather about adding a latency term and a throughput term (the 300GB/s on an A100). Something tricky about the latency side of this figure is that it's not reported, so the best I can do is guess "approximately small" for which some experienced people told me that 10 microseconds is a good bet.

Because of the compute factors, to calculate the latency of a single token decoding step we'd have two formulas - one for memory bandwidth bound (small batch; also meaning most communication time is latency) and another for flops bound (large batch; also meaning most communication time is throughput).

Equations for a small batch (say 1, so we can drop the batch factor) would be; (where \\(N\\) is the number of accelerators)
{% katex(block=true) %}
\text{compute} = \frac{2 \cdot P}{N \cdot A_\text{bm}}\\
\text{comms} = n_\text{layers} \cdot 4 \cdot 10\mu\text{s}
{% end %}
To explain these equations, we have \\(2 \cdot P\\) because we need to pass all the parameters through the memory, and each parameter is two bytes. \\(A_\text{bm}\\) is the accelerator memory bandwidth, and this cost is split across accelerators. For comms, we have \\(n_\text{layers} \cdot 4\\) and then the 10 microseconds. It leaves out a bunch of factors, but I'm happy throwing them away for simplicity as the latency is guessed anyway (and comms is small relative to compute, which we'll see in a bit).

For large batches (say 512), where \\(B\\) is the batch size;

{% katex(block=true) %}
\text{compute} = B \cdot \frac{2 \cdot P}{N \cdot A_f}\\
\text{comms} = B \cdot  \frac{n_\text{layers} \cdot 4 \cdot d_\text{model}}{A_c}
{% end %}

Where \\(A_f\\) is the flops of the accelerator and \\(A_c\\) is the comms bandwidth. We do \\(2\cdot P\\) flops of operations, which can be intuited by the fact that we matmul through all the parameters, and as mentioned earlier, a matrix-vector multiplication is \\(2mn\\) given \\(A \in \mathbb{R}^{m\times n}, b \in \mathbb{R}^{n}\\). There are some other operations such as various softmaxes, multiplying \\(q\cdot k\\) but they take negligible time compared to the matmuls against the weights matrices, which are a factor of \\({d_\text{model}}^2\\) rather than \\(d_\text{model}\\) (or say, vocab size). The next section has a step through of the attention and MLP mechanisms and flop counting.

For comms, we see the four communication steps per layer all multiplied by \\(d_{model}\\) as explained in the model parallelism section. Then it's all divided by the comms bandwidth.

As an exercise, we'll calculate this time for a Gopher sized model! We'll run it on 16xA100 40GB SXM. Gopher reports as a 280B parameter model, but we're going to drop 20B as they're positional encoding parameters, for 260B parameters.

{% katex(block=true) %}
d_\text{model} = 16384\\
n_\text{layers} = 80\\
A_f = 312\text{TFLOPS}\\
A_c = 300\text{GBps}\\
A_\text{bm} = 1,555\text{GBps}
{% end %}

For a small batch, for a total of 27 ms per token generated.
{% katex(block=true) %}
\text{compute} = \frac{2 \cdot P}{N \cdot A_\text{bm}} = \frac{2 \cdot 260\text{e}9}{16\cdot 1.5\text{e}12} \approx 0.0217 \approx 22\text{ms} \\
\text{comms} = n_\text{layers} * 4 * 10\mu\text{s}= 128\cdot 4  \cdot  10\mu\text=5120\mu\text{s} \approx 5\text{ms}
{% end %}

For a large batch of 512, for a total of 62ms per token generated (per batch, so in the 62ms 512 tokens are generated).
{% katex(block=true) %}
\text{compute} = B \cdot \frac{2 \cdot P}{N \cdot A_f} = 512 \cdot \frac{2 \cdot 260\text{e}9}{16 \cdot 312\text{e}12} \approx 0.053 \approx 53\text{ms}\\
\text{comms} = B \cdot \frac{n_\text{layers} \cdot 4 \cdot d_\text{model}}{A_c} = 512 \cdot \frac{ 80 \cdot 4 \cdot 16384}{300\text{e}9} \approx 9\text{ms}
{% end %}

As an exercise, try calculating the large batch speed for a 52B on 4xGPUs at batch size 256. The compute should be about 21ms and comms should be about 2ms.

Also note here, that we don't want the comms to be greater than the compute! In these calculations I summed the comms and compute time, but logically there is no reason they can't be partially run in parallel (though it's hard). These numbers still land quite close to what should be acquired in practice, and lean towards being an optimal compute case, as it assumes optimal hardware usage and good fusing, plus we didn't factor in a lot of compute like softmaxes, attention and positional encoding. I'd be surprised if someone had an inferencing setup that resulted in numbers lower than what this math comes up with given some core setup details (like int8 would call for different math).

### batch sizes
In the previous section, we have two calculations for when something memory bandwidth bound versus flops bound. To figure out which is at play we can compare these numbers;
{% katex(block=true) %}
\text{mem bandwidth time} = \frac{2 \cdot P}{N \cdot A_\text{bm}}\\
\text{flops time} = B \cdot \frac{2 \cdot P}{N \cdot A_f}
{% end %}

And it becomes obvious why \\(B\\) is an important factor, as the memory bandwidth is not affected by \\(B\\), while flops is. This is about to be the same calculation we did in the [kv cache](#kv-cache) section (where the difference is that the kv cache is 1/6th of the compute and memory usage) where the min batch size for memory bandwidth bound is \\(A_\text{bw}/A_c = 208\\). This is a handy ratio!

To calculate when the capacity goes from mostly kv cache to mostly weights is trivial, and also isn't a binary in the same way (nothing special happens when your kv cache starts taking up more memory than your weights). But what about comms? For comms we want to see that the rate is higher than \\(A_c\\), like so;
> TODO Comms calculations, also talk about the flops/communication ratio, talk about potentially some steps stuck on comms

### flops counting
Previously;
> We do \\(2\cdot P\\) flops of operations, which can be intuited by the fact that we matmul through all the parameters, and as mentioned earlier, a matrix-vector multiplication is \\(2mn\\) given \\(A \in \mathbb{R}^{m\times n}, b \in \mathbb{R}^{n}\\).

This is correct reasoning, but also incomplete. For complete reasoning, the easiest thing to do is to walk through all the transformer steps and check that we get \\(2P\\).

To start, why is a matmul of a matrix-vector 2mn? These [lecture notes](https://www.stat.cmu.edu/~ryantibs/convexopt-F18/scribes/Lecture_19.pdf) explain that fairly thoroughly, and it also makes sense from there that a matrix-matrix multiplication is \\(2mnp\\) if we multiplied \\(A \in \mathbb{R}^{m\times n}, B \in \mathbb{R}^{n \times p}\\). And then a vector-vector multiplication is just \\(2n\\). (The lecture notes are helpful for explaining the factor of 2).

> TODO: insert attention + mlp code

The following calculations are per token, per layer. I describe \\(W_q, W_k, W_v \in \mathbb{R}^{d_\text{model}\times d_\text{model}}\\) where it's more accurate to say we have \\(W_q^i, W_k^i, W_v^i \in \mathbb{R}^{d_\text{model}\times d_\text{head}}\\), where \\(i\\) goes up to \\(n_\text{heads}\\). But for the sake of calculating latency, I simplify \\(W_q, W_k, W_v\\) to include all the heads.

- Computing qkv
    - Let \\(t_e\\) be our token embedding. Then we multiply \\(t_e \in \mathbb{R}^{1\times d_\text{model}}\\) by \\(W_q, W_k, W_v \in \mathbb{R}^{d_\text{model}\times d_\text{model}}\\). We do that multiplication three times for each of \\(q, k, v\\).
    - Flop count: \\({d_\text{model}}^2 \cdot 2 \cdot 3\\)
- Calculate z
    - This is \\(\text{softmax}((q\cdot k)\div\sqrt{d_\text{head}}) \cdot v = z\\)
    - No matrices are multiplied, the number of flops is some factor of \\(d_\text{model}\\).
- Multiply by the output projection matrix
    - We multiply \\(W_o \in \mathbb{R}^{d_\text{model}\times d_\text{model}}\\), by \\(z \in \mathbb{R}^{d_\text{model}\times1}\\).
    - Flop count:  \\(2 \cdot {d_\text{model}}^2\\)
- Feed-forward
    - We have our MLP weights \\(W_1, W_2 \in \mathbb{R}^{4\times d_\text{model}} \\).
    - The MLP is two linear transformations (read: matmul), with a GeLU  in the middle.
    - Then the flops are \\((2\cdot d_\text{model}  + (6\cdot d_\text{model}) + (2\cdot d_\text{model} \cdot 4\cdot d_\text{model})\\) for the linear transform, GeLU and another linear transform.
    - Flop count:  \\(16 \cdot {d_\text{model}}^2 \\)
- Positional encoding
    - The original transformer has a cosine positional encoding scheme, which is an addition to the token embedding. No matrix multiplies to see here!
- Some other things
    - There are typically layernorms that happen after each attention, where the weights there are a vector of length \\(d_\text{model}\\).
    - There's another linear layer and then a softmax that sits on top, which is our output (token) embedding or unembedding or de-embedding or embedding\\(^{-1}\\.


Adding up all the flops!

{% katex(block=true) %}
F = n_\text{layers} \cdot (2 \cdot 3  \cdot {d_\text{model}}^2  + 2\cdot d_\text{model}^2  + 16\cdot d_\text{model}^2 )\\
 = n_\text{layers} \cdot 24 \cdot {d_\text{model}}^2
{% end %}
Subbing in our 8192 model, we should get about 100B flops;

{% katex(block=true) %}
F = 64\cdot(24\cdot 8192^2 + 23 \cdot 8192)
 = 103079215104 \text{flops}
{% end %}

103079215104 over two is about 51.5B. We're a lil under (we get 51.5B instead of 52B) but if we recall from the parameter counting session, there are 51.5B parameters if we exclude the token embeddings and there is just about half a billion of token embeddings given their 65536 vocab size. It would be reasonable to do the latency calculations with \\(2\cdot 12\cdot n_\text{layers} \cdot {d_\text{model}}^2\\) instead of \\(2\cdot P\\), but it's just about a 2% difference.


### leftover latency calculations
What about all the operations I left out? It is really hard to count these operations! The reported flops is specifically for doing matrix multiplies, as GPUs come with specialised hardware for matmuls, so it would be wrong to try to count the FLOPs in. Though mostly, deciding how many FLOPs a square root needs is a high-knowledge and high-precision endeavor.

How do we count memory bandwidth time into the softmax? It wasn't factored into our memory bandwidth calculation and our memory bandwidth:compute ratio is 208, so softmax will always be bounded by memory bandwidth. Softmax is quite a key component of our compute (and it's a little unfortunate that we optimised our hardware to do a lot of matmul, and now we're asking it to do a lot of softmax). I'll cheat on the "arithmetic" theme and pull up this table from [Data Movement is All You Need, 2021](https://proceedings.mlsys.org/paper/2021/file/c9e1074f5b3f9fc8ea15d152add07294-Paper.pdf).

![](../../img/arithmetic_transformers/dataisall.png)


