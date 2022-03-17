+++
title = "Transformer Inference Performance Arithmetic"
date = 2020-04-20
weight = 1
path = "transformer-inference-arithmetic"
+++

This post assumes some prior knowledge about transformers, say at having understood most of [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) but not having internalised all of it. This is entirely focused on decoder-only architectures but can be extrapolated to encoder-decoder or encoder-only architectures.

> Would like to extend extremely large amount of credit to [James Bradbury](https://twitter.com/jekbradbury) for his tremendous help in teaching me about performance concepts and reviewing the post. Also big thanks to Horace He and Mo Bavarian for iterating with me (i prematurely put names here pls halp), and to Jim Wu for teaching me how to write math notation.

### what are the parameters?
Each weight or a parameter is a float that was tuned during training and is usually two bytes as most training is done half-precision now([bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)). Not everything is trained/served bfloat16, but it's at least half-precision (at least since [the GPT-3 Paper](https://arxiv.org/pdf/2005.14165.pdf) in 2020) which gets us the two bytes.

It's useful to break down what these weights are, so that we can understand how the components are used for inferencing later.

The weights loosely consist of the following, per each layer:

 - \\( q,k,v \\) matrices, which are each \\(d_{model} \cdot n_{heads}\cdot d_{head} \\)
 - projector weights, which are also \\(d_{model}\cdot n_{heads}\cdot d_{head} \\) and used right before the MLP layer (the feed-foward neural network that's layered on top of self-attention)
 - MLP weights, which are two matrices of each \\({d_{model}}^2 \cdot 4\\)


> TODO: diagram here?

The four in the MLP weights calculation is based on architecture, but basically every transformer since the [original from 2017](https://arxiv.org/pdf/1706.03762.pdf) has gone with that ratio — where the MLP is 4 four times the size of the model embedding dimension. In a vast majority of transformer architectures, \\(n_{heads}\cdot d_{head} = d_{model}\\). You can see this in [all the GPT models](https://arxiv.org/pdf/2005.14165.pdf) at Table 2.1 (the 13B model is off by 20, but might just be a... typo?), in the [Gopher Models](https://arxiv.org/pdf/2112.11446.pdf) in Table 1 (where what I called \\(d_{head}\\), they called "Key/Value Size"). This is not necessarily the case, but can be assumed.

So then we have a handy equation to calculate the number of parameters!

{% katex(block=true) %}P = 12 \cdot n_{layers} \cdot {d_{model}}^2 {% end %}

With these, we can practice seeing how the factor of four in the MLP layers and the relationship of \\(n_{heads}\cdot d_{head} = d_{model}\\) holds true with the dimensions in the [inaugural Anthropic paper](https://arxiv.org/pdf/2112.00861.pdf) in Table 1, where only \\(n_{layers}\\),  \\(d_{model}\\) and  \\(P\\) are supplied.

{% katex(block=true) %}
P = 12 * n_{layers} \cdot {d_{model}}^2\\
= 12 \cdot 64 \cdot 8192^2\\
= 51,539,607,552
{% end %}

This is not *quite* 52B. It's probably cheating to round up by half a billion parameters, but we can account for them! The equation above is most of the parameters, but we're missing token embeddings. Anthropic uses a 65536 vocab size, so we get \\(n_{tokens} * d_{model} = 536,870,912 \\). Adding \\(536,870,912 + 51,539,607,552 = 52,076,478,464\\). There might be some other paramers like biases for the models, or for Gopher 280B there's 21.5B weights spent on some relative positional encoding method presented in the [Transformer XL paper](https://arxiv.org/abs/1901.02860).


### kv cache
Before sampling execuse, there's a forwards pass, or input encoding step which computes some matrices from the context provided to the model. We'll call this a kv cache, because \\(k, v \\) are the values stored in the cache (for each attention layer). Others may call it a past cache (aka, the open source GPT-2 implementation called it `past`).

The purpose of this, is that it would be inefficient to recalculate those values every time we wanted to generate a new token. With the computed \\(k, v \\) values, we can save quite a bit of computation. Per token, the number of bytes we store is

{% katex(block=true) %} 2\cdot n_{layers} \cdot d_{head} \cdot n_{heads} \cdot 2 {% end %}

Where we have 2 for \\(k\\) and \\(v\\), then we store that per each layer, and each of those values is a \\(d_{head} \times n_{heads}\\) matrix. Then multiply by two again for the number of bytes.

Our weights that we multiply by the token embeddings are \\(W_k, W_v \in \mathbb{R}^{d_{model}\times d_{model}}\\) and then each token embedding is \\(t_e\in \mathbb{R}^{1\times d_{model}}\\). So then the computation to compute \\(k\\) and \\(v\\) for all our layers is

{% katex(block=true) %}
n_{layers} \cdot 2 \cdot {d_{model}}^2\cdot 2
{% end %}

To compute just \\(k\\) we multiply \\(t_e\\) by \\(W_k\\), which takes \\(2 \cdot {d_{model}}^2\\) flops, as the computation for a matrix-vector multiplication is \\(2mn\\) given \\(A \in \mathbb{R}^{m\times n}, b \in \mathbb{R}^{n}\\). We have another factor of two as we do \\(k\\) and \\(v\\) and then a factor of \\(n_{layers}\\).

This means for a 52B parameter model (taking Anthropic's, where \\(d_{model} = 8192\\) and \\(n_{layers} = 64\\)). The flops are;
{% katex(block=true) %}
64 \cdot 2 \cdot 8192^2\cdot 2 = 17,179,869,184
{% end %}

Say we have an A100 GPU, where we do \\(312\text{e}12\\) [flops per second](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf). That means for one token, using a value from the kv cache saves \\(17,179,869,184/312\text{e}12 = 0.00005506368\\) seconds. If say we have 1000 tokens of context, and then want to generate 200 tokens, then we'll have to do that 1000 times to generate the first token, then 1001 for the next token ... then 1200 for the last token. So we do that per-token kv calculation (sum of the consecutive range from 1000 to 1200) times, which is 221100.

So then the total number of seconds is:
{% katex(block=true) %}
221100 \times 0.00005506368 = 12.174579648 \text{seconds}
{% end %}

This is not the whole story (given parallelism or overheads and [tradeoffs](https://twitter.com/pommedeterre33/status/1491314217370398725?s=21) associated with storing this cache). For example, if we're serving small requests we may be memory bandwidth bound rather than flops, in which case saving us flop time may not be that valuable. But twelve is a lot of seconds!


### capacity
Given the number of parameters, and the amount of kv cache we need to store, we should start worrying about how much space is on our accelerators to do that. Now is a good time to put up a table of our accelerators -- for this blog post we'll only work with Nvidia A100 GPUs (which are generally speaking, the best GPUs we can get for inference).

|                          | A100 40GB SXM | A100 80GB SXM |
|--------------------------|---------------|---------------|
| BFLOAT16 Flops           | 312TF         | 312TF         |
| Capacity                 | 40GB          | 80GB          |
| GPU Memory Bandwidth     | 1555GB/s      | 2039GB/s      |
| Communication Bandwidth  | 300GB/s       | 300GB/s       |

For now we only need to pay attention to capacity. The numbers are pulled from [this doc](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf), where the communication bandwidth is marked as 300GB/s instead of 600GB/s because 600GB/s is bidirectional, which we don't use. It's also worth noting that these are all absolute best case scenarios, and in practice this performance should not be expected (though you should be close!)

Given the parameter count, we can multiply by two to get bytes. So to calculate the size of the weights for a 52B model.
{% katex(block=true) %}
52,076,478,464 \cdot 2 = 104,152,956,928 \text{bytes} \approx 104\text{GB}\\
{% end %}

Oh no! This doesn't fit in one GPU! In practice, people can't really get access to the 80GB GPUs yet (see [GCP](https://web.archive.org/web/20220316203021/https://cloud.google.com/compute/docs/gpus) and [AWS](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) GPU offerings). So for the 40GB GPUs, we'd need at least three GPUs just to have all the weights loaded in (will discuss how to use the GPUs together in the next section). That leaves us \\(120-104 = 16GB\\) left for our kv cache. Is that enough? Back to our equation for kv cache cost per token, again with a 52B model;

{% katex(block=true) %}
2\cdot n_{layers} \cdot d_{head} \cdot n_{heads} \cdot 2
 = 2\cdot 64 \cdot 8192 * 2
 = 2,097,152 \text{bytes}
\approx 0.002 GB
{% end %}
And then we'd do \\(16/0.002 \approx 8000\\) tokens can fit into our kv cache with this GPU set up, or that we could do up to a batch size 40 where each item is 200 tokens. For four GPUs, we'd get \\(56/0.002 \approx 23000\\).

> TODO: figure out why this number is so big when cohere ooms at batch size 2 of 2048 tokens, is the math wrong? am i missing factors? is cohere code broken?

But the capacity goes a bit elsewhere. Though we already account for the weights, there are some other intermediate steps that take capacity, but they should be relatively negligible.

Maybe we want more GPUs! If we're going to pay the capacity cost of storing weights, it may be advantageous to attach more GPUs together so that we can do more inferencing at once without paying the cost of storing weights many times. Additionally, larger batches have time savings per-token-generated though it's not perfect and will eventually get flop bound (adding to the batch will be about as fast as waiting for the batch to execute and then running another batch).

### model parallelism
I'm not going to build up full understanding of model parallelism and all the implementation details, because [HuggingFace has done a great job](https://huggingface.co/docs/transformers/parallelism), but here's a description of it that's particularly relevant for inferencing.

We have all these computations we'd like to do. All these matmuls on big matrices are expensive! Our model has limited memory bandwidth, and we have to pass all our weights through to complete a single decoding step! Wouldn't it be nice to insert more GPUs to start dividing that compute cost by an arbitrarily large number (which in practice, is 16 GPUs attached to one machine, but maybe AWS can only do 8 [I haven't found evidence that they offer 16]).

> TODO: diagram here?

The answer, is that it's tricky — how do we split up our weights? We have to pay some latency to communicate between the models — when does it occur, how often and how much?

We will assume Tensor Parallel (also sometimes called model parallel) where we will split down the middle of the model. In pipeline parallel we would split the layers accross GPUs, in tensor parallel, we vertically cut through the layer stack. So that means that some operations one accelerator can run the computations for the shard it has, and the communication doesn't have to occur until a computation needs to be done with all the previous steps.

For our inferencing this happens at;

- three times to account for \\(q, k, v \\) for which \\(n_{layers} \cdot d_{model}\\) is communicated as \\(q,k,v \in \mathbb{R}^{1\times d_{model}} \\).
- once for the output projection, which is the same dimension of \\(p \in \mathbb{R}^{1\times d_{model}} \\).
- eight times to account for the two MLP operations, which is the dimension of \\(w_i, w_o \in \mathbb{R}^{4\times d_{model}} \\).

The full \\(q, k, v \\) set needs to come together to do this step of the attention;

{% katex(block=true) %}
\text{softmax}(\frac{q\cdot k}{\sqrt{d_{head}}}) \cdot v = z
{% end %}

The projection \\(p \\) needs to come together to be passed into the MLP layer which does a multiplication that blows up \\(p\\) into \\(x \in \mathbb{R}^{1\times 4\cdot d_{model}} \\) with one operation of \\(p\cdot w_i\\), where  \\( w_i \in \mathbb{R}^{d_{model}\times4d_{model}}\\). That output \\(x\\) is then multiplied by \\(w_o\\), where \\( w_o \in \mathbb{R}^{4d_{model}\times d_{model}}\\)

> TODO: reason through why there needs to be all these communications for the MLP layers

There are some other optimisations that can go on and nontrivial implementation details that go into tensor parallelism which are not in scope!

### latency calculations
![](../../img/arithmetic_transformers/batchsize.png)
This is a useful little guide for thinking about transformer inference. We've discussed the capacity fairly thoroughly, mapped out comms in the model parallelism section and discussed general compute steps.

For memory bandwidth, the cost we pay is the time it takes to run all our weights through the bandwidth as we do compute on them. If we run our computations slower than we can load the weights, then we are flops bound and the memory bandwidth isn't a factor in latency calculations. If we have a small number of matmuls to do, then maybe we'll be throttled by memory bandwidth. How many matmuls we have is dependent on the batch size.

For comms, it's not about boundedness, but rather when we hit the throughput-limited communication, aka once we are sending enough data fast enough such that it gets throttled by the 300GB/s on an A100. Something tricky about the latency side of this figure is that it's not reported. Once we hit the throughput limitation, we can calculate quickly it can be done, but the latency numbers aren't reported, so the best at least I can do is guess "approximately small" for which some experienced people told me that 10microseconds is a good bet.

Because of the compute factors, to calculate the latency of a single token decoding step we'd have two formulas - one for memory bandwidth bound and another for flops bound.

Equations for a small batch (say 1, so we can drop the batch factor) would be; (where \\(N\\) is the number of accelerators)
{% katex(block=true) %}
\text{compute} = \frac{2 \cdot P}{N \cdot A_{bm}}\\
\text{comms} = n_{layers} \cdot 4 \cdot 10\mu\text{s}
{% end %}
To explain these equations, we have \\(2 \cdot P\\) because we need to pass all the parameters through the memory, and each parameter is two bytes. \\(A_{bm}\\) is the accelerator memory bandwidth, and this cost is split across accelerators. For comms, we have \\(n_{layers} \cdot 4\\) because of \\(q, k, v, p \\) and then the 10microseconds. It leaves out a bunch of factors, but I'm happy throwing them away for simplicity as the latency is guessed anyway (and comms is small relative to compute, which we'll see in a bit).

For large batches (say 512), where \\(B\\) is the batch size;

{% katex(block=true) %}
\text{compute} = B \cdot \frac{2 \cdot P}{N \cdot A_f}\\
\text{comms} = B \cdot  \frac{n_{layers} \cdot 12 \cdot d_{model}}{A_c}
{% end %}

Where \\(A_f\\) is the flops of the accelerator and \\(A_c\\) is the comms bandwidth. We do \\(2\cdot P\\) flops of operations, which can be intuited by the fact that we matmul through all the parameters, and as mentioned earlier, a matrix-vector multiplication is \\(2mn\\) given \\(A \in \mathbb{R}^{m\times n}, b \in \mathbb{R}^{n}\\). There are some other operations such as various softmaxes, multiplying \\(q\cdot k\\) but they take negligible time compared to the matmuls against the weights matrices, which are a factor of \\({d_{model}}^2\\) rather than \\(d_{model}\\) (or say, vocab size). The next section has a step through of the attention and MLP mechanisms and flop counting.

For comms, we see the magic \\(12\\) again, which is our \\(q, k, v, p\\) and MLP weights -- all multiplied by \\(d_{model}\\) as explained in the model parallelism section. Then it's all divided by the comms bandwidth.

As an exercise, we'll calculate this time for a Gopher sized model! We'll run it on 16xA100 40GB SXM. Gopher reports as a 280B parameter model, but we're going to drop 20B as they're positional encoding parameters, for 260B parameters.

{% katex(block=true) %}
d_{model} = 16384\\
n_{layers} = 80\\
A_f = 312\text{TFLOPS}\\
A_c = 300\text{GBps}\\
A_{bm} = 1,555\text{GBps}
{% end %}

For a small batch, for a total of 27 ms per token generated.
{% katex(block=true) %}
\text{compute} = \frac{2 \cdot P}{N \cdot A_{bm}} = \frac{2 \cdot 260\text{e}9}{16\cdot 1.5\text{e}12} \approx 0.0217 \approx 22\text{ms} \\
\text{comms} = n_{layers} * 4 * 10\mu\text{s}= 128\cdot 4  \cdot  10\mu\text=5120\mu\text{s} \approx 5\text{ms}
{% end %}

For a large batch of 512, for a total of 80ms per token generated (per batch, so in the 80ms 512 tokens are generated).
{% katex(block=true) %}
\text{compute} = B \cdot \frac{2 \cdot P}{N \cdot A_f} = 512 \cdot \frac{2 \cdot 260\text{e}9}{16 \cdot 312\text{e}12} \approx 0.053 \approx 53ms\\
\text{comms} = B \cdot \frac{n_{layers} \cdot 12 \cdot d_{model}}{A_c} = 512 \cdot \frac{ 80 \cdot 12 \cdot 16384}{300\text{e}9} \approx 27ms
{% end %}

As an exercise, try calculating the large batch speed for a 52B on 4xGPUs at batch size 2048. The compute should be 85ms and comms should be 53ms.

Also note here, that we can't have the comms be greater than the compute! In these calculations I summed the comms and compute time, but logically there is no reason they can't be partially run in parallel (though it's hard). These numbers still land quite close to what should be acquired in practice, and lean towards being an optimal compute case (optimal hardware usage, we didn't factor in a lot of compute like softmaxes, attention, positional encoding, assumed perfect fusing).

### how many batches is enough?
In the previous section, we have two calculations for when something memory bandwidth bound versus flops bound. To figure out which is at play we can compare these numbers;


### flops counting and compiler optimisations

> TODO: walk through attention + mlp steps to get to the 2P flops and talk about how softmaxes can prolly be fused away. maybe also talk about the flops/communication ratio here?



> TODO: would be very silly if i didn't include a profile