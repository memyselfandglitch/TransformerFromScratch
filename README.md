# KV Cache Inspection With SmolLM2

This project studies how the key-value cache changes during autoregressive text generation with a decoder-only transformer. The experiment uses Hugging Face's `HuggingFaceTB/SmolLM2-135M` model and inspects `past_key_values` during manual generation.

The goal is not to improve text quality. The goal is to understand how the KV cache grows with prompt length, generated tokens, batch size, model depth, KV heads, head dimension, and dtype.

## Model Used

The model used for the main experiment is:

```text
HuggingFaceTB/SmolLM2-135M
```

Important architecture values:

```text
num_layers = 30
num_attention_heads = 9
num_kv_heads = 3
hidden_size = 576
head_dim = hidden_size / num_attention_heads = 64
```

This model uses grouped-query attention: it has 9 query heads but only 3 key/value heads. This reduces KV-cache memory compared with a model where every attention head has its own key/value head.

## What The KV Cache Stores

During generation, a decoder-only transformer repeatedly predicts one new token. Without a KV cache, the model would recompute attention keys and values for all previous tokens at every generation step.

With `use_cache=True`, the model stores previous keys and values in `past_key_values`.

For one transformer layer, the cache shape is typically:

```text
K = (batch_size, num_kv_heads, cached_tokens, head_dim)
V = (batch_size, num_kv_heads, cached_tokens, head_dim)
```

The full cache contains one K tensor and one V tensor for every layer.

## Memory Formula

The total KV-cache memory is:

```text
batch_size * cached_tokens * num_layers * 2 * num_kv_heads * head_dim * bytes_per_value
```

The `2` appears because the model stores both keys and values.

Parameter meanings:

- `batch_size`: number of prompts processed at once.
- `cached_tokens`: prompt tokens plus generated tokens.
- `num_layers`: number of transformer blocks. Each layer has its own KV cache.
- `num_kv_heads`: number of key/value heads. This can be smaller than attention heads in grouped-query attention.
- `head_dim`: dimension of each attention head.
- `bytes_per_value`: memory used by each cached number. `float32` uses 4 bytes; `float16` and `bfloat16` use 2 bytes.

## Observed Batched Run

In one batched experiment, the notebook printed:

```text
manual batch step 20 | cached_tokens=  27 | layer0 K=(9, 3, 27, 64) | layer0 V=(9, 3, 27, 64) | cache=10.68 MiB
```

This means:

```text
batch_size = 9
cached_tokens = 27
num_layers = 30
num_kv_heads = 3
head_dim = 64
bytes_per_value = 4  # float32
```

The cache memory is:

```text
9 * 27 * 30 * 2 * 3 * 64 * 4
= 11,197,440 bytes
= 10.68 MiB
```

The prompt batch had shape:

```text
input_ids shape = (9, 7)
```

So the initial prompt cache had 7 cached tokens. After manually decoding 20 tokens:

```text
cached_tokens = 7 + 20 = 27
```

## Per-Token Growth

For the same batch and model, every additional decoded token adds:

```text
batch_size * num_layers * 2 * num_kv_heads * head_dim * bytes_per_value
```

Plugging in the observed values:

```text
9 * 30 * 2 * 3 * 64 * 4
= 414,720 bytes
= 0.40 MiB per generated token
```

So the cache grows linearly during decoding:

```text
cached_tokens = prompt_tokens + generated_tokens
```

Example layer-0 shapes:

```text
prompt       -> K=(9, 3, 7, 64)
decode step 1 -> K=(9, 3, 8, 64)
decode step 2 -> K=(9, 3, 9, 64)
...
decode step 20 -> K=(9, 3, 27, 64)
```

## What Makes KV Cache Grow Quickly

KV cache increases significantly when any of these increase:

- Long prompts or long context windows.
- Large `max_new_tokens`.
- Larger batch size.
- More transformer layers.
- More KV heads.
- Larger head dimension.
- Higher precision dtype, such as `float32` instead of `float16`.
- Beam search, because multiple candidate continuations are kept alive.

For SmolLM2-135M at `float32`, one token for one batch item costs:

```text
30 * 2 * 3 * 64 * 4 = 46,080 bytes
```

That is about 45 KiB per token per batch item.

With batch size 9:

```text
45 KiB * 9 = about 405 KiB per generated token
```

This matches the measured `0.40 MiB` growth per generated token.

## Notes About Batching

When batching prompts of different lengths for a decoder-only model, padding must be handled carefully.

Use left padding:

```python
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

Then tokenize with:

```python
inputs = tokenizer(
    prompts,
    padding=True,
    return_tensors="pt",
).to(device)
```

The `attention_mask` tells the model which positions are real prompt tokens and which positions are padding.

## Main Takeaway

The KV cache is a memory-speed tradeoff. It increases memory usage because keys and values are stored for every cached token at every layer, but it avoids recomputing attention history during generation.

The cache grows linearly with:

```text
batch size
cached tokens
number of layers
number of KV heads
head dimension
dtype size
```

KV caching changes speed and memory behavior. It does not make the model smarter or improve text quality.

## Real-Model Cache Layout Experiment

The separate `kv_cache_layout_test.ipynb` notebook includes an experimental section that patches the real Hugging Face SmolLM/Llama attention path.

The baseline Hugging Face dynamic cache stores each layer as:

```text
[batch_size, kv_heads, cached_tokens, head_dim]
```

The patched cache stores each layer as:

```text
[batch_size, cached_tokens, kv_heads, head_dim]
```

This is not just a `transpose` view. The custom cache physically stores K/V tensors in the alternate layout from the prompt prefill onward, and the patched attention function reads that layout during manual decoding.

The experiment compares decode latency for:

```text
baseline HF layout: [B, H_kv, L, D]
patched BLHD layout: [B, L, H_kv, D]
```

The memory usage should remain almost the same because the same number of K/V values are stored. Any latency difference comes from layout, strides, tensor operations, and backend behavior.

This is still a Python-level Hugging Face experiment, not a vLLM-style fused-kernel implementation. vLLM gets its speedups by changing cache layout together with custom attention kernels and a paged memory manager.

## Large K/V Tensor Matrix Operation Experiment

The separate `kv_tensor_matvec_benchmark.ipynb` notebook isolates the next question: how large a K or V tensor can fit locally, and how matrix operation latency changes with tensor size and layout.

This notebook does not load a large model. It keeps the KV-cache motivation, then benchmarks simple large tensor operations:

```text
matrix @ vector
matrix @ matrix
scores[b, h, l] = dot(q[b, h, d], K[b, h, l, d])
```

It also compares contiguous `[B, H_kv, L, D]`, non-contiguous transpose views, and contiguous `[B, L, H_kv, D]` storage. This helps separate raw tensor computation effects from Hugging Face model overhead before moving the same investigation to larger models on Colab.

The notebook also includes diagnostic sweeps for arithmetic intensity, tall-skinny versus square-ish matrices, single-token versus multi-token query operations, and batch size.
