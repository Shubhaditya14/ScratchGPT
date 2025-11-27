# ScratchGPT
This project was built to understand how GPT-2 works internally by implementing it from scratch. These are the key concepts learned:

🔤 Tokenization (GPT-2 BPE)

Text is converted to subword tokens using Byte Pair Encoding.

Token IDs → embeddings → model.

Outputs are token IDs converted back to text.

🎯 Next-Token Prediction (x vs y)

Input sequence x.

Target sequence y = x shifted by 1.

Model learns: predict the next token at every step.

🔢 Embeddings

Token embeddings + positional embeddings.

Combine to form the input to the transformer.

🧠 Self-Attention

Compute Q, K, V from embeddings.

Attention scores = Q · Kᵀ / sqrt(d).

Softmax → weights → weighted sum of V.

Allows model to focus on relevant previous tokens.

🧩 Multi-Head Attention

Several attention heads in parallel.

Concatenate head outputs → linear projection.

Helps model learn different relationships at once.

🏗 Transformer Block

Multi-head attention

Feed-forward network (MLP)

Residual connections

LayerNorm

Stacked to build the full model.

🔥 Causal Masking

Ensures each token can only attend to previous tokens.

Enforces autoregressive generation.

🎛 Optimization

AdamW optimizer

Warmup + cosine LR decay

Gradient clipping

Cross entropy loss on logits vs targets

⚡ Training Pipeline

Batch and sequence length configuration

Dataloader producing (x, y) pairs

Forward → loss → backward → optimizer step

Periodic evaluation and checkpointing

🤖 Text Generation

Autoregressive sampling loop

Temperature, top-k, top-p sampling

Using the trained model to produce text token-by-token
