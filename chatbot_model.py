import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import numpy as np
import time
import pickle
import argparse

# Imports and Initial Setup:

# Libraries like torch, numpy, and argparse are imported. torch is used for building and training neural networks, numpy for numerical operations, 
# and argparse for parsing command-line arguments.

# Argument Parsing:

# An argument parser is set up to accept a -batch_size parameter from the command line. This demonstrates how to get external configurations 
# but isn't used further in the script.

# Data Processing:

# Loads a vocabulary from a file, creates mappings from characters to integers and vice versa (string_to_int, int_to_string), and defines functions 
# for encoding and decoding strings to and from sequences of integers.

# Model Components:

# Defines several PyTorch nn.Module classes for the components of the GPT model:
# Head: Implements a single head of self-attention.
# MultiHeadAttention: Combines multiple Head instances to form multi-head self-attention.
# FeedForward: A feedforward neural network used within each transformer block.
# Block: A single transformer block, combining multi-head attention and a feedforward network with layer normalization.
# GPTLanguageModel: The complete model, assembling the embedding layers, multiple Block instances, and the output layer to predict the next token.

# Model Initialization and Loading:

# Initializes an instance of GPTLanguageModel with the vocabulary size and loads model parameters from a file. This demonstrates how to save and load 
# models with PyTorch, although the model initialization just before loading is redundant since the loaded model will overwrite it.

# Interactive Text Generation:

# Enters an infinite loop, prompting the user for input and using the model to generate and print a completion. This demonstrates how to use the model 
# for generating text based on input prompts.

# Text Generation Method (generate):

# Within GPTLanguageModel, the generate method takes an initial sequence of indices (tokens) and generates a specified number of additional tokens one
#  at a time. It showcases how to use the model's outputs to sample the next token and concatenate it to the input for subsequent predictions.


# parser = argparse.ArgumentParser(description = 'This is a demo')
# parser.add_argument('-batch_size', type=str, required=True, help='Please provide a batch_size')
# args = parser.parse_args()
# print(f'batch size: {args.batch_size}')

device = 'cpu'
batch_size = 32
block_size = 64
max_iters = 200
learning_rate = 3e-4 #1e-3, 1e-4
# eval_interval = 500
eval_iters = 100
dropout = 0.2
n_embd = 384
n_head = 4
n_layer = 4


chars = ""
with open("wizardofoz/vocab.txt",'r',encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)

string_to_int = { ch: i for i, ch in enumerate(chars) }
int_to_string = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.attn = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.attn(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x


class LargeLanguageModel(nn.Module):
   def __init__(self, vocab_size):
      super().__init__()
      self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
      self.position_embedding_table = nn.Embedding(block_size, n_embd)
      self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range (n_layer)])

      self.ln_f = nn.LayerNorm(n_embd)
      self.lm_head = nn.Linear(n_embd, vocab_size)
      self.apply(self.__init__weights)

   def __init__weights(self, module):
       if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

   def forward(self, index, targets=None):
      B, T = index.shape
      tok_emb = self.token_embedding_table(index)  # (B, T, C)
      pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
      x = tok_emb + pos_emb  # (B, T, C)
      x = self.blocks(x)  # (B, T, C)
      x = self.ln_f(x)  # (B, T, C)
      logits = self.lm_head(x)  # (B, T, vocab_size)
      if targets is None:
            loss = None
      else:
         B, T, C = logits.shape
         logits = logits.view(B*T, C)
         targets = targets.view(B*T)
         loss = F.cross_entropy(logits, targets)

      return logits, loss
   
   def generate(self, index, max_new_tokens):
      for _ in range(max_new_tokens):
         index_cond = index[:, -block_size:]
         logits, loss = self.forward(index_cond)
         logits = logits[:, -1, :]
         probs = F.softmax(logits, dim=-1)
         index_next = torch.multinomial(probs, num_samples=1)
         index = torch.cat((index, index_next), dim=1)
      return index
   
model = LargeLanguageModel(vocab_size)
print('Loading model parameters...')
with open('woo_model_1.pkl', 'rb') as f:
    model = pickle.load(f)
print('Loaded successfully!')
m=model.to(device)

while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f'Completion:\n{generated_chars}')