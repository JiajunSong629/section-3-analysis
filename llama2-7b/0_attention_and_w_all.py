from transformers import LlamaForCausalLM
import torch
import numpy as np
import os
import math
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import seaborn as sns
from torch.nn import functional as F
from analysis_utils import create_folder

llama = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    output_attentions=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
configuration = llama.config
num_layer = configuration.num_hidden_layers
num_heads = configuration.num_key_value_heads
d_model = configuration.hidden_size
d_head = d_model // num_heads


vocab_size = configuration.vocab_size
T0 = 25
rep = 3
batch_size = 5

sample_int = np.random.randint(
    low=0, high=configuration.vocab_size, size=T0 * batch_size
).reshape(batch_size, T0)

sample_int = np.array(
    [np.concatenate([segment for _ in range(rep)]) for segment in sample_int]
)
# sample_int = np.hstack([2 * np.ones((batch_size, 1), dtype=int), sample_int])
input_ids = torch.Tensor(sample_int).long().cuda()
seq_len = input_ids.size(1)

with torch.no_grad():
    out = llama(input_ids)

attentions = np.array([a.float().numpy(force=True).mean(0) for a in out.attentions])
# attentions = attentions.mean(1)

num_layer = configuration.num_hidden_layers
num_heads = configuration.num_attention_heads
d_model = configuration.hidden_size
d_head = d_model // num_heads

##############################################################################

dir_name = "simple_visz"
create_folder(dir_name)
filename = os.path.join(dir_name, f"induction_head_scores.txt")

scores = np.zeros((num_layer, num_heads))
for layer in range(num_layer):
    for head in range(num_heads):
        A = attentions[layer, head]
        A_adjusted = np.zeros((seq_len, seq_len))
        A_adjusted[1:, 1:] = A[1:, 1:] / np.sum(A[1:, 1:], axis=1, keepdims=True)
        diag1 = np.diag(A_adjusted, -(T0 - 1))[1:]
        diag2 = np.diag(A_adjusted, -(2 * T0 - 1))[1:]
        diag = np.concatenate((diag1[:-T0], diag1[-T0:] + diag2))
        scores[layer, head] = np.mean(diag)

idx_sort = np.argsort(scores, axis=None)[::-1]
IH_list = [
    [idx_sort[j] // num_heads, idx_sort[j] % num_heads] for j in range(len(idx_sort))
]

with open(filename, "w") as file:
    print(
        f"Ranking induction heads (most likely to unlikely) by attention scores",
        file=file,
    )
    for j, pair in enumerate(IH_list):
        layer, head = pair[0], pair[1]
        print(f"Layer {layer} Head {head}: score {scores[layer, head]}", file=file)
        if j <= 20:
            print(f"Layer {layer} Head {head}: score {scores[layer, head]}")

#######################################################################################

dir_name = "simple_visz"
create_folder(dir_name)
filename = os.path.join(dir_name, f"shifting_head_scores.txt")

scores = np.zeros((num_layer, num_heads))
for layer in range(num_layer):
    for head in range(num_heads):
        A = attentions[layer, head]
        A_adjusted = np.zeros((seq_len, seq_len))
        A_adjusted[1:, 1:] = A[1:, 1:] / np.sum(A[1:, 1:], axis=1, keepdims=True)
        diag = np.diag(A_adjusted, -1)[1:]
        scores[layer, head] = np.mean(diag)

idx_sort = np.argsort(scores, axis=None)[::-1]
Shifting_list = [
    [idx_sort[j] // num_heads, idx_sort[j] % num_heads] for j in range(len(idx_sort))
]

with open(filename, "w") as file:
    print(
        f"Ranking shifting heads (most likely to unlikely) by attention scores",
        file=file,
    )
    for j, pair in enumerate(Shifting_list):
        layer, head = pair[0], pair[1]
        print(f"Layer {layer} Head {head}: score {scores[layer, head]}", file=file)
        if j <= 20:
            print(f"Layer {layer} Head {head}: score {scores[layer, head]}")

####################################################################################

W_all = torch.zeros(num_layer, num_heads, 4, d_model, d_head)

for ilayer, layer in enumerate(llama.model.layers):
    attn = layer.self_attn

    W_q = attn.q_proj.weight.T.view(d_model, num_heads, d_model // num_heads)
    W_k = attn.k_proj.weight.T.view(d_model, num_heads, d_model // num_heads)
    W_v = attn.v_proj.weight.T.view(d_model, num_heads, d_model // num_heads)
    W_o = attn.o_proj.weight.T.view(num_heads, d_model // num_heads, d_model)

    for ihead in range(num_heads):
        W_all[ilayer, ihead, 0] = W_q[:, ihead, :]  # (d_model, d_head)
        W_all[ilayer, ihead, 1] = W_k[:, ihead, :]  # (d_model, d_head)
        W_all[ilayer, ihead, 2] = W_v[:, ihead, :]  # (d_model, d_head)
        W_all[ilayer, ihead, 3] = W_o[ihead, :, :].T  # (d_model, d_head)


torch.save(W_all, "W_all.pt")
