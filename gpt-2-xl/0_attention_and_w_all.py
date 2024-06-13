from transformers import GPT2Model
import torch
import numpy as np
import os
from analysis_utils import create_folder

np.random.seed(2024)

model = GPT2Model.from_pretrained("gpt2-xl", output_attentions=True)
configuration = model.config
num_layer = 48
num_heads = 25
d_model = 1600
d_head = d_model // num_heads

vocab_size = configuration.vocab_size
T0 = 25
rep = 3
batch_size = 50

sample_int = np.random.randint(
    low=0, high=configuration.vocab_size, size=T0 * batch_size
).reshape(batch_size, T0)

sample_int = np.array(
    [np.concatenate([segment for _ in range(rep)]) for segment in sample_int]
)
input_ids = torch.Tensor(sample_int).long()
seq_len = input_ids.size(1)

with torch.no_grad():
    output = model(input_ids)

attentions = np.array([a.detach().numpy().mean(0) for a in output.attentions])

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

W0 = model.wte.weight
vals_flip, vecs_flip = np.linalg.eigh((W0.T @ W0).numpy(force=True))
vals, vecs = vals_flip[::-1], vecs_flip[:, ::-1]

W_all = torch.zeros(num_layer, num_heads, 4, d_model, d_head)

for layer in range(num_layer):
    W_q, W_k, W_v = model.h[layer].attn.c_attn.weight.split(d_model, dim=1)
    W_q = W_q.view(d_model, num_heads, d_model // num_heads)
    W_k = W_k.view(d_model, num_heads, d_model // num_heads)
    W_v = W_v.view(d_model, num_heads, d_model // num_heads)
    W_o = model.h[layer].attn.c_proj.weight.view(
        num_heads, d_model // num_heads, d_model
    )

    for head in range(num_heads):
        W_all[layer, head, 0] = W_q[:, head, :]  # (d_model, d_head)
        W_all[layer, head, 1] = W_k[:, head, :]  # (d_model, d_head)
        W_all[layer, head, 2] = W_v[:, head, :]  # (d_model, d_head)
        W_all[layer, head, 3] = W_o[head, :, :].T  # (d_model, d_head)


torch.save(W_all, "W_all.pt")
