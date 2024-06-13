import copy
import torch
from transformers import GemmaForCausalLM, GPT2Model

# gemma = GemmaForCausalLM.from_pretrained("google/gemma-2b", output_attentions=True)
# gemma.eval()

# # Print original weights
# print("Original weights:")
# print(gemma.model.layers[0].self_attn.k_proj.weight)

# # Print new weights
# print("\nNew weights:")
# gemma.model.layers[0].self_attn.k_proj.weight.data = torch.randn(256, 204)
# print(gemma.model.layers[0].self_attn.k_proj.weight)


# g2 = GemmaForCausalLM.from_pretrained(
#     "google/gemma-2b",
#     output_attentions=True,
#     device_map="cuda",
# )

# g2.eval()

# rank = 5
# P = torch.randn(2048, 2048)

# print("BEFORE", g2.model.layers[0].self_attn.k_proj.weight.T.sum().item())

# W = copy.deepcopy(g2.model.layers[0].self_attn.k_proj.weight.T.data)
# W = torch.tensor(P, device="cuda").float() @ W

# print(W.shape)
# with torch.no_grad():
#     g2.model.layers[0].self_attn.k_proj.weight.copy_(W.T)

# print("AFTER", g2.model.layers[0].self_attn.k_proj.weight.T.sum().item())


m = GPT2Model.from_pretrained("gpt2")
n = copy.deepcopy(m)

print("n", n.h[0].attn.c_attn.weight.sum())
print("m", m.h[0].attn.c_attn.weight.sum())

# edit n
d_model = 768
head = 5
d_head = 64
ind = range(d_model * 1 + head * d_head, d_model * 1 + head * d_head + d_head)
n.h[0].attn.c_attn.weight.data[:, ind] = torch.randn(d_model, d_head)


# check m
print("n", n.h[0].attn.c_attn.weight.sum())
print("m", m.h[0].attn.c_attn.weight.sum())
