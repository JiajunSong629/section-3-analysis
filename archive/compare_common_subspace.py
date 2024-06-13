from transformers import GemmaForCausalLM, GemmaTokenizer
from transformers import GPT2Model
import torch
import copy
import gc
import numpy as np


def gpt2_w_all():
    model = GPT2Model.from_pretrained("gpt2", output_attentions=True)

    num_layer = 12
    num_heads = 12
    d_model = 768
    d_head = d_model // num_heads

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

    return W_all


def ih_list(model_name):

    def extract(pth):
        # extract IH list
        lines = open(pth).readlines()
        IH_list = []
        for line in lines[1:]:
            layerhead = line.split(":")[0]
            head = layerhead.split("Head")[1]
            layer = layerhead.split("Head")[0].split("Layer")[1]
            IH_list.append([int(layer), int(head)])
        return IH_list

    return extract(pth=f"{model_name}/simple_visz/induction_head_scores.txt")


def gemma_2b_w_all():
    gemma = GemmaForCausalLM.from_pretrained("google/gemma-2b", output_attentions=True)
    configuration = gemma.config
    model = gemma.model
    num_layer = configuration.num_hidden_layers
    num_heads = configuration.num_attention_heads
    d_model = configuration.hidden_size
    d_head = d_model // num_heads

    W_all = torch.zeros(num_layer, num_heads, 4, d_model, d_head)
    for ilayer, layer in enumerate(model.layers):
        attn = layer.self_attn
        W_q = attn.q_proj.weight.T.view(d_model, num_heads, d_model // num_heads)
        W_k = attn.k_proj.weight.T
        W_v = attn.v_proj.weight.T
        W_o = attn.o_proj.weight.T.view(num_heads, d_model // num_heads, d_model)

        for ihead in range(num_heads):
            W_all[ilayer, ihead, 0] = W_q[:, ihead, :]  # (d_model, d_head)
            W_all[ilayer, ihead, 1] = W_k  # (d_model, d_head)
            W_all[ilayer, ihead, 2] = W_v  # (d_model, d_head)
            W_all[ilayer, ihead, 3] = W_o[ihead, :, :].T  # (d_model, d_head)

    return W_all


def calc_common_subspace(W_all, IH_list, K=10):
    num_layer, num_heads, _, d_model, d_head = W_all.shape
    IH_list = IH_list
    W_qk_all = np.zeros((K, d_model, d_model))
    for i in range(K):
        Layer, Head = IH_list[i][0], IH_list[i][1]
        W_qk = W_all[Layer, Head, 0] @ W_all[Layer, Head, 1].T
        W_qk_all[i] = W_qk.numpy(force=True)

    U, S, Vt_common = np.linalg.svd(W_qk_all.reshape(-1, d_model))
    return Vt_common


def edit_gma2(Vt_common, IH_list, rank=5, K=30):
    g2 = GemmaForCausalLM.from_pretrained(
        "google/gemma-2b",
        output_attentions=True,
        device_map="cuda",
    )

    g2.eval()

    rank = 5
    V = Vt_common[:rank, :].T
    P = V @ V.T
    P = np.eye(P.shape[0]) - P

    for j in range(K):
        layer, head = IH_list[j][0], IH_list[j][1]
        print("BEFORE", layer, head)
        print(g2.model.layers[layer].self_attn.k_proj.weight.T.sum().item())
        W = copy.deepcopy(g2.model.layers[layer].self_attn.k_proj.weight.T.data)
        W = torch.tensor(P, device="cuda").float() @ W

        with torch.no_grad():
            g2.model.layers[layer].self_attn.k_proj.weight.copy_(W.T)
        print("AFTER", g2.model.layers[layer].self_attn.k_proj.weight.T.sum().item())


def edit_gpt2(Vt_common, IH_list, rank=5, K=30):
    model = GPT2Model.from_pretrained("gpt2")
    num_layer = 12
    num_heads = 12
    d_model = 768
    d_head = d_model // num_heads

    rank = 5
    V = Vt_common[:rank, :].T
    P = V @ V.T
    P = np.eye(P.shape[0]) - P

    for j in range(K):
        layer, head = IH_list[j][0], IH_list[j][1]
        ind = range(d_model * 1 + head * d_head, d_model * 1 + head * d_head + d_head)

        print("BEFORE", layer, head)
        print(model.h[layer].attn.c_attn.weight.data[:, ind].sum().item())
        W = copy.deepcopy(model.h[layer].attn.c_attn.weight.data[:, ind])
        W = torch.tensor(P).float() @ W
        model.h[layer].attn.c_attn.weight.data[:, ind] = W
        print("AFTER", model.h[layer].attn.c_attn.weight.data[:, ind].sum().item())


def main():
    w_all_gpt2 = gpt2_w_all()
    w_all_gma2 = gemma_2b_w_all()
    ih_gpt2 = ih_list("gpt-2")
    ih_gma2 = ih_list("gemma-2b")

    Vt_gpt2 = calc_common_subspace(w_all_gpt2, ih_gpt2)
    Vt_gma2 = calc_common_subspace(w_all_gma2, ih_gma2)

    print("BEGIN EDITING Gemma-2b\n" + "#" * 100)
    edit_gma2(Vt_gma2, ih_gma2)

    print("\n\nBEGIN EDITING GPT-2\n" + "#" * 100)
    edit_gpt2(Vt_gpt2, ih_gpt2)


main()
