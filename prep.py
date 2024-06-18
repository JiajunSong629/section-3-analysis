import torch
import numpy as np
import os
import fire
from utils import create_folder, load_model, make_input_ids, get_configs

np.random.seed(2024)


def get_attentions(model, input_ids):
    with torch.no_grad():
        for i in range(input_ids.size(0)):
            cur_batch_input_ids = input_ids[i : i + 1]
            cur_attentions = model(cur_batch_input_ids).attentions
            cur_attentions = np.array(
                [a.float().numpy(force=True).mean(0) for a in cur_attentions]
            )[np.newaxis, :]
            if i == 0:
                attentions = cur_attentions
            else:
                attentions = np.vstack([attentions, cur_attentions])

    return attentions


def measure_IH_PTH(attentions, seg_len, is_IH):
    sample_size, num_layer, num_head, T, _ = attentions.shape
    scores = np.zeros((num_layer, num_head))
    offset = -(seg_len - 1) if is_IH else -1

    for layer in range(num_layer):
        for head in range(num_head):

            #         A = attentions[layer, head]
            #         A_adjusted = np.zeros((T, T))
            #         A_adjusted[1:, 1:] = A[1:, 1:] / np.sum(A[1:, 1:], axis=1, keepdims=True)
            #         diag1 = np.diag(A_adjusted, -(seg_len - 1))[1:]
            #         diag2 = np.diag(A_adjusted, -(2 * seg_len - 1))[1:]
            #         diag = np.concatenate((diag1[:-seg_len], diag1[-seg_len:] + diag2))
            #         scores[layer, head] = np.mean(diag)

            A = attentions[:, layer, head]
            A_adjusted = np.zeros((sample_size, T, T))
            A_adjusted[:, 1:, 1:] = A[:, 1:, 1:] / np.sum(
                A[:, 1:, 1:], axis=2, keepdims=True
            )
            scores[layer, head] = np.mean(
                np.array(
                    [
                        np.mean(np.diag(A_adjusted[i], offset)[1:])
                        for i in range(sample_size)
                    ]
                )
            )

    idx_sort = np.argsort(scores, axis=None)[::-1]
    head_list = [
        [idx_sort[j] // num_head, idx_sort[j] % num_head] for j in range(len(idx_sort))
    ]

    for layer, head in head_list[:20]:
        print(f"Layer {layer} Head {head}: score {scores[layer, head]}")

    return head_list


def get_W_all(model, model_name):
    num_layer, num_head, d_model, d_head, _ = get_configs(model_name)
    W_all = torch.zeros(num_layer, num_head, 4, d_model, d_head)

    if model_name in ["gpt2", "gpt2-xl"]:
        model = model.transformer
        for layer in range(num_layer):
            W_q, W_k, W_v = model.h[layer].attn.c_attn.weight.split(d_model, dim=1)
            W_q = W_q.view(d_model, num_head, d_head)
            W_k = W_k.view(d_model, num_head, d_head)
            W_v = W_v.view(d_model, num_head, d_head)
            W_o = model.h[layer].attn.c_proj.weight.view(num_head, d_head, d_model)

            for head in range(num_head):
                W_all[layer, head, 0] = W_q[:, head, :]  # (d_model, d_head)
                W_all[layer, head, 1] = W_k[:, head, :]  # (d_model, d_head)
                W_all[layer, head, 2] = W_v[:, head, :]  # (d_model, d_head)
                W_all[layer, head, 3] = W_o[head, :, :].T  # (d_model, d_head)

    elif model_name in ["gemma-7b", "llama2-7b"]:
        model = model.model
        for ilayer, layer in enumerate(model.layers):
            attn = layer.self_attn

            W_q = attn.q_proj.weight.T.view(d_model, num_head, d_head)
            W_k = attn.k_proj.weight.T.view(d_model, num_head, d_head)
            W_v = attn.v_proj.weight.T.view(d_model, num_head, d_head)
            W_o = attn.o_proj.weight.T.view(num_head, d_head, d_model)

            for ihead in range(num_head):
                W_all[ilayer, ihead, 0] = W_q[:, ihead, :]  # (d_model, d_head)
                W_all[ilayer, ihead, 1] = W_k[:, ihead, :]  # (d_model, d_head)
                W_all[ilayer, ihead, 2] = W_v[:, ihead, :]  # (d_model, d_head)
                W_all[ilayer, ihead, 3] = W_o[ihead, :, :].T  # (d_model, d_head)

    return W_all


def main(
    model_name,
    batch_size=50,
    seg_len=25,
    rep=2,
):
    vocab_size = get_configs(model_name)[-1]
    model = load_model(model_name)

    input_ids = make_input_ids(
        batch_size=batch_size,
        seg_len=seg_len,
        rep=rep,
        vocab_size=vocab_size,
        prepend_bos=model_name == "gemma-7b",
    )

    attentions = get_attentions(model, input_ids)
    IH = measure_IH_PTH(attentions=attentions, seg_len=seg_len, is_IH=True)
    PTH = measure_IH_PTH(attentions=attentions, seg_len=seg_len, is_IH=False)
    W_all = get_W_all(model=model, model_name=model_name)

    save_dir = f"checkpoints/{model_name}"
    create_folder(save_dir)

    torch.save(IH, f"{save_dir}/IH.pt")
    torch.save(PTH, f"{save_dir}/PTH.pt")
    torch.save(W_all, f"{save_dir}/W_all.pt")


if __name__ == "__main__":
    fire.Fire(main)
