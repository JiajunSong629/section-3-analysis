import copy
import torch
import gc
import fire
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from utils import (
    get_qkov_weight,
    get_configs,
    create_folder,
    load_model,
    inference_probs_and_errs,
    make_input_ids,
    custom_svd,
    calc_rotary_R_mat,
)


def projection_edit(
    model,
    model_name,
    layer_head_pairs,
    P,
    component,
):
    if model_name in ["llama2-7b", "gemma-7b"]:
        P_tensor = torch.tensor(P, device="cuda", dtype=torch.bfloat16)
    else:
        P_tensor = torch.tensor(P, device="cuda").float()

    for layer, head in layer_head_pairs:
        if component == "QK":
            W = get_qkov_weight(model, model_name, layer, head, "k")
            W.copy_(P_tensor @ copy.deepcopy(W))

        elif component == "OV":
            W = get_qkov_weight(model, model_name, layer, head, "o")
            W.copy_(copy.deepcopy(W) @ P_tensor)

    return model


def plot(
    probs,
    errs,
    rank_min,
    rank_max,
    rank_step,
    save_to,
):
    # make plots
    fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 6 * 1))
    ave_probs = [np.mean(vals) for vals in probs.values()]
    ave_errs = [np.mean(vals) for vals in errs.values()]

    axs[0].plot(
        range(rank_min, rank_max + 1, rank_step),
        ave_probs,
        "-o",
        label="projected",
    )
    axs[1].plot(
        range(rank_min, rank_max + 1, rank_step),
        ave_errs,
        "-o",
        label="projected",
    )
    axs[0].axhline(y=ave_probs[0], linestyle="dashed", label="original")
    axs[1].axhline(y=ave_errs[0], linestyle="dashed", label="original")

    for j in range(2):
        titles = [f"Pred {a} under projection" for a in ["probs", "errs"]]
        axs[j].set_xlabel("Subspace rank", weight="bold")
        axs[j].set_ylabel("Target token pred probs/errs", weight="bold")
        axs[j].set_ylim(0, 1)
        axs[j].set_title(titles[j], weight="bold")

    axs[0].legend()
    plt.savefig(save_to, bbox_inches="tight")
    plt.close()


def calc_V(W_all, IH, use_R=False, max_rel_dist=0):
    num_layer, num_heads, _, d_model, d_head = W_all.shape
    K = len(IH)

    W_qk_all = np.zeros((K, d_model, d_model))
    for i in range(K):
        Layer, Head = IH[i][0], IH[i][1]
        if use_R:
            R = calc_rotary_R_mat(
                d_head=d_head,
                max_seq_len=60,
                max_rel_dist=max_rel_dist,
            )[-1]
            W_qk = W_all[Layer, Head, 0] @ R @ W_all[Layer, Head, 1].T
        else:
            W_qk = W_all[Layer, Head, 0] @ W_all[Layer, Head, 1].T

        W_qk_all[i] = W_qk.numpy(force=True)

    U, S, Vt_common = custom_svd(W_qk_all.reshape(-1, d_model))
    return Vt_common


def Vt_to_projection(Vt, rank, project_out):
    V = Vt[:rank, :].T
    P = V @ V.T
    P = np.eye(P.shape[0]) - P if project_out else P
    return P


def proj_exp(
    model_name,
    batch_size,
    seg_len,
    rep,
    ignore_segment,
    ignore_burning,
    K0,
    K1,
    component,
    proj_out,
    rank_min,
    rank_max,
    rank_step,
    save_to,
):
    IH = torch.load(f"checkpoints/{model_name}/IH.pt")
    PTH = torch.load(f"checkpoints/{model_name}/PTH.pt")
    W_all = torch.load(f"checkpoints/{model_name}/W_all.pt")

    T_range = range(seg_len * ignore_segment + ignore_burning, rep * seg_len - 1)
    probs = defaultdict(list)
    errs = defaultdict(list)

    input_ids = make_input_ids(
        batch_size=batch_size,
        seg_len=seg_len,
        rep=rep,
        vocab_size=get_configs(model_name)[-1],
        seed=2024,
        prepend_bos=model_name == "gemma-7b",
    )

    Vt = calc_V(
        W_all,
        IH[:K0],
        use_R=model_name in ["llama2-7b", "gemma-7b"],
        max_rel_dist=seg_len,
    )

    layer_head_pairs = IH[:K1] if component == "QK" else PTH[:K1]

    for rank in range(rank_min, rank_max + 1, rank_step):
        print("RANK", rank)
        P = Vt_to_projection(Vt, rank, proj_out)

        model = load_model(model_name)
        model_edit = projection_edit(
            model=model,
            model_name=model_name,
            layer_head_pairs=layer_head_pairs,
            component=component,
            P=P,
        )

        prob, err = inference_probs_and_errs(model_edit, input_ids)
        probs[rank] = prob[:, T_range]
        errs[rank] = err[:, T_range]

        del model_edit, model
        torch.cuda.empty_cache()
        gc.collect()

        print("ERR", errs[rank].mean())
        print("PROB", probs[rank].mean())

    plot(probs, errs, rank_min, rank_max, rank_step, save_to)

    return probs, errs


def main(
    model_name,
    batch_size=50,
    seg_len=25,
    rep=3,
    ignore_segment=1,
    ignore_burning=4,
    K0=10,
    K1=30,
    rank_min=0,
    rank_max=100,
    rank_step=5,
):
    create_folder("Figs/proj")

    for component in ["QK", "OV"]:
        for proj_out in [True, False]:
            proj_exp(
                model_name=model_name,
                batch_size=batch_size,
                rep=rep,
                seg_len=seg_len,
                ignore_segment=ignore_segment,
                ignore_burning=ignore_burning,
                K0=K0,
                K1=K1,
                proj_out=proj_out,
                component=component,
                rank_min=rank_min,
                rank_max=rank_max,
                rank_step=rank_step,
                save_to=f"Figs/proj/{model_name}_{component}_proj{proj_out}_K0_{K0}_K1_{K1}.png",
            )


if __name__ == "__main__":
    fire.Fire(main)
