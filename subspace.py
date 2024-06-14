import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import fire

from utils import calc_QK_OV, create_folder, custom_svd, load_model


def subspace_matching(
    W_all,
    IH=None,
    PTH=None,
    rank=10,
    num_samp=50,
    method="largest",
    save_to=None,
):
    if IH is not None and PTH is None:
        K0 = len(IH)
        K1 = K0
        LayerHeadPair0, LayerHeadPair1 = IH, IH
    elif IH is None and PTH is not None:
        K1 = len(PTH)
        K0 = K1
        LayerHeadPair0, LayerHeadPair1 = PTH, PTH
    else:
        K0, K1 = len(IH), len(PTH)
        LayerHeadPair0, LayerHeadPair1 = IH, PTH

    num_layer, num_heads, _, d_model, d_head = W_all.shape

    # first calculate a random baseline
    match_baseline = np.zeros(num_samp)
    for i in range(num_samp):
        mat1 = np.random.randn(d_model, d_head) @ np.random.randn(d_head, d_model)
        mat2 = np.random.randn(d_model, d_head) @ np.random.randn(d_head, d_model)
        U1, s1, Vt1 = np.linalg.svd(mat1)
        U2, s2, Vt2 = np.linalg.svd(mat2)
        _, s_match_u, _ = np.linalg.svd(Vt1[:rank, :] @ Vt2[:rank, :].T)
        match_baseline[i] = (
            s_match_u[0] if method == "largest" else np.sqrt(np.mean(s_match_u**2))
        )

    s_match = np.zeros((K0, K1))
    for i0 in range(K0):
        for i1 in range(K1):
            Layer0, Head0 = LayerHeadPair0[i0][0], LayerHeadPair0[i0][1]
            Layer1, Head1 = LayerHeadPair1[i1][0], LayerHeadPair1[i1][1]
            W_0 = calc_QK_OV(
                W_all,
                Layer0,
                Head0,
                QK=(IH is not None),
                OV=(IH is None),
            )
            W_1 = calc_QK_OV(
                W_all,
                Layer1,
                Head1,
                OV=(PTH is not None),
                QK=(PTH is None),
            )

            U_0, s_0, Vt_0 = custom_svd(W_0.numpy(force=True))
            U_1, s_1, Vt_1 = custom_svd(W_1.numpy(force=True))

            A0 = Vt_0.T if IH is not None else U_0
            A1 = U_1 if PTH is not None else Vt_1.T

            _, s, _ = custom_svd(A0[:, :rank].T @ A1[:, :rank])
            s_match[i0, i1] = s[0] if method == "largest" else np.sqrt(np.mean(s**2))

    yticklabels = [f"L{layerhead[0]}H{layerhead[1]}" for layerhead in LayerHeadPair0]
    xticklabels = [f"L{layerhead[0]}H{layerhead[1]}" for layerhead in LayerHeadPair1]
    baseline = np.mean(match_baseline)
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    sns.heatmap(
        s_match,
        ax=axs[0],
        xticklabels=xticklabels,
        yticklabels=yticklabels,
    )
    axs[0].set_title(f"Baseline: {baseline:.3f}")

    axs[1].hist(s_match.flatten(), bins=30, edgecolor="white")
    axs[1].axvline(x=baseline, color="red", linestyle="dashed")
    axs[1].set_title(f"Baseline: {baseline:.3f}")

    plt.savefig(save_to, bbox_inches="tight")
    plt.close()

    return s_match, match_baseline


def main(
    model_name,
    K=10,
    rank=10,
    method="largest",
):
    create_folder("Figs/subspace")

    W_all = torch.load(f"checkpoints/{model_name}/W_all.pt")
    IH = torch.load(f"checkpoints/{model_name}/IH.pt")[:K]
    PTH = torch.load(f"checkpoints/{model_name}/PTH.pt")[:K]

    subspace_matching(
        W_all,
        IH=IH,
        PTH=None,
        rank=rank,
        save_to=f"Figs/subspace/{model_name}_subspace_IH_K{K}_rank{rank}_{method}.png",
    )

    subspace_matching(
        W_all,
        IH=None,
        PTH=PTH,
        rank=rank,
        save_to=f"Figs/subspace/{model_name}_subspace_PTH_K{K}_rank{rank}_{method}.png",
    )

    subspace_matching(
        W_all,
        IH=IH,
        PTH=PTH,
        rank=rank,
        save_to=f"Figs/subspace/{model_name}_subspace_IH_PTH_K{K}_rank{rank}_{method}.png",
    )


if __name__ == "__main__":
    fire.Fire(main)
