# importing required libraries
import torch
import torch.nn.functional as F
import math, copy, re
import os
import gc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from transformers import GemmaForCausalLM


def create_folder(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


def calc_QK_OV(W_all, layer, head, QK=False, OV=False, return_original=False):
    """
    calc_QK_OV returns the W_qk or W_ov matrix at specified layer and head
    Arguments:
        W_all is a tensor of shape (num_layer, num_heads, 4, d_model, d_head) extracted from pretrained models; see definition in related function
        layer is the specified layer
        head is the specified head
    Returns:
        A 2-d torch array of size (d_model, d_model)
    """
    assert QK + OV == 1, "Only one of QK and OV should be True, the other is false"
    if QK:
        if return_original:
            return W_all[layer, head, 0], W_all[layer, head, 1]
        W_qk = W_all[layer, head, 0] @ W_all[layer, head, 1].T
        return W_qk
    else:
        if return_original:
            return W_all[layer, head, 3], W_all[layer, head, 2]
        W_ov = W_all[layer, head, 3] @ W_all[layer, head, 2].T
        return W_ov


################################# Matching ###################################
def qkov_matching(W_all, IH_list, save_dir, num_top=100):
    """
    qkov_matching calculates W_qk * W_ov and related statistics, then making plots between every W_qk in the induction head list
        and any W_qk in layers earlier to W_qk
    Arguments:
        W_all is a tensor of shape (num_layer, num_heads, 4, d_model, d_head) extracted from pretrained models; see definition in related function
        IH_list is a list of (layer, head) pairs which are top-scoring induction heads
        save_dir is the folder for saving plots
        num_top is the number of coordinates to show in the plots; for visual reasons, usually num_top < d_model is better
    Returns:
        res: a list of 3-order arrays, each array of size (Layer, num_heads, 4) that stores the mean & std statistics for multiplying
            an QK component from IH_list with each earlier OV component
    """
    K = len(IH_list)
    num_layer, num_heads, _, d_model, d_head = W_all.shape
    create_folder(save_dir)

    for i, (Layer, Head) in enumerate(IH_list):
        create_folder(os.path.join(save_dir, f"IH_L{Layer}_H{Head}"))
        res = []
        match_scores = np.zeros((Layer, num_heads, 4))

        # W_qk = W_all[Layer, Head, 0] @ W_all[Layer, Head, 1].T
        W_qk = calc_QK_OV(W_all, Layer, Head, QK=True)

        for layer in range(Layer):
            for head in range(num_heads):
                save_to = os.path.join(
                    save_dir, f"IH_L{Layer}_H{Head}", f"layer_{layer}_head_{head}"
                )
                # W_ov = W_all[layer, head, 3] @ W_all[layer, head, 2].T
                W_ov = calc_QK_OV(W_all, layer, head, OV=True)
                W_qkov = (W_qk @ W_ov).numpy(force=True)

                # plot the heatmap
                fig, axs = plt.subplots(1, 1, figsize=(8, 8))
                sns.heatmap(W_qkov[:num_top, :num_top], ax=axs)
                axs.set_title(
                    f"QK: L{Layer}H{Head}, OV: L{layer}H{head}", weight="bold"
                )
                plt.savefig(save_to, bbox_inches="tight")
                plt.close()
                # calculate mean & std
                match_scores[layer, head, :2] = np.mean(np.diag(W_qkov)), np.std(
                    np.diag(W_qkov)
                )
                match_scores[layer, head, 2:4] = np.mean(W_qkov), np.std(W_qkov)

        # plot normalized score
        scores_nml = (match_scores[:, :, 0] - match_scores[:, :, 2]) / match_scores[
            :, :, 3
        ]
        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        sns.heatmap(
            scores_nml,
            ax=axs,
            xticklabels=np.arange(1, num_heads + 1, dtype=int),
            yticklabels=np.arange(1, Layer + 1, dtype=int),
            cmap="bwr",
        )
        axs.set_xlabel("Head", weight="bold")
        axs.set_ylabel("Layer", weight="bold")
        axs.set_title(f"IH Layer {Layer}, Head {Head}", weight="bold")
        plt.savefig(os.path.join(save_dir, f"IH_L{Layer}_H{Head}"), bbox_inches="tight")
        plt.close()
        res.append(match_scores)

    return res


def qkov_matching_summary(W_all, IH_list, Shifting_list, save_dir):
    K0, K1 = len(IH_list), len(Shifting_list)
    num_layer, num_heads, _, d_model, d_head = W_all.shape
    create_folder(save_dir)

    scores_nml = np.zeros((K0, K1))
    for i0, (Layer0, Head0) in tqdm(enumerate(IH_list)):
        for i1, (Layer1, Head1) in enumerate(Shifting_list):
            W_qk = calc_QK_OV(W_all, Layer0, Head0, QK=True)
            W_ov = calc_QK_OV(W_all, Layer1, Head1, OV=True)
            W_qkov = (W_qk @ W_ov).numpy(force=True)
            scores_nml[i0, i1] = (np.mean(np.diag(W_qkov)) - np.mean(W_qkov)) / np.std(
                W_qkov
            )

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    yticklabels = [f"L{layerhead[0]}H{layerhead[1]}" for layerhead in IH_list]
    xticklabels = [f"L{layerhead[0]}H{layerhead[1]}" for layerhead in Shifting_list]
    sns.heatmap(scores_nml, ax=axs, xticklabels=xticklabels, yticklabels=yticklabels)
    axs.set_xlabel("Previous token head", weight="bold")
    axs.set_ylabel("IH head", weight="bold")
    axs.set_title(f"IH-Shifting matching", weight="bold")
    plt.savefig(
        os.path.join(save_dir, f"IH_PTH_matching_{K0}_{K1}"), bbox_inches="tight"
    )
    plt.close()

    plt.hist(scores_nml.flatten(), bins=30, edgecolor="white")
    plt.title(f"Histogram of IH-PTH matching Z-score", weight="bold")
    plt.savefig(
        os.path.join(save_dir, f"IH_PTH_matching_histogram_{K0}_{K1}"),
        bbox_inches="tight",
    )
    plt.close()

    return scores_nml


def custom_svd(M):
    """
    assume M = (n, p) and n << p
    U @ S @ V.t = M

    1. M @ M.t = U @ S^2 @ U.t
    2. V.t = inv(S) @ U.t @ M
    3. V = M.t @ U @ inv(S)
    returns U, S, V.T
    """
    n, p = M.shape
    if n > p:
        V, S, Ut = custom_svd(M.T)
        return Ut.T, S, V.T

    G = M @ M.T  # n, n
    S2, U = np.linalg.eigh(G)
    S = np.sqrt(np.abs(S2))

    sorted_indices = np.argsort(S)[::-1]
    S = S[sorted_indices]
    U = U[:, sorted_indices]

    V = M.T @ U @ np.diag(1 / S)

    return U, S, V.T


def svd2(M):
    # M: n, p;  n > p
    G = M.T @ M  # p, p, M = U S Vt, G = V S2 Vt
    S2, V = np.linalg.eigh(G)  # G = V S2 V.T
    return V


def svdAB(A, B):
    """
    returns the SVD of A@B.T

    UaSaVa = A, UbSbVb = B.T
    C = SaVaUbSb

    UcScVc = C

    returns UaUc, Sc, VcVb
    """
    Ua, Sa, Vat = custom_svd(A)
    Ub, Sb, Vbt = custom_svd(B.T)
    C = np.diag(Sa) @ Vat @ Ub @ np.diag(Sb)
    Uc, Sc, Vct = custom_svd(C)
    return Ua @ Uc, Sc, Vct @ Vbt


def subspace_matching(
    W_all, save_dir, IH_list=None, Shifting_list=None, ranks=[10], num_samp=50
):
    """
    subspace_matching calculates generalized cosine similarity scores between two subspaces---one QK component taken from IH_list and the other
        OV component taken from Shifting_list, for every QK-OV combinations
    Arguments:
        W_all is a tensor of shape (num_layer, num_heads, 4, d_model, d_head) extracted from pretrained models; see definition in related function
        save_dir is the folder for saving plots
        IH_list is a list of (layer, head) pairs which are top-scoring induction heads
        Shifting_list is a list of (layer, head) pairs which are top-scoring shifting heads, at least one of IH_list, Shifting_list must be not None
        ranks is a list of integers as the subspace rank
        num_samp is the number of random runs when calculating a random baseline
        score_type is the type of generalized cosine similarity scores,
    Returns:
        s_match: 4-order array of size (K0, K1, R, 2) where K0 and K1 are the length of the two lists, R is the length of ranks
            we report two scoring methods--- if 'average' then average of singular values between two projection matrices
            if 'largest' then the top singular value
        match_baseline: similar scoring results using random matrices
    """
    assert (IH_list is not None) or (
        Shifting_list is not None
    ), "At least one of IH_list, Shifting_list must be not None"
    if IH_list is not None and Shifting_list is None:
        K0 = len(IH_list)
        K1 = K0
        LayerHeadPair0, LayerHeadPair1 = IH_list, IH_list
    elif IH_list is None and Shifting_list is not None:
        K1 = len(Shifting_list)
        K0 = K1
        LayerHeadPair0, LayerHeadPair1 = Shifting_list, Shifting_list
    else:
        K0, K1 = len(IH_list), len(Shifting_list)
        LayerHeadPair0, LayerHeadPair1 = IH_list, Shifting_list

    R = len(ranks)
    num_layer, num_heads, _, d_model, d_head = W_all.shape
    create_folder(save_dir)

    # first calculate a random baseline
    match_baseline = np.zeros((num_samp, R, 2))
    for i in tqdm(range(num_samp)):
        # mat1 = np.random.randn(d_model, d_head) @ np.random.randn(d_head, d_model)
        # mat2 = np.random.randn(d_model, d_head) @ np.random.randn(d_head, d_model)
        # U1, s1, Vt1 = np.linalg.svd(mat1)
        # U2, s2, Vt2 = np.linalg.svd(mat2)

        U1, s1, Vt1 = svdAB(
            np.random.randn(d_model, d_head), np.random.randn(d_model, d_head)
        )
        U2, s2, Vt2 = svdAB(
            np.random.randn(d_model, d_head), np.random.randn(d_model, d_head)
        )

        for j, rank in enumerate(ranks):
            _, s_match_u, _ = np.linalg.svd(Vt1[:rank, :] @ Vt2[:rank, :].T)
            match_baseline[i, j, 0] = s_match_u[0]
            match_baseline[i, j, 1] = np.sqrt(np.mean(s_match_u**2))

    s_match = np.zeros((K0, K1, R, 2))
    for i0 in range(K0):
        for i1 in range(K1):
            Layer0, Head0 = LayerHeadPair0[i0][0], LayerHeadPair0[i0][1]
            Layer1, Head1 = LayerHeadPair1[i1][0], LayerHeadPair1[i1][1]
            # W_qk0 = W_all[Layer0, Head0, 0] @ W_all[Layer0, Head0, 1].T
            # W_qk1 = W_all[Layer1, Head1, 0] @ W_all[Layer1, Head1, 1].T
            # U_qk0, s_qk0, Vt_qk0 = np.linalg.svd(W_qk0.numpy(force=True))
            # U_qk1, s_qk1, Vt_qk1 = np.linalg.svd(W_qk1.numpy(force=True))
            W_00, W_01 = calc_QK_OV(
                W_all,
                Layer0,
                Head0,
                QK=(IH_list is not None),
                OV=(IH_list is None),
                return_original=True,
            )
            W_10, W_11 = calc_QK_OV(
                W_all,
                Layer1,
                Head1,
                OV=(Shifting_list is not None),
                QK=(Shifting_list is None),
                return_original=True,
            )

            # U_0, s_0, Vt_0 = np.linalg.svd(W_0.numpy(force=True))
            # U_1, s_1, Vt_1 = np.linalg.svd(W_1.numpy(force=True))
            U_0, s_0, Vt_0 = svdAB(W_00.numpy(force=True), W_01.numpy(force=True))
            U_1, s_1, Vt_1 = svdAB(W_10.numpy(force=True), W_11.numpy(force=True))

            A0 = Vt_0.T if IH_list is not None else U_0
            A1 = U_1 if Shifting_list is not None else Vt_1.T

            for j, rank in enumerate(ranks):
                _, s, _ = np.linalg.svd(A0[:, :rank].T @ A1[:, :rank])
                s_match[i0, i1, j, 0] = s[0]
                s_match[i0, i1, j, 1] = np.sqrt(np.mean(s**2))

    for j, rank in enumerate(ranks):
        for k, method in enumerate(["largest", "mean"]):
            save_to = os.path.join(save_dir, f"rank_{rank}_{method}")
            yticklabels = [
                f"L{layerhead[0]}H{layerhead[1]}" for layerhead in LayerHeadPair0
            ]
            xticklabels = [
                f"L{layerhead[0]}H{layerhead[1]}" for layerhead in LayerHeadPair1
            ]
            baseline = np.mean(match_baseline[:, j, k])
            fig, axs = plt.subplots(1, 1, figsize=(6, 6))
            sns.heatmap(
                s_match[:, :, j, k],
                ax=axs,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
            )
            axs.set_title(f"Baseline: {baseline:.3f}")
            plt.savefig(save_to, bbox_inches="tight")
            plt.close()

            plt.hist(s_match[:, :, j, k].flatten(), bins=30, edgecolor="white")
            plt.axvline(x=baseline, color="red", linestyle="dashed")
            plt.title(f"Baseline: {baseline:.3f}")
            plt.savefig(
                os.path.join(save_dir, f"rank_{rank}_{method}_histogram"),
                bbox_inches="tight",
            )
            plt.close()

    return s_match, match_baseline


def pred_probs(output, W_out_embed, correct_next_token_ids, logits=None):
    """
    pred_probs calculates the prediction probabilities given outputs of a model
    Arguments:
        output is the output of a transformer model, tensor size (batch_size, seq_len, d_model),
        W_out_embed is the output embedding matrix (or classification matrix), tensor size (vocab_size, d_model)
        correct_next_token_ids is the target token ids, tensor size (batch_size, seq_len)
        logits are optionally logits of a transformer model, which, if not given, will be calculated
                based on output and W_out_embed
    Returns:
        res: 3-order tensor of size (batch_size, seq_len, 3), containing top-2 prediction probs and prob for the target tokens
        out_indices: 3-order tensor of size (batch_size, seq_len, 2), containing top-2 prediction token indices
    """
    B, T, D = output.size()
    vocab_size, D2 = W_out_embed.size()
    B3, T3 = correct_next_token_ids.size()
    assert D == D2, "Inconsistent dimensions from inputs output and W_out_embed"
    assert (
        B == B3 and T == T3
    ), "Inconsistent batch size from inputs output and correct_next_token_ids"

    if logits is not None:
        B2, T2, vocab_size2 = logits.size()
        assert B2 == B and T2 == T and D2 == D, "Inconsistent dimensions from logits"
    else:
        logits = output @ W_out_embed.T

    res = np.zeros((B, T, 3))
    probs = F.softmax(logits.float(), dim=-1).numpy(force=True)

    sorted, indices = torch.sort(logits, dim=-1, descending=True)
    indices = indices.numpy(force=True)
    for i in range(B):
        for t in range(T):
            res[i, t, :2] = probs[i, t, indices[i, t, :2]]
            res[i, t, 2] = probs[i, t, correct_next_token_ids[i, t]]
    out_indices = indices[:, :, :2]
    return res, out_indices


################################# Intervention: shuffling heads ###################################


def load_gemma():
    gemma = GemmaForCausalLM.from_pretrained(
        "google/gemma-7b",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        output_attentions=True,
    )
    gemma.eval()
    return gemma


def make_input_ids(batch_size, T0, rep, vocab_size):
    # draw batch of random tokens and make repetitions
    sample_int = np.random.randint(
        low=0, high=vocab_size, size=batch_size * T0
    ).reshape(batch_size, T0)
    sample_int = np.concatenate(tuple([sample_int] * rep), axis=1)
    input_ids = torch.Tensor(sample_int).long()

    return input_ids.cuda()


def inference_probs_and_errs(model, input_ids):
    with torch.no_grad():
        for i in range(input_ids.size(0)):
            cur_batch_input_ids = input_ids[i : i + 1]
            cur_logits = model(cur_batch_input_ids).logits
            if i == 0:
                logits = cur_logits
            else:
                logits = torch.concat([logits, cur_logits])

    probs = F.softmax(logits.float(), dim=-1)
    top_prob, pred_next_token_ids = torch.topk(probs, dim=-1, k=1)

    return top_prob, pred_next_token_ids


def print_gpu_mem_usage():
    torch.cuda.reset_peak_memory_stats(device=None)
    print(
        f"gpu used {torch.cuda.max_memory_allocated(device=None)/ np.power(2, 30): .3f}G memory"
    )


def _layer_head_pairs(config, LayerHeadPairs, replace_from_outside_list):
    K = len(LayerHeadPairs)
    num_layer = config.num_hidden_layers
    d_model = config.hidden_size
    num_heads = config.num_key_value_heads
    d_head = d_model // num_heads

    # define ayerHeadPairs0: the list of (layer, head) pairs that provide tensors to be copied,
    # randomly selected heads outside LayerHeadPairs
    if replace_from_outside_list:
        # list1, list2 = np.random.randint(low=1, high=num_layer-1, size=K), np.random.randint(low=0, high=num_heads, size=K)
        # LayerHeadPairs0 = [[list1[j], list2[j]] for j in range(K)]
        LayerHeadPairs0 = []
        for trial in range(5000):
            layer, head = np.random.randint(
                low=1, high=num_layer - 1
            ), np.random.randint(low=0, high=num_heads)
            if [layer, head] not in LayerHeadPairs:
                LayerHeadPairs0.append([layer, head])
            if len(LayerHeadPairs0) == K:
                break
    else:
        LayerHeadPairs0 = LayerHeadPairs

    return LayerHeadPairs0


def _components_to_copy(model, config, LayerHeadPairs0):
    K = len(LayerHeadPairs0)
    num_layer = config.num_hidden_layers
    d_model = config.hidden_size
    num_heads = config.num_key_value_heads
    d_head = d_model // num_heads

    components_copy = {}
    # copying tensors from the provide list (layer, head)
    for j in range(K):
        layer, head = LayerHeadPairs0[j][0], LayerHeadPairs0[j][1]
        ind = range(head * d_head, head * d_head + d_head)

        name = f"L_{layer}_H_{head}_Q_weight"
        components_copy[name] = copy.deepcopy(
            model.layers[layer].self_attn.q_proj.weight.T[:, ind].data
        )
        name = f"L_{layer}_H_{head}_K_weight"
        components_copy[name] = copy.deepcopy(
            model.layers[layer].self_attn.k_proj.weight.T[:, ind].data
        )
        name = f"L_{layer}_H_{head}_V_weight"
        components_copy[name] = copy.deepcopy(
            model.layers[layer].self_attn.v_proj.weight.T[:, ind].data
        )
        name = f"L_{layer}_H_{head}_O_weight"
        components_copy[name] = copy.deepcopy(
            model.layers[layer].self_attn.o_proj.weight.T[ind, :].data
        )

    return components_copy


def exchange_edit(
    model,
    config,
    LayerHeadPairs,
    component="QK",
    replace_from_outside_list=False,
    rnd_seed=42,
):
    """
    exchange_edit shuffles the QK components (or OV components) of heads within a list of heads
    Arguments:
        model: a transformer model, e.g., GPT2Model.from_pretrained("gpt2")
        config: model configuration
        LayerHeadPairs: a list of (layer, head) pairs indicating heads that will be shuffled
        component: if 'QK' then shuffle the weight and bias (namely W_qk and b_qk) with those in other heads;
                if 'OV' then shuffle the weight and bias (namely W_ov and b_ov) with those in other heads;
        replace_from_outside_list: if false (default), simply shuffle heads in LayerHeadPairs;
                if True, instead of shuffling heads, replacing heads in LayerHeadPairs by randomly
                sampled heads in the entire model, used for caculating the random baseline.
        rnd_seed: random seed
    Returns:
        model_edit: a transformer similar to model but weights are shuffled
        LayerHeadPairs0: useful if replace_from_outside_list is True, in which randomly sampled layer head pairs are returned
    """
    assert not hasattr(
        model, "transformer"
    ), "Don't use transformer than includes a separate LMHead"
    assert (
        component == "QK" or component == "OV"
    ), "component should be either 'QK' or 'OV'"

    K = len(LayerHeadPairs)
    num_layer = config.num_hidden_layers
    d_model = config.hidden_size
    num_heads = config.num_key_value_heads
    d_head = d_model // num_heads

    LayerHeadPairs0 = _layer_head_pairs(
        config=config,
        LayerHeadPairs=LayerHeadPairs,
        replace_from_outside_list=replace_from_outside_list,
    )
    components_copy = _components_to_copy(
        model=model, config=config, LayerHeadPairs0=LayerHeadPairs0
    )

    # a random permutation
    perm = torch.randperm(K)
    # edit model
    if component == "QK":
        for j in range(K):
            layer, head = LayerHeadPairs[j][0], LayerHeadPairs[j][1]
            layer_perm, head_perm = (
                LayerHeadPairs0[perm[j]][0],
                LayerHeadPairs0[perm[j]][1],
            )

            ind = range(head * d_head, head * d_head + d_head)
            name = f"L_{layer_perm}_H_{head_perm}_Q_weight"
            model.layers[layer].self_attn.q_proj.weight.T.data[:, ind] = (
                components_copy[name]
            )
            name = f"L_{layer_perm}_H_{head_perm}_K_weight"
            model.layers[layer].self_attn.k_proj.weight.T.data[:, ind] = (
                components_copy[name]
            )

    elif component == "OV":
        for j in range(K):
            layer, head = LayerHeadPairs[j][0], LayerHeadPairs[j][1]
            layer_perm, head_perm = (
                LayerHeadPairs0[perm[j]][0],
                LayerHeadPairs0[perm[j]][1],
            )
            ind = range(head * d_head, head * d_head + d_head)
            name = f"L_{layer_perm}_H_{head_perm}_V_weight"
            model.layers[layer].self_attn.v_proj.weight.T.data[:, ind] = (
                components_copy[name]
            )
            name = f"L_{layer_perm}_H_{head_perm}_O_weight"
            model.layers[layer].self_attn.o_proj.weight.T.data[ind, :] = (
                components_copy[name]
            )

    return model, LayerHeadPairs0


################################# Intervention: weight projection ###################################


def calc_common_subspace(
    W_all,
    LayerHeadPairs,
    K=None,
):
    """
    calc_common_subspace calculates the common subspace $\mathcal{V}$ that is crucial for induction heads from a list of
        QK weight matrices; it stacks all componnent weight matrices [W_1; W_2; ... W_K] and applies SVD
    Arguments:
        W_all is a tensor of shape (num_layer, num_heads, 4, d_model, d_head) extracted from pretrained models; see definition in related function
        LayerHeadPairs: a list of (layer, head) pairs used for calculating common subspace, normally just top IH list
        rank: the subspace rank
        K: the number of componnent weight matrices used to calculate common subspace; if None, equals length of LayerHeadPair
    Returns:
        V: a matrix of size (d_model, rank), whose columns are orthonormal basis of the common subspace
        P: a projection matrix of size (d_model, d_model) defined by V @ V.T
    """

    if K is None:
        K = len(LayerHeadPairs)
    num_layer, num_heads, _, d_model, d_head = W_all.shape

    W_qk_all = np.zeros((K, d_model, d_model))
    for i in range(K):
        Layer, Head = LayerHeadPairs[i][0], LayerHeadPairs[i][1]
        W_qk = W_all[Layer, Head, 0] @ W_all[Layer, Head, 1].T
        W_qk_all[i] = W_qk.numpy(force=True)

    U, S, Vt_common = custom_svd(W_qk_all.reshape(-1, d_model))  # takes long
    return Vt_common


def projection_edit(model, config, LayerHeadPairs, P, component="QK"):
    """
    projection_edit projects out several QK or OV weight matrices in a list
    Arguments:
        model: a transformer model, e.g., GPT2Model.from_pretrained("gpt2")
        config: model configuration
        LayerHeadPairs: a list of (layer, head) pairs indicating heads that will projected
        component: if 'QK' then the W_k matrix will be projected, which effectively removes a subspace from W_qk
                if 'OV' then W_o matrix will be projected, which effectively removes a subspace from W_ov
    Returns:
        model_edit: a transformer similar to model but weights are shuffled
    """
    assert not hasattr(
        model, "transformer"
    ), "Don't use transformer than includes a separate LMHead"
    assert (
        component == "QK" or component == "OV"
    ), "component should be either 'QK' or 'OV'"

    K = len(LayerHeadPairs)
    d_model = config.hidden_size
    num_heads = config.num_key_value_heads
    d_head = d_model // num_heads

    for j in range(K):
        layer, head = LayerHeadPairs[j][0], LayerHeadPairs[j][1]
        if component == "QK":
            ind = range(head * d_head, head * d_head + d_head)
            # W = copy.deepcopy(
            #     model.layers[layer].self_attn.k_proj.weight.T.data[:, ind]
            # )
            # W = torch.tensor(P, dtype=torch.bfloat16, device="cuda") @ W
            # model.layers[layer].self_attn.k_proj.weight.T.data[:, ind] = W
            W = copy.deepcopy(model.layers[layer].self_attn.k_proj.weight.data.T)
            W[:, ind] = torch.tensor(P, dtype=torch.bfloat16, device="cuda") @ W[:, ind]
            with torch.no_grad():
                model.layers[layer].self_attn.k_proj.weight.copy_(W.T)

        elif component == "OV":
            ind = range(head * d_head, head * d_head + d_head)
            W = copy.deepcopy(
                model.layers[layer].self_attn.o_proj.weight.T.data[ind, :]
            )
            W = W @ torch.tensor(P, dtype=torch.bfloat16, device="cuda")
            model.layers[layer].self_attn.o_proj.weight.T.data[ind, :] = W

    return model


def project_exp(
    config,
    W_all,
    seg_len,
    vocab_size,
    LayerHeadPair0,
    LayerHeadPair,
    save_dir,
    component="QK",
    project_out=True,
    rep=3,
    batch_size=50,
    max_rank=100,
    step=5,
):
    """
    project_exp projects the heads indicated in a list of (layer, head) of a model, and then evaluates the prediction
        probabilities of the edited model based on random repeated tokens;
        See a similar function "shuffle_exp".
    Arguments:
        model is a transformer model, e.g., GPT2Model.from_pretrained("gpt2"), no LMHead should be in the model
        config is the configuration of the model
        W_all is a tensor of shape (num_layer, num_heads, 4, d_model, d_head) extracted from pretrained models; see definition in related function
        seg_len is the repeating segment length when drawing random tokens
        vocab_size is the vocabulary size
        LayerHeadPair0: a list of (layer, head) pairs used to calculate common subspace
        LayerHeadPair: a list of (layer, head) pairs to perform model edits namely projection; often larger than LayerHeadPair0
        save_dir: save_dir is the folder for saving plots
        component: either 'QK or 'OV' that indicates which component in a head to be projected
        project_out: if True, project OUT common subspace from weights; if False, only keep common subspace
        rep: the number of repetition of a random token segment
        batch_size: number of sequences evaluated by the edited model; probabilities will be averaged
        max_rank: iterating the experiment for rank in range(1,max_rank+1,step), i.e., projecting subspaces of varying ranks
        step: the increment in iterations
    Returns:
        probs: a dictionary containing the probabilities of original model, edited model, and a random baseline model
            each is a 2-array of shape (batch_size, *)
        errs: similar to probs, but gives errors of top predicted tokens

    """
    T0 = seg_len  # repeating segment length
    T_cnt = T0 + T0 // 2  # starting position when calculating accuracy
    K0, K = len(LayerHeadPair0), len(LayerHeadPair)
    create_folder(save_dir)

    from transformers import gemmaForCausalLM

    gemma = gemmaForCausalLM.from_pretrained(
        "meta-gemma/gemma-2-7b-hf",
        output_attentions=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    print_gpu_mem_usage()

    # draw batch of random tokens and make repetitions
    sample_int = np.random.randint(
        low=0, high=vocab_size, size=batch_size * T0
    ).reshape(batch_size, T0)
    sample_int = np.concatenate(tuple([sample_int] * rep), axis=1)
    input_ids = torch.Tensor(sample_int).long()
    correct_next_token_ids = torch.Tensor(
        np.concatenate((sample_int[:, 1:], sample_int[:, :1]), axis=1)
    ).long()

    # shuffle heads
    probs = {}
    errs = {}
    with torch.no_grad():
        for i in range(batch_size // 5):
            cur_batch_input_ids = input_ids.cuda()[i * 5 : i * 5 + 5]
            if i == 0:
                out = gemma.model(cur_batch_input_ids)[0]
            else:
                out = torch.concat([out, gemma.model(cur_batch_input_ids)[0]])

    res, indices = pred_probs(out, gemma.lm_head.weight, correct_next_token_ids)
    probs["original"] = res[:, T_cnt:-1, 2]
    errs["original"] = (
        indices[:, T_cnt:-1, 0] != correct_next_token_ids[:, T_cnt:-1:].numpy()
    )

    del gemma, out
    torch.cuda.empty_cache()
    gc.collect()
    print_gpu_mem_usage()

    print("Start subspace")
    Vt_common = calc_common_subspace(W_all, LayerHeadPair0)

    for rank in range(1, max_rank + 1, step):
        print(rank)
        V = Vt_common[:rank, :].T
        P = V @ V.T
        P = np.eye(P.shape[0]) - P if project_out else P

        gemma = gemmaForCausalLM.from_pretrained(
            "meta-gemma/gemma-2-7b-hf",
            output_attentions=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        gemma.eval()

        model_edit = projection_edit(
            gemma.model,
            config,
            LayerHeadPair,
            P,
            component=component,
        )
        with torch.no_grad():
            for i in range(batch_size // 5):
                cur_batch_input_ids = input_ids.cuda()[i * 5 : i * 5 + 5]
                cur_out = model_edit(cur_batch_input_ids)[0]
                if i == 0:
                    out = cur_out
                else:
                    out = torch.concat([out, cur_out])

        res, indices = pred_probs(out, gemma.lm_head.weight, correct_next_token_ids)
        probs[rank] = res[:, T_cnt:-1, 2]
        errs[rank] = (
            indices[:, T_cnt:-1, 0] != correct_next_token_ids[:, T_cnt:-1:].numpy()
        )

        del model_edit, gemma, out
        torch.cuda.empty_cache()
        gc.collect()
        print_gpu_mem_usage()

    # make plots
    fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 6 * 1))
    save_to = os.path.join(
        save_dir, component + f"_proj_out_{project_out}_K0_{K0}_K_{K}"
    )
    ave_probs = [np.mean(vals) for vals in probs.values()]
    ave_errs = [np.mean(vals) for vals in errs.values()]

    axs[0].plot(range(1, max_rank + 1, step), ave_probs[1:], "-o", label="projected")
    axs[0].axhline(y=ave_probs[0], linestyle="dashed", label="original")
    axs[1].plot(range(1, max_rank + 1, step), ave_errs[1:], "-o", label="projected")
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

    return probs, errs
