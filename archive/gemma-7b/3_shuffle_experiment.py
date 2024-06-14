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
from analysis_utils import (
    create_folder,
    inference_probs_and_errs,
    make_input_ids,
    load_gemma,
    exchange_edit,
)


# extract IH list
lines = open("simple_visz/induction_head_scores.txt").readlines()
IH_list = []
for line in lines[1:]:
    layerhead = line.split(":")[0]
    head = layerhead.split("Head")[1]
    layer = layerhead.split("Head")[0].split("Layer")[1]
    IH_list.append([int(layer), int(head)])


# extract Shifting list
lines = open("simple_visz/shifting_head_scores.txt").readlines()
Shifting_list = []
for line in lines[1:]:
    layerhead = line.split(":")[0]
    head = layerhead.split("Head")[1]
    layer = layerhead.split("Head")[0].split("Layer")[1]
    Shifting_list.append([int(layer), int(head)])

##########################################################################


def shuffle_exp(
    config,
    seg_len,
    vocab_size,
    LayerHeadPair,
    save_dir,
    component="QK",
    rep=3,
    batch_size=50,
    max_heads=50,
    ignore_segment=1,
    ignore_burning=4,
):

    T_range = range(ignore_segment * seg_len + ignore_burning, rep * seg_len - 1)
    K = min(len(LayerHeadPair), max_heads)
    create_folder(save_dir)

    gemma = load_gemma()

    print(gemma)
    # shuffle heads
    probs = {}
    errs = {}
    names = ["original", "shuffle", "random baseline"]

    # draw batch of random tokens and make repetitions
    input_ids = make_input_ids(
        batch_size=batch_size,
        T0=seg_len,
        rep=rep,
        vocab_size=vocab_size,
    )

    pred_prob, pred_next_token_ids = inference_probs_and_errs(
        model=gemma,
        input_ids=input_ids,
    )
    probs["original"] = pred_prob[:, T_range].numpy(force=True)
    errs["original"] = (input_ids[:, 1:] != pred_next_token_ids[:, :-1, 0]).numpy(
        force=True
    )[:, T_range]

    del gemma, out
    torch.cuda.empty_cache()
    gc.collect()

    for baseline in [False, True]:
        gemma = GemmaForCausalLM.from_pretrained(
            "google/gemma-7b",
            output_attentions=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        gemma.eval()

        gemma_edit, _ = exchange_edit(
            gemma,
            config,
            LayerHeadPair[:K],
            component=component,
            replace_from_outside_list=baseline,
        )

        pred_prob, pred_next_token_ids = inference_probs_and_errs(
            model=gemma_edit,
            input_ids=input_ids,
        )
        probs["original"] = pred_prob[:, T_range].numpy(force=True)
        errs["original"] = (input_ids[:, 1:] != pred_next_token_ids[:, :-1, 0]).numpy(
            force=True
        )[:, T_range]

        del model_edit, gemma, out
        torch.cuda.empty_cache()
        gc.collect()
        # print_gpu_mem_usage()

        print(probs[names[baseline + 1]].shape, errs[names[baseline + 1]].shape)

    # make plots
    fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 6 * 1))
    save_to = os.path.join(save_dir, component + f"_K_{K}_T0_{seg_len}")

    for k, name in enumerate(names):
        ave_probs = np.mean(probs[name], axis=0)
        ave_errs = np.mean(errs[name], axis=0)
        axs[0].plot(range(len(ave_probs)), ave_probs, "-o", label=name)
        axs[1].plot(range(len(ave_errs)), ave_errs, "-o", label=name)

    for j in range(2):
        titles = [f"Pred {a} under shuffling" for a in ["probs", "errs"]]
        axs[j].set_xlabel("Token position", weight="bold")
        axs[j].set_ylabel("Target token pred probs/errs", weight="bold")
        axs[j].set_title(titles[j], weight="bold")
        axs[j].set_ylim(0, 1)

    axs[0].legend()
    plt.savefig(save_to, bbox_inches="tight")
    plt.close()

    return probs, errs


seg_len = 25  # repeating segment length
for K in [5, 10, 15, 20, 25, 30]:
    print("#" * 50)
    print(K)
    print("#" * 50)

    K0, K1 = K, K
    save_dir = "Figs/shuffle"
    probs_QK, errs_QK = shuffle_exp(
        LlamaConfig(),
        seg_len,
        LlamaConfig().vocab_size,
        IH_list[:K0],
        save_dir,
        component="QK",
        batch_size=50,
    )

    probs_OV, errs_OV = shuffle_exp(
        LlamaConfig(),
        seg_len,
        LlamaConfig().vocab_size,
        Shifting_list[:K1],
        save_dir,
        component="OV",
        batch_size=50,
    )
