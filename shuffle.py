import copy
import torch
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import gc
import fire
from utils import (
    make_input_ids,
    create_folder,
    inference_probs_and_errs,
    load_model,
    get_configs,
    get_qkov_weight,
)


def collect_random_layer_head_pairs(model_name, K):
    num_layer, num_head, d_model, d_head, _ = get_configs(model_name)
    LH = []
    while len(LH) < K:
        layer = np.random.randint(low=1, high=num_layer - 1)
        head = np.random.randint(low=0, high=num_head)
        if [layer, head] not in LH:
            LH.append([layer, head])
    return LH


def collect_components_to_copy(model, model_name, layer_head_pairs):
    components_copy = {}
    for ilayer, ihead in layer_head_pairs:
        for name in ["Q", "K", "O", "V"]:
            component_name = f"L_{ilayer}_H_{ihead}_{name}_weight"
            components_copy[component_name] = copy.deepcopy(
                get_qkov_weight(model, model_name, ilayer, ihead, name.lower()).data
            )

    return components_copy


def exchange_edit(
    model,
    model_name,
    layer_head_pairs,
    component="QK",
    type="original",
):
    K = len(layer_head_pairs)

    if type == "original":
        return model
    elif type == "random baseline":
        layer_head_pairs_1 = collect_random_layer_head_pairs(model_name, K)
    else:
        perm = torch.randperm(K)
        layer_head_pairs_1 = [layer_head_pairs[perm[j]] for j in range(K)]

    components_copy = collect_components_to_copy(
        model=model,
        model_name=model_name,
        layer_head_pairs=layer_head_pairs_1,
    )

    if component == "QK":
        for (layer, head), (layer_perm, head_perm) in zip(
            layer_head_pairs, layer_head_pairs_1
        ):
            for name in ["Q", "K"]:
                component_name = f"L_{layer_perm}_H_{head_perm}_{name}_weight"
                w = get_qkov_weight(
                    model,
                    model_name,
                    layer,
                    head,
                    component=name.lower(),
                )
                # print("BEFORE", w.sum())
                w.copy_(components_copy[component_name])
                # print("TMP", w.sum())
                # print(
                #     "AFTER",
                #     get_qkov_weight(
                #         model, model_name, layer, head, component=name.lower()
                #     ).sum(),
                # )

    elif component == "OV":
        for (layer, head), (layer_perm, head_perm) in zip(
            layer_head_pairs, layer_head_pairs_1
        ):
            for name in ["O", "V"]:
                component_name = f"L_{layer_perm}_H_{head_perm}_{name}_weight"
                w = get_qkov_weight(
                    model, model_name, layer, head, component=name.lower()
                )
                w.copy_(components_copy[component_name])

    return model


def plot(probs, errs, save_to):
    fig, axs = plt.subplots(2, 2, figsize=(6 * 2, 6 * 2))

    for name in probs:
        ave_probs = np.mean(probs[name], axis=0)
        ave_errs = np.mean(errs[name], axis=0)
        axs[0, 0].plot(range(len(ave_probs)), ave_probs, "-o", label=name)
        axs[0, 1].plot(range(len(ave_errs)), ave_errs, "-o", label=name)

        axs[1, 0].hist(ave_probs, label=name, alpha=0.3)
        axs[1, 1].hist(ave_errs, label=name, alpha=0.3)

    for j in range(2):
        titles = [f"Pred {a} under shuffling" for a in ["probs", "errs"]]
        axs[0, j].set_xlabel("Token position", weight="bold")
        axs[0, j].set_ylabel("Target token pred probs/errs", weight="bold")
        axs[0, j].set_title(titles[j], weight="bold")

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()

    plt.savefig(save_to, bbox_inches="tight")
    plt.close()


def shuffle_exp(
    model_name,
    batch_size,
    seg_len,
    rep,
    ignore_segment,
    ignore_burning,
    layer_head_pairs,
    component,
    n_exp,
    save_to,
):

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

    for name in ["original", "random baseline", "shuffle"]:
        print("NAME", name)

        for _ in range(n_exp):

            model = load_model(model_name)
            model_edit = exchange_edit(
                model=model,
                model_name=model_name,
                layer_head_pairs=layer_head_pairs,
                component=component,
                type=name,
            )

            prob, err = inference_probs_and_errs(model_edit, input_ids)
            probs[name].append(prob[:, T_range])
            errs[name].append(err[:, T_range])

            del model_edit, model
            torch.cuda.empty_cache()
            gc.collect()

        probs[name] = np.vstack(probs[name])
        errs[name] = np.vstack(errs[name])

        print("ERR", errs[name].mean())
        print("PROB", probs[name].mean())

    plot(probs, errs, save_to)

    return probs, errs


def main(
    model_name,
    K=5,
    n_exp=1,
    batch_size=50,
    seg_len=25,
    rep=3,
    ignore_segment=1,
    ignore_burning=4,
):

    IH = torch.load(f"checkpoints/{model_name}/IH.pt")
    PTH = torch.load(f"checkpoints/{model_name}/PTH.pt")

    create_folder("Figs")
    create_folder(f"Figs/shuffle")

    print("K", K)
    shuffle_exp(
        model_name=model_name,
        batch_size=batch_size,
        seg_len=seg_len,
        rep=rep,
        n_exp=n_exp,
        layer_head_pairs=IH[:K],
        component="QK",
        ignore_burning=ignore_burning,
        ignore_segment=ignore_segment,
        save_to=f"Figs/shuffle/{model_name}_QK_{K}_T0_{seg_len}.png",
    )

    shuffle_exp(
        model_name=model_name,
        batch_size=batch_size,
        seg_len=seg_len,
        rep=rep,
        layer_head_pairs=PTH[:K],
        component="OV",
        n_exp=n_exp,
        ignore_burning=ignore_burning,
        ignore_segment=ignore_segment,
        save_to=f"Figs/shuffle/{model_name}_OV_{K}_T0_{seg_len}.png",
    )


if __name__ == "__main__":
    fire.Fire(main)
