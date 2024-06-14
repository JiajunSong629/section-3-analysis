from transformers import LlamaForCausalLM
import torch
import numpy as np
import os
import gc
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import json


def create_folder(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


def print_gpu_mem_usage():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device=None)
    print(
        f"gpu used {torch.cuda.max_memory_allocated(device=None)/ np.power(2, 30): .3f}G memory"
    )


def load_llama():
    llama = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        output_attentions=True,
    )
    llama.eval()
    print_gpu_mem_usage()
    return llama


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
    _, pred_next_token_ids = torch.topk(probs, dim=-1, k=1)

    return probs, pred_next_token_ids


def projection_edit(model, LayerHeadPairs, P, component):
    config = model.config
    d_model = config.hidden_size
    num_heads = config.num_key_value_heads
    d_head = d_model // num_heads

    for layer, head in LayerHeadPairs:
        if component == "QK":
            ind = range(head * d_head, head * d_head + d_head)
            W = copy.deepcopy(model.model.layers[layer].self_attn.k_proj.weight.data.T)

            W[:, ind] = torch.tensor(P, device="cuda", dtype=torch.bfloat16) @ W[:, ind]
            with torch.no_grad():
                model.model.layers[layer].self_attn.k_proj.weight.copy_(W.T)

        elif component == "OV":
            ind = range(head * d_head, head * d_head + d_head)
            W = copy.deepcopy(model.model.layers[layer].self_attn.o_proj.weight.data.T)

            W[ind, :] = W[ind, :] @ torch.tensor(P, device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                model.model.layers[layer].self_attn.o_proj.weight.copy_(W.T)

    return model


def projection_edit_by_layer(model, LayerHeadPairs, P_early, P_late, component):
    config = model.config
    d_model = config.hidden_size
    num_heads = config.num_key_value_heads
    d_head = d_model // num_heads

    for layer, head in LayerHeadPairs:
        if layer <= 12:
            P = P_early
        else:
            P = P_late

        if component == "QK":
            ind = range(head * d_head, head * d_head + d_head)
            W = copy.deepcopy(model.model.layers[layer].self_attn.k_proj.weight.data.T)

            W[:, ind] = torch.tensor(P, device="cuda", dtype=torch.bfloat16) @ W[:, ind]
            with torch.no_grad():
                model.model.layers[layer].self_attn.k_proj.weight.copy_(W.T)

        elif component == "OV":
            ind = range(head * d_head, head * d_head + d_head)
            W = copy.deepcopy(model.model.layers[layer].self_attn.o_proj.weight.data.T)

            W[ind, :] = W[ind, :] @ torch.tensor(P, device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                model.model.layers[layer].self_attn.o_proj.weight.copy_(W.T)

    return model


def plot(
    probs,
    errs,
    max_rank,
    step,
    save_to,
):
    # make plots
    fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 6 * 1))
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


def plot_attentions(attentions, LayerHeadPair, save_dir, fname_prefix):
    for layer, head in LayerHeadPair:
        sns.heatmap(attentions[layer][head].float().numpy(force=True)[1:, 1:])
        plt.savefig(f"{save_dir}/{fname_prefix}_L{layer}H{head}.png")
        plt.close()


def main(
    config,
    LayerHeadPair,
    save_dir,
    Vt_common,
    component="QK",
    project_out=True,
    T0=25,
    rep=3,
    batch_size=50,
    max_rank=100,
    ignore_segment=1,
    ignore_burning=4,
    step=5,
    figname=None,
):

    np.random.seed(2024)

    T_range = range(ignore_segment * T0 + ignore_burning, rep * T0 - 1)
    create_folder(save_dir)

    input_ids = make_input_ids(
        batch_size=batch_size,
        T0=T0,
        rep=rep,
        vocab_size=config.vocab_size,
    )
    probs = {}
    errs = {}

    # original
    llama = load_llama()
    pred_prob, pred_next_token_ids = inference_probs_and_errs(
        model=llama,
        input_ids=input_ids,
    )
    probs["original"] = pred_prob[:, T_range].numpy(force=True)
    errs["original"] = (input_ids[:, 1:] != pred_next_token_ids[:, :-1, 0]).numpy(
        force=True
    )[:, T_range]

    print(errs["original"].mean())

    del llama

    for rank in range(step, max_rank + 1, step):
        print(rank)

        V = Vt_common[:rank, :].T
        P = V @ V.T
        P = np.eye(P.shape[0]) - P if project_out else P

        llama = load_llama()
        llama_edit = projection_edit(
            llama,
            LayerHeadPair,
            P,
            component,
        )
        pred_prob, pred_next_token_ids = inference_probs_and_errs(
            model=llama_edit,
            input_ids=input_ids,
        )
        probs[rank] = pred_prob[:, T_range].numpy(force=True)
        errs[rank] = (input_ids[:, 1:] != pred_next_token_ids[:, :-1, 0]).numpy(
            force=True
        )[:, T_range]

        print(errs[rank].mean())

        del llama, llama_edit

    plot(
        probs=probs,
        errs=errs,
        max_rank=max_rank,
        step=step,
        save_to=os.path.join(save_dir, figname),
    )

    return probs, errs


def extract_IH_list(filter_later_layers=True):
    lines = open("simple_visz/induction_head_scores.txt").readlines()
    IH_list = []
    for line in lines[1:]:
        layerhead = line.split(":")[0]
        head = layerhead.split("Head")[1]
        layer = layerhead.split("Head")[0].split("Layer")[1]
        if int(layer) < 16 or (not filter_later_layers):
            IH_list.append([int(layer), int(head)])

    return IH_list


if __name__ == "__main__":
    from transformers import LlamaConfig

    configuration = LlamaConfig()

    # extract Shifting list
    lines = open("simple_visz/shifting_head_scores.txt").readlines()
    Shifting_list = []
    for line in lines[1:]:
        layerhead = line.split(":")[0]
        head = layerhead.split("Head")[1]
        layer = layerhead.split("Head")[0].split("Layer")[1]
        Shifting_list.append([int(layer), int(head)])

    ######################################################################

    # IH_list = extract_IH_list()
    # IH_list = [(L, H) for L, H in IH_list if L <= 12][:10]

    # probs, errs = main(
    #     config=configuration,
    #     LayerHeadPair=IH_list,
    #     save_dir="Figs/project_experiment_test",
    #     component="QK",
    #     project_out=True,
    #     T0=25,
    #     rep=3,
    #     ignore_burning=4,
    #     ignore_segment=1,
    #     batch_size=50,
    #     max_rank=100,
    #     step=10,
    #     Vt_common=np.load("Vt_common_early10.npy"),
    #     figname="QK_proj_out_True_K10_only_early_layers.png",
    # )

    # IH_list = extract_IH_list()
    # IH_list = [(L, H) for L, H in IH_list if L > 12][:10]

    # probs, errs = main(
    #     config=configuration,
    #     LayerHeadPair=IH_list,
    #     save_dir="Figs/project_experiment_test",
    #     component="QK",
    #     project_out=True,
    #     T0=25,
    #     rep=3,
    #     ignore_burning=4,
    #     ignore_segment=1,
    #     batch_size=50,
    #     max_rank=100,
    #     step=10,
    #     Vt_common=np.load("Vt_common_late10.npy"),
    #     figname="QK_proj_out_True_K10_only_late_layers.png",
    # )

    # IH_list = extract_IH_list()
    # IH_list = [(L, H) for L, H in IH_list if L <= 12][:10]

    # probs, errs = main(
    #     config=configuration,
    #     LayerHeadPair=IH_list,
    #     save_dir="Figs/project_experiment_test",
    #     component="QK",
    #     project_out=True,
    #     T0=25,
    #     rep=3,
    #     ignore_burning=4,
    #     ignore_segment=1,
    #     batch_size=50,
    #     max_rank=100,
    #     step=10,
    #     Vt_common=np.load("Vt_common_R25_early10.npy"),
    #     figname="QK_proj_out_True_K10_only_early_layers_R25.png",
    # )

    # IH_list = extract_IH_list()
    # IH_list = [(L, H) for L, H in IH_list if L <= 12][:10]

    # probs, errs = main(
    #     config=configuration,
    #     LayerHeadPair=IH_list,
    #     save_dir="Figs/project_experiment_test",
    #     component="QK",
    #     project_out=True,
    #     T0=25,
    #     rep=3,
    #     ignore_burning=4,
    #     ignore_segment=1,
    #     batch_size=50,
    #     max_rank=100,
    #     step=10,
    #     Vt_common=np.load("Vt_common_R25.npy"),
    #     figname="QK_proj_out_True_K10_only_early_R25.png",
    # )

    # IH_list = extract_IH_list()
    # IH_list = [(L, H) for L, H in IH_list if L <= 12][:30]

    # probs, errs = main(
    #     config=configuration,
    #     LayerHeadPair=IH_list,
    #     save_dir="Figs/project_experiment_test",
    #     component="QK",
    #     project_out=True,
    #     T0=25,
    #     rep=3,
    #     ignore_burning=4,
    #     ignore_segment=1,
    #     batch_size=50,
    #     max_rank=100,
    #     step=10,
    #     Vt_common=np.load("Vt_common_R25.npy"),
    #     figname="QK_proj_out_True_K30_only_early_R25.png",
    # )

    for K in [10, 30, 50]:
        IH_list = extract_IH_list()[:K]

        probs, errs = main(
            config=configuration,
            LayerHeadPair=IH_list,
            save_dir="Figs/project",
            component="QK",
            project_out=True,
            T0=25,
            rep=3,
            ignore_burning=4,
            ignore_segment=1,
            batch_size=50,
            max_rank=100,
            step=5,
            Vt_common=np.load("Vt_common_R25.npy"),
            figname=f"QK_proj_out_True_K{K}_R25.png",
        )

    # IH_list = extract_IH_list()
    # IH_list = IH_list[:10]

    # probs, errs = main(
    #     config=configuration,
    #     LayerHeadPair=IH_list,
    #     save_dir="Figs/project",
    #     component="QK",
    #     project_out=True,
    #     T0=25,
    #     rep=3,
    #     ignore_burning=4,
    #     ignore_segment=1,
    #     batch_size=50,
    #     max_rank=100,
    #     step=10,
    #     Vt_common=np.load("Vt_common.npy"),
    #     figname="QK_proj_out_True_K10.png",
    # )

    # IH_list = extract_IH_list()
    # IH_list = IH_list[:30]

    # probs, errs = main(
    #     config=configuration,
    #     LayerHeadPair=IH_list,
    #     save_dir="Figs/project",
    #     component="QK",
    #     project_out=True,
    #     T0=25,
    #     rep=3,
    #     ignore_burning=4,
    #     ignore_segment=1,
    #     batch_size=50,
    #     max_rank=100,
    #     step=10,
    #     Vt_common=np.load("Vt_common.npy"),
    #     figname="QK_proj_out_True_K30.png",
    # )
