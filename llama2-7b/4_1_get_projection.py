import torch
import numpy as np
from analysis_utils import custom_svd
from Rotary_test import calc_rotary_R_mat

W_all = torch.load("W_all.pt")  # torch.float32


# extract IH list
def extract_IH_list():
    lines = open("simple_visz/induction_head_scores.txt").readlines()
    IH_list = []
    for line in lines[1:]:
        layerhead = line.split(":")[0]
        head = layerhead.split("Head")[1]
        layer = layerhead.split("Head")[0].split("Layer")[1]
        IH_list.append([int(layer), int(head)])

    return IH_list


def calc_V(IH_list, save_to, K=10, use_R=False, max_rel_dist=0):
    num_layer, num_heads, _, d_model, d_head = W_all.shape

    W_qk_all = np.zeros((K, d_model, d_model))
    for i in range(K):
        Layer, Head = IH_list[i][0], IH_list[i][1]
        if use_R:
            W_qk = (
                W_all[Layer, Head, 0]
                @ calc_rotary_R_mat(
                    d_head=d_head,
                    max_seq_len=60,
                    max_rel_dist=max_rel_dist,
                )[-1]
                @ W_all[Layer, Head, 1].T
            )

        else:
            W_qk = W_all[Layer, Head, 0] @ W_all[Layer, Head, 1].T

        W_qk_all[i] = W_qk.numpy(force=True)

    U, S, Vt_common = custom_svd(W_qk_all.reshape(-1, d_model))

    np.save(save_to, Vt_common)

    return Vt_common


IH_list = extract_IH_list()[:10]
calc_V(IH_list=IH_list, max_rel_dist=25, use_R=True, save_to="Vt_common_R25.npy")
calc_V(IH_list=IH_list, max_rel_dist=25, use_R=False, save_to="Vt_common.npy")

# IH_list = extract_IH_list()
# early_IH_list = [(L, H) for L, H in IH_list if L <= 12][:10]
# late_IH_list = [(L, H) for L, H in IH_list if L > 12][:10]

# calc_V(early_IH_list, "Vt_common_early10.npy")
# calc_V(late_IH_list, "Vt_common_late10.npy")
