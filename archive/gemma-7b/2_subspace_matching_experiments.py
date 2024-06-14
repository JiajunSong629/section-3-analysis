import torch

W_all = torch.load("W_all.pt")

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

####################################################################

from analysis_utils import subspace_matching

K = 20
s_match1, match_baseline1 = subspace_matching(
    W_all, "Figs/subspace_matching_IH", IH_list=IH_list[:K], ranks=[2, 3, 5, 10]
)
s_match2, match_baseline2 = subspace_matching(
    W_all,
    "Figs/subspace_matching_SH",
    Shifting_list=Shifting_list[:K],
    ranks=[2, 3, 5, 10],
)
s_match3, match_baseline3 = subspace_matching(
    W_all,
    "Figs/subspace_matching_IH_SH",
    IH_list=IH_list[:K],
    Shifting_list=Shifting_list[:K],
    ranks=[2, 3, 5, 10],
)
