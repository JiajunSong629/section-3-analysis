import torch
import os
from analysis_utils import qkov_matching_summary, qkov_matching

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

######################################################################


def create_folder(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


create_folder("Figs")
create_folder("Figs/diagonal")

W_all = torch.load("W_all.pt")


for K0, K1 in [(10, 10), (30, 30)]:
    scores = qkov_matching_summary(
        W_all, IH_list[:K0], Shifting_list[:K1], "Figs/diagonal"
    )
