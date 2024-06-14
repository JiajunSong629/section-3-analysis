import torch
from transformers import GemmaForCausalLM, GemmaConfig


configuration = GemmaConfig().from_pretrained("google/gemma-2b")
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


##############################################################


from analysis_utils import project_exp

save_dir = "Figs/project"
seg_len = 20
probs, errs = project_exp(
    configuration,
    W_all,
    seg_len,
    configuration.vocab_size,
    IH_list[:10],
    IH_list[:30],
    save_dir,
    component="QK",
    project_out=True,
    max_rank=100,
    step=5,
)


probs, errs = project_exp(
    configuration,
    W_all,
    seg_len,
    configuration.vocab_size,
    IH_list[:10],
    IH_list[:30],
    save_dir,
    component="QK",
    project_out=False,
    max_rank=100,
    step=5,
)

probs, errs = project_exp(
    configuration,
    W_all,
    seg_len,
    configuration.vocab_size,
    IH_list[:10],
    Shifting_list[:30],
    save_dir,
    component="OV",
    project_out=True,
    max_rank=100,
    step=5,
)


probs, errs = project_exp(
    configuration,
    W_all,
    seg_len,
    configuration.vocab_size,
    IH_list[:10],
    Shifting_list[:30],
    save_dir,
    component="OV",
    project_out=False,
    max_rank=100,
    step=5,
)
