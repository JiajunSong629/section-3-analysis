import torch
from transformers import set_seed
from transformers import GPT2Model

set_seed(2024)
from analysis_utils import project_exp

model = GPT2Model.from_pretrained("gpt2-xl", output_attentions=True)
configuration = model.config
num_layer = 12
num_heads = 12
d_model = 768
d_head = d_model // num_heads

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

vocab_size = configuration.vocab_size
save_dir = "Figs/project"
seg_len = 25
probs, errs = project_exp(
    configuration,
    W_all,
    seg_len,
    vocab_size,
    IH_list[:10],
    IH_list[:30],
    save_dir,
    component="QK",
    project_out=True,
)
probs, errs = project_exp(
    configuration,
    W_all,
    seg_len,
    vocab_size,
    IH_list[:10],
    IH_list[:30],
    save_dir,
    component="QK",
    project_out=False,
)
probs, errs = project_exp(
    configuration,
    W_all,
    seg_len,
    vocab_size,
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
    vocab_size,
    IH_list[:10],
    Shifting_list[:30],
    save_dir,
    component="OV",
    project_out=False,
    max_rank=100,
    step=5,
)
