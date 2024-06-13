from analysis_utils import shuffle_exp
from transformers import GPT2Model
from transformers import GPT2Config

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


model = GPT2Model.from_pretrained("gpt2", output_attentions=True)
configuration = model.config

seg_len = 25  # repeating segment length
for K in [5, 10, 15, 20]:
    K0, K1 = K, K
    save_dir = "Figs/shuffle"
    probs_QK, errs_QK = shuffle_exp(
        model,
        configuration,
        seg_len,
        configuration.vocab_size,
        IH_list[:K0],
        save_dir,
        component="QK",
    )
    probs_OV, errs_OV = shuffle_exp(
        model,
        configuration,
        seg_len,
        configuration.vocab_size,
        Shifting_list[:K1],
        save_dir,
        component="OV",
    )
