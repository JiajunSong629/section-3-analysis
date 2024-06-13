from transformers import LlamaConfig

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
from analysis_utils import shuffle_exp

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
