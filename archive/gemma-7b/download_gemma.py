from transformers import GemmaForCausalLM, GemmaTokenizer

model = GemmaForCausalLM.from_pretrained("google/gemma-2b")
tokenizer = GemmaTokenizer.from_pretrained("google/gemma-2b")
