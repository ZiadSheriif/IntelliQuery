from datasets import load_dataset

dataset = load_dataset("aadityaubhat/GPT-wiki-intro")

print(dataset['train'][0])