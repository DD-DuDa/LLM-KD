from datasets import load_dataset
import random
import json

def get_gen_dataset(dataset_name, max_sample=None, tokenizer=None):
    if dataset_name == "wikitext":
        return get_wiki_dataset(max_sample)
    elif dataset_name == "eg-balanced":
        return get_eg_balanced_dataset(max_sample, tokenizer)
    else:
        raise ValueError(f"{dataset_name} not implement yet")

def extract_random_dataset(sources, targets, max_sample=None):
    if max_sample is not None:
        if max_sample <= len(sources):
            print(f"only use {max_sample} samples")
            random_indices = random.sample(range(len(sources)), max_sample)
            sources = [sources[i] for i in random_indices]
            targets = [targets[i] for i in random_indices]
        else:
            print("max_sample exceeds the length of the array. Using the entire array.")
            sources = sources
            targets = targets
    else:
        print(f"using the whole {len(sources)} samples")

    return sources, targets

def get_wiki_dataset(max_sample):
    wiki_dataset = load_dataset("wikitext", 'wikitext-2-raw-v1', split='train')
    # wiki_dataset = load_dataset("/root/model/datasets/wikitext/wikitext", 'wikitext-2-raw-v1', split='train')

    wiki_long = []
    for text in wiki_dataset['text']:
        if len(text) > 512:
            wiki_long.append(text)
    wiki_front = [long[:128] for long in wiki_long]

    targets = sources = wiki_front

    return extract_random_dataset(sources, targets, max_sample)

def get_eg_balanced_dataset(max_sample, tokenizer):
    """Load EG-balanced dataset for boolean QA"""
    json_path = "../data/train_EG-balanced_NL2.json"
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    sources = []
    targets = []
    
    for item in data:
        input_text = item["instruct"] + " " + item["question"]
        label = "true" if item["label"] else "false"
        sources.append(input_text)
        targets.append(f" {label}{tokenizer.eos_token}")
    
    return extract_random_dataset(sources, targets, max_sample)
    