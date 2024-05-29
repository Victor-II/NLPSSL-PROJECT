import torch
import tqdm
import json
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, BertTokenizer

DS_PATH = {
    'vitaminc': ['tals/vitaminc', '../data/vitaminc'],
    'nli_fever': ['pietrolesci/nli_fever', '../data/nli_fever']
}

MODEL_CHECKPOINT = ''

def process_vitaminc(dataset: list[dict]) -> list[dict]:
    out = []
    for item in dataset:
        evidence = item['evidence']
        claim = item['claim']
        original_label = item['label']

        if original_label == 'SUPPORTS':
            label = 0
        elif original_label == 'REFUTES':
            label = 2
        else:
            label = 1

        out.append({
            'evidence': evidence,
            'claim': claim,
            'label': label
        })

    return out

def process_nli_fever(dataset: list[dict]) -> list[dict]:
    out = []
    for item in dataset:
        evidence = item['hypothesis']
        claim = item['premise']
        original_label = item['fever_gold_label']

        if original_label == 'SUPPORTS':
            label = 0
        elif original_label == 'REFUTES':
            label = 2
        else:
            label = 1

        out.append({
            'evidence': evidence,
            'claim': claim,
            'label': label
        })

    return out

def save_data(dataset: list[dict], path: str) -> None:
    with open(path, 'x', encoding='utf8') as f:
        f.write('')
    for item in dataset:
        with open(path, 'a', encoding='utf8') as f:
            json.dump({
                'evidence': item['evidence'],
                'claim': item['claim'],
                'label': item['label']
            }, f)
            f.write('\n')


def generate_datasets() -> None:
    for dataset, args in DS_PATH.items():
        print(f'=> generating {dataset} dataset')
        data = load_dataset(args[0])
        for split, ds in data.items():
            print(f'=> processing "{split}" split')
            processed = eval(f'process_{dataset}(ds)')
            path = f'{args[1]}/{split}.json'
            print(f'=> saving "{split}" split to {path}')
            save_data(processed, path)

def load_from_json(path):
    dataset = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            dataset.append(item)
    return dataset

def load_combined_dataset(split: str='train'):
    dataset = []
    for _, args in DS_PATH.items():
        ds = load_from_json(f'{args[1]}/{split}.json')
        dataset.extend(ds)
    return ds

class DS(Dataset):
    def __init__(self, dataset, tokenizer_model='bert', tokenizer_max_len=512) -> None:
        self.dataset = dataset
        if tokenizer_model == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        elif tokenizer_model == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
        self.tokenizer_max_len = tokenizer_max_len

    def process_data(self, index):
        evidence = self.dataset[index]['evidence']
        claim = self.dataset[index]['claim']
        label = self.dataset[index]['label']

        if random.random() > 0.95:
            claim = evidence
            label = 0
        elif label == 2 and random.random() > 0.9:
            claim = self.dataset[index]['evidence']
            evidence = self.dataset[index]['claim']

        try:
            tokens = self.tokenizer(evidence, claim, padding='max_length', max_length=self.tokenizer_max_len, truncation='only_first')
        except:
            tokens = self.tokenizer(evidence, claim, padding='max_length', max_length=self.tokenizer_max_len, truncation=True)

        return [
            torch.tensor(tokens['input_ids']),
            torch.tensor(tokens['attention_mask']),
            torch.tensor(tokens['token_type_ids']) if 'token_type_ids' in tokens.keys() else None,
            torch.tensor(label)
        ]

    
    def __getitem__(self, index):
        input_ids, attention_mask, token_type_ids, label =  self.process_data(index)

        if token_type_ids:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'label': label
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label
            }

    def __len__(self):
        return len(self.dataset)

class DL():
    def __init__(self, max_data_size):
        pass

    def load_data(self):
        pass

    def train(self):
        pass

    def val(self):
        pass

if __name__ == '__main__':
    
    ds = load_combined_dataset()
    print(ds[0])

    