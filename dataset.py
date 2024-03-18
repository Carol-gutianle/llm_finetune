import torch
import json
from torch.utils.data import Dataset
from dataclasses import dataclass
import re
import ast
import sys
from pprint import pprint
 
 
def get_args_from_json(json_file_path, args_dict):
    import json
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)
 
    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]
 
    return args_dict
 
 
PROMPT_TEMPLATE = (
    'Below is an instruction that describe a task.'
    'Write a response that appropriately completes the request.\n\n'
    '### Instruction:\n{instruction}\n\n### Response:'
)

IGNORE_INDEX = -100

class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

@dataclass
class Collate:
    def __init__(self, tokenizer, args) -> None:
        self.instruct_column = args.instruct_column
        self.query_column = args.query_column
        self.response_column = args.response_column
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_length
        
    def __call__(self, batch):
        sources = []
        targets = []
        prompt = PROMPT_TEMPLATE
        for example in batch:
            if isinstance(example, str):
                example = json.loads(example)
            input = example[self.query_column]
            output = example[self.response_column]
            if self.instruct_column in example:
                instruction = example[self.instruct_column]
                instruction += f'\n{input}'
                source = prompt.format_map({'instruction': instruction})
            else:
                source = prompt.format_map({'instruction': input})
            target = f'{self.tokenizer.bos_token}{output}{self.tokenizer.eos_token}'
            sources.append(source)
            targets.append(target)
        tokenized_sources = self.tokenizer(sources, return_attention_mask=False, add_special_tokens=False)
        tokenized_targets = self.tokenizer(targets, return_attention_mask=False, add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        
        for s, t in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
            input_ids = (s + t)[: self.max_seq_len]
            labels = ([IGNORE_INDEX] * len(s) + t)[: self.max_seq_len]
            assert len(input_ids) == len(labels)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(input_ids))
            labels = labels + [IGNORE_INDEX] * (self.max_seq_len - len(labels))
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            
        results = {
            'input_ids': torch.tensor(all_input_ids),
            'labels': torch.tensor(all_labels)
        }
        
        return results        