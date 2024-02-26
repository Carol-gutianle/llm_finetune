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
            instruction = example[self.instruct_column]
            input = example[self.query_column]
            output = example[self.response_column]
            if input:
                instruction += f'\n{input}'
            source = prompt.format_map({'instruction': instruction})
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
    
class ConfigParser:
    def __init__(self, config):
        self.config = config
        assert isinstance(config, dict)
        args = sys.argv
        args = args[1:]
        self.args = args

    def judge_type(self, value):
        """利用正则判断参数的类型"""
        if value.isdigit():
            return int(value)
        elif re.match(r'^-?\d+\.?\d*$', value):
            return float(value)
        elif value.lower() in ["true", "false"]:
            return True if value == "true" else False
        else:
            try:
                st = ast.literal_eval(value)
                return st
            except Exception as e:
                return value

    def get_args(self):
        return_args = {}
        for arg in self.args:
            arg = arg.split("=")
            arg_name, arg_value = arg
            if "--" in arg_name:
                arg_name = arg_name.split("--")[1]
            elif "-" in arg_name:
                arg_name = arg_name.split("-")[1]
            return_args[arg_name] = self.judge_type(arg_value)
        return return_args

    # 定义一个函数，用于递归获取字典的键
    def get_dict_keys(self, config, prefix=""):
        result = {}
        for k, v in config.items():
            new_key = prefix + "_" + k if prefix else k
            if isinstance(v, dict):
                result.update(self.get_dict_keys(v, new_key))
            else:
                result[new_key] = v
        return result

    # 定义一个函数，用于将嵌套字典转换为类的属性
    def dict_to_obj(self, merge_config):
        # 如果d是字典类型，则创建一个空类
        if isinstance(merge_config, dict):
            obj = type("", (), {})()
            # 将字典的键转换为类的属性，并将字典的值递归地转换为类的属性
            for k, v in merge_config.items():
                setattr(obj, k, self.dict_to_obj(v))
            return obj
        # 如果d不是字典类型，则直接返回d
        else:
            return merge_config

    def set_args(self, args, cls):
        """遍历赋值"""
        for key, value in args.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise Exception(f"参数【{key}】不在配置中，请检查！")
        return cls

    def parse_main(self):
        # 获取命令行输入的参数
        cmd_args = self.get_args()
        # 合并字典的键，用_进行连接
        merge_config = self.get_dict_keys(self.config)
        # 将字典配置转换为类可调用的方式
        class_config = self.dict_to_obj(merge_config)
        # 合并命令行参数到类中
        cls = self.set_args(cmd_args, class_config)
        return cls        