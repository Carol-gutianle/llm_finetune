import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from omegaconf import OmegaConf


from shutil import copy
from pprint import pprint
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import RandomSampler, DataLoader
from dataclasses import dataclass
from dataset import Collate, MyDataset
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM

from peft import (
    LoraConfig,
    get_peft_model
)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    
def main():
    args = OmegaConf.load('train.yaml')
    pprint(vars(args))
    
    # training settings
    gradient_accumulation_steps = args.gradient_accumulation_steps
    device_map = 'auto'
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype = torch.float16,
        device_map = device_map,
    )
    
    tokenizer = LlamaTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    train_data = MyDataset(load_dataset('locuslab/TOFU', 'full')['train'])
    
    
    collate = Collate(tokenizer, args)
    # peft_config = LoraConfig(
    #     r = args.lora_r,
    #     lora_alpha = 16,
    #     target_modules="q_proj,v_proj,k_proj,o_proj".split(","),
    #     lora_dropout=0.1,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     inference_mode=False,
    # )
    # model = get_peft_model(model, peft_config)
    # print_trainable_parameters(model)
    
    training_args = TrainingArguments(
        per_device_train_batch_size = args.train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = 500,
        num_train_epochs = args.num_train_epochs,
        learning_rate = args.lr,
        fp16 = False,
        logging_steps = 100,
        optim = "adamw_torch",
        save_strategy = "steps",
        eval_steps = None,
        save_steps = 5000,
        output_dir = args.output_dir,
        save_total_limit = 1,
        ddp_find_unused_parameters = False if ddp else None,
        group_by_length = False,
        remove_unused_columns = False
    )
    
    trainer = Trainer(
        model = model,
        train_dataset = train_data,
        args = training_args,
        data_collator = collate
    )
    
    model.config.use_cache = False

    trainer.train()

    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
