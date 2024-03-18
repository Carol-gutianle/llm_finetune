import torch

from omegaConf import OmegaConf

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from peft import (
    PeftModel,
    LoraConfig
)

def inference(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    if args.do_peft:
        model = PeftModel.from_pretrained(
            model,
            args.lora_weights_path,
            torch_dtype = torch.float16
        )
        
    model.eval()
    
    # get_data
    test_prompt = '''Are the details of Jaime Vasquez's birth documented?'''
    input_ids = tokenizer(test_prompt, return_tensors='pt')['input_ids'].to(device)
    generate_ids = model.generate(
        input_ids = input_ids,
        return_dict_in_generate = True,
        max_new_tokens = 50
    )
    output = tokenizer.decode(generate_ids.sequence[0]).replace(test_prompt, '')
    print('response:', output)
    
if __name__ == "__main__":
    args = OmegaConf.load('inference.yaml')
    