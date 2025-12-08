import os
import sys
import json
import random

from tqdm import tqdm
from datetime import datetime

import torch
import numpy as np

import datasets

import deepspeed

from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers.integrations import HfDeepSpeedConfig

from configs.model_configs import model_configs

from models.configuration_llama import LlamaConfig

from utils.dataset_utils import StreamingTrainingParquet, EvaluatingDataset
from utils.callback_utils import CheckpointingCallback, CustomLoggingCallback
from utils.trainer_utils import TrainerWithDatasetCheckpointing

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

root = os.getcwd()
tokenizer_path = 'meta-llama/Meta-Llama-3-8B'

cache_dir = ''  # set a cache_dir

batch_size = 128
valid_size = 16384

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

lambada = datasets.load_dataset('EleutherAI/lambada_openai', 'default', cache_dir=cache_dir)
lambada_ppl = tokenizer('\n\n'.join(lambada['test']['text']), return_tensors='pt')

print(f'{lambada_ppl.input_ids.size(1) = }')

wikitext = datasets.load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', cache_dir=cache_dir)
wikitext_ppl = tokenizer('\n\n'.join(wikitext['validation']['text']), return_tensors='pt')

print(f'{wikitext_ppl.input_ids.size(1) = }')

path_root = root + '/checkpoints'

models = [

    ('RoPE-DCLM-376M-4k', 'SII-xrliu/RoPE-DCLM-376M-4k', {'imag': False, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EH-DCLM-376M-4k', 'SII-xrliu/RoPEPP_EH-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EC-DCLM-376M-4k', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, 64), 

    ('ALiBi-DCLM-376M-4k', 'SII-xrliu/ALiBi-DCLM-376M-4k', {}, 64), 
    ('FoPE-DCLM-376M-4k', 'SII-xrliu/FoPE-DCLM-376M-4k', {}, 64), 
    ('Pythia-DCLM-376M-4k', 'SII-xrliu/Pythia-DCLM-376M-4k', {}, 64), 

    ('RoPE-DCLM-376M-32k', 'SII-xrliu/RoPE-DCLM-376M-32k', {'imag': False, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EH-DCLM-376M-32k', 'SII-xrliu/RoPEPP_EH-DCLM-376M-32k', {'imag': True, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EC-DCLM-376M-32k', 'SII-xrliu/RoPEPP_EC-DCLM-376M-32k', {'imag': True, 'imag_mode': 'imag2'}, 64), 

    ('RoPE-DCLM-776M-4k', 'SII-xrliu/RoPE-DCLM-776M-4k', {'imag': False, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EH-DCLM-776M-4k', 'SII-xrliu/RoPEPP_EH-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EC-DCLM-776M-4k', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, 64), 

    ('ALiBi-DCLM-776M-4k', 'SII-xrliu/ALiBi-DCLM-776M-4k', {}, 64), 
    ('FoPE-DCLM-776M-4k', 'SII-xrliu/FoPE-DCLM-776M-4k', {}, 64), 
    ('Pythia-DCLM-776M-4k', 'SII-xrliu/Pythia-DCLM-776M-4k', {}, 64), 

    ('RoPE-DCLM-776M-32k', 'SII-xrliu/RoPE-DCLM-776M-32k', {'imag': False, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EH-DCLM-776M-32k', 'SII-xrliu/RoPEPP_EH-DCLM-776M-32k', {'imag': True, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EC-DCLM-776M-32k', 'SII-xrliu/RoPEPP_EC-DCLM-776M-32k', {'imag': True, 'imag_mode': 'imag2'}, 64), 

    ('RoPE-DCLM-1_5B-4k', 'SII-xrliu/RoPE-DCLM-1_5B-4k', {'imag': False, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EH-DCLM-1_5B-4k', 'SII-xrliu/RoPEPP_EH-DCLM-1_5B-4k', {'imag': True, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EC-DCLM-1_5B-4k', 'SII-xrliu/RoPEPP_EC-DCLM-1_5B-4k', {'imag': True, 'imag_mode': 'imag2'}, 64), 

    ('RoPE-DCLM-1_5B-32k', 'SII-xrliu/RoPE-DCLM-1_5B-32k', {'imag': False, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EH-DCLM-1_5B-32k', 'SII-xrliu/RoPEPP_EH-DCLM-1_5B-32k', {'imag': True, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EC-DCLM-1_5B-32k', 'SII-xrliu/RoPEPP_EC-DCLM-1_5B-32k', {'imag': True, 'imag_mode': 'imag2'}, 64), 

]


results = {}

for abbr, path, rope_config, max_out_len in models:

    result_path = f"{root}/results/{abbr}.json"
    print(result_path)

    if os.path.exists(result_path):
        result = json.load(open(result_path, mode='r'))
    else:
        result = {}
    
    config = LlamaConfig.from_pretrained(path)
    config.use_cache = False
    config._attn_implementation = "flash_attention_2"
    config.torch_dtype = torch.float16
    config.rope_config = rope_config
    config.ignore_index = config.eos_token_id

    if 'FoPE' in abbr:
        from rope_pp.modeling_llama_fope import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.float16, config=config).cuda()
    elif 'ALiBi' in abbr:
        from rope_pp.modeling_llama_alibi import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.float16, config=config).cuda()
    elif 'Pythia' in abbr:
        from rope_pp.modeling_llama_pythia import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.float16, config=config).cuda()
    else:
        from models.modeling_llama_rope_pp import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.float16, config=config).cuda()
    batch_size = 8

    with torch.no_grad():

        stride = 512
        max_length = 4096

        if 'ppl_lambada' in result:
            ppl_lambada = result['ppl_lambada']
        else:
            lls = []
            for i in tqdm(range(0, lambada_ppl.input_ids.size(1), stride)):
                begin_loc = max(i + stride - max_length, 0)
                end_loc = i + stride
                input_ids = lambada_ppl.input_ids[:,begin_loc:end_loc].cuda()
                labels = input_ids.clone().cuda()
                labels[:,:-stride] = config.ignore_index
                loss_ = model(input_ids=input_ids, labels=labels).loss
                log_likelihood = loss_ * stride
                lls.append(log_likelihood)
            ppl_lambada = float(torch.exp(torch.stack(lls).sum() / i).cpu().numpy())

        if 'ppl_wikitext' in result:
            ppl_wikitext = result['ppl_wikitext']
        else:
            lls = []
            for i in tqdm(range(0, wikitext_ppl.input_ids.size(1), stride)):
                begin_loc = max(i + stride - max_length, 0)
                end_loc = i + stride
                input_ids = wikitext_ppl.input_ids[:,begin_loc:end_loc].cuda()
                labels = input_ids.clone().cuda()
                labels[:,:-stride] = config.ignore_index
                loss_ = model(input_ids=input_ids, labels=labels).loss
                log_likelihood = loss_ * stride
                lls.append(log_likelihood)
            ppl_wikitext = float(torch.exp(torch.stack(lls).sum() / i).cpu().numpy())

        result.update({
            'ppl_lambada': ppl_lambada, 
            'ppl_wikitext': ppl_wikitext, 
        })
        results.update({abbr: result})

    with open(result_path, mode='w+') as file:
        json.dump(result, file, indent=4)
    
    print(f'model {abbr} is over !')

suffix = str(datetime.now()).split('.')[0].replace('-', '').replace(':', '').replace(' ', '_')

with open(f'results/test_ppl-{suffix}.csv', mode='w+') as file:
    line_title = ','.join(['dataset'] + list(results))
    file.write(line_title + '\n')
   
    line_ppl = ','.join(['ppl_lambada'] + list([f"{results[key]['ppl_lambada']:.6f}" for key in results]))
    file.write(line_ppl + '\n')
   
    line_ppl = ','.join(['ppl_wikitext'] + list([f"{results[key]['ppl_wikitext']:.6f}" for key in results]))
    file.write(line_ppl + '\n')
