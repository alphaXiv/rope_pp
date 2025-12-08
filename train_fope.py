import os
import sys
import json
import random
from datetime import datetime

import torch
import numpy as np

import datasets

import deepspeed

from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers.integrations import HfDeepSpeedConfig

from models.configuration_llama import LlamaConfig
from models.modeling_llama_fope import LlamaForCausalLM

from utils.dataset_utils import StreamingTrainingJsonlZSD, EvaluatingDataset
from utils.callback_utils import CheckpointingCallback, CustomLoggingCallback
from utils.trainer_utils import TrainerWithDatasetCheckpointing

root = os.getcwd()
tokenizer_path = 'meta-llama/Meta-Llama-3-8B'

cache_dir = ''  # set a cache_dir

train_dataset_path = ''  # path of mlfoundations/dclm-baseline-1.0
train_dataset_label = 'text'

valid_dataset_path = ''  # path of Pile-CC
valid_dataset_split = 'validation'
valid_dataset_abbr = 'pile_cc'
valid_dataset_label = 'content'

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.set_default_dtype(torch.bfloat16)

import argparse

parser = argparse.ArgumentParser(description='define fp config')

parser.add_argument('--config_abbr', type=str, default='376m')
parser.add_argument('--save_abbr', type=str, default='376m')

parser.add_argument('--local_rank', type=int, default=-1)

args = parser.parse_args()

config_abbr = args.config_abbr
config_path = f'{root}/configs/rope-{config_abbr}-config.json'

save_abbr = args.save_abbr

gradient_accumulation_steps = 1

batch_size = 128
max_length = 4096
valid_size = 16384

max_steps = 100000
eval_steps = 500
warmup_steps = 500

save_steps = 10000
steps_to_save = [100, max_steps]

# ref: https://www.deepspeed.ai/docs/config-json/

ds_config = {
    "bf16": {
        "enabled": True
    },
    "zero_allow_untested_optimizer": True,
    "zero_force_ds_cpu_optimizer": False,
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e7,
        "stage3_max_live_parameters": 1e7,
        "stage3_max_reuse_distance": 1e7,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "gradient_accumulation_steps": 1,
    "steps_per_print": 100,
    "train_batch_size": batch_size,
    "wall_clock_breakdown": False, 
}

dschf = HfDeepSpeedConfig(ds_config)

# ref: https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/partition_parameters.py#L603

with deepspeed.zero.Init(dtype=torch.bfloat16, config_dict_or_path=ds_config):

    config = LlamaConfig.from_pretrained(config_path)
    config.gradient_checkpointing = True
    config.use_cache = False
    config._attn_implementation = "flash_attention_2"
    config.torch_dtype = torch.bfloat16

    config.fourier = True
    config.fourier_dim = 128
    config.fourier_init = "eye_xavier_norm"
    config.fourier_init_norm_gain = 0.3
    config.fourier_separate_basis = True
    config.fourier_separate_head = True
    config.fourier_learnable = False
    config.fourier_norm = False
    config.fourier_ignore_zero = True
    config.rope_init_distribution: str = "exponential"
    config.rope_no_pos: bool = False
    config.rope_no_rotary: bool = False
    config.rope_no_repetition: bool = False
    config.rope_full_precision: bool = True
    config.rope_learnable: bool = False
    config.rope_include_neg_freq: bool = False
    config.rope_init_upper_freq: float = 1.0
    config.rope_init_floor_freq: float = 0.0
    config.rope_floor_freq_ratio: float = 1.0
    config.rope_upper_freq_ratio: float = 1.0
    config.rope_clamp_floor_freq: bool = True
    config.rope_clamp_floor_to_zero: bool = True
    config.rope_clamp_upper_freq: bool = False
    config.rope_clamp_upper_to_zero: bool = True
    config.rope_zero_freq_ratio: float = -1.0
    config.rope_clamp_to_linear: bool = False
    config.rope_clamp_to_linear_mode: str = "ceil"
    config.len_extra = False
    config.init_device = "cuda"
    
    config.ignore_index = config.eos_token_id

    model = LlamaForCausalLM(config=config)

rank = torch.distributed.get_rank()
size = torch.distributed.get_world_size()

train_args = {
    'per_device_train_batch_size': batch_size // size, 
    'per_device_eval_batch_size': batch_size // size, 
    'do_train': True, 'do_eval': True, 'bf16': True, 'bf16_full_eval': True, 
    'optim': 'adamw_torch', 'learning_rate': 5e-4, 'weight_decay': 0.1, 
    'adam_beta1': 0.95, 'adam_beta2': 0.99, 'num_train_epochs': 1, 
    'lr_scheduler_type': 'constant_with_warmup', 'warmup_steps': warmup_steps,

    'label_names': [], 'output_dir': f'{root}/checkpoints/{save_abbr}', 
    'eval_strategy': 'steps', 'eval_steps': eval_steps, 
    'logging_strategy': 'steps', 'logging_steps': 1, 
    'save_strategy': 'steps', 'save_steps': save_steps, 
    'gradient_accumulation_steps': gradient_accumulation_steps, 
    'max_steps': max_steps, 'disable_tqdm': True, 'save_only_model': True, 
}

if rank == 0:
    print(f'{config = }', '\n')
    print('train_args = ', json.dumps(train_args, indent=2), '\n')
    print('model is ready !', '\n')
    print('save ckpt at', sorted(list(range(0, max_steps, save_steps))[1:] + steps_to_save),'\n')

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

if rank == 0:
    print('tokenizer is ready !', '\n')

valid_dataset = datasets.load_dataset(valid_dataset_path, split=valid_dataset_split, 
                                      cache_dir=cache_dir)
valid_dataset = valid_dataset.select(range(valid_size))

if rank == 0:
    print(valid_dataset, '\n')

train_dataset = StreamingTrainingJsonlZSD(data_root=train_dataset_path, tokenizer=tokenizer, 
                                          label_name=train_dataset_label, train_length=max_length, 
                                          num_data=max_steps * batch_size, seed=seed)
valid_dataset = EvaluatingDataset(dataset=valid_dataset, tokenizer=tokenizer, 
                                  label_name=valid_dataset_label, valid_length=max_length)

if rank == 0:
    print('dataset is ready !', '\n')

# ref: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "rope_pp"
os.environ["WANDB_DIR"] = f"{root}/wandb"

training_args = TrainingArguments(
    report_to='wandb', logging_dir=f'{root}/wandb',
    run_name=f'{save_abbr}-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
    include_for_metrics='loss', deepspeed=ds_config, **train_args
)

if rank == 0:
    print('checkpoints and model will be saved in', train_args['output_dir'], '\n')

start_time = datetime.now()

trainer = TrainerWithDatasetCheckpointing(
    model=model, tokenizer=tokenizer, args=training_args,
    train_dataset=train_dataset, eval_dataset={valid_dataset_abbr: valid_dataset}, 
    callbacks=[CheckpointingCallback(steps_to_save=steps_to_save), 
               CustomLoggingCallback(max_steps=max_steps, batch_size=batch_size, max_length=max_length, 
                                     world_size=size, valid_dataset_abbr=valid_dataset_abbr, 
                                     logging_steps=20)
                ], 
)

trainer.can_return_loss = True

trainer.train()

if rank == 0:
    end_time = datetime.now()
    total_time = end_time - start_time
    print(f"[{str(end_time)}] 100.00% {max_steps} / {max_steps} [{str(total_time)} / {str(total_time)}]")
    print('training is over !')
