import os
import sys
import json
import random
from datetime import datetime

import torch

from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback, TrainerState, TrainerControl


class CustomLoggingCallback(TrainerCallback):

    def __init__(self, max_steps, batch_size, max_length, world_size, valid_dataset_abbr, logging_steps=10):
        self.cur_step, self.max_steps = 0, max_steps
        self.start_time, self.valid_total_time = datetime.now(), 0
        self.batch_token = batch_size * max_length
        self.world_size = world_size
        self.valid_dataset_abbr = valid_dataset_abbr
        self.logging_steps = logging_steps

        # Handle both distributed and single GPU training
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        now = datetime.now()
        if logs is not None and 'grad_norm' in logs:

            if self.cur_step == 0:
                self.start_time = now
                total_time = now - self.start_time
                log_dict = {'num_consume_token': 0, 'avg_tgs': 0}
                if self.rank == 0 and self.cur_step % self.logging_steps == 0:
                    print(f"[{str(now)}]   0.00% {self.cur_step:{len(str(self.max_steps))}d} / {self.max_steps}", 
                        f"[{str(total_time)} / {str(total_time)}] {log_dict}", end=', ')

            else:
                total_time = now - self.start_time
                train_total_time = total_time.total_seconds() - self.valid_total_time
                num_consume_token = self.cur_step * self.batch_token
                avg_tgs = num_consume_token / train_total_time / self.world_size

                cur_percent = self.cur_step / self.max_steps * 100
                final_time = total_time / self.cur_step * self.max_steps
                log_dict = {'num_consume_token': num_consume_token, 'avg_tgs': avg_tgs}
                if self.rank == 0 and self.cur_step % self.logging_steps == 0:
                    print(f"[{str(now)}] {cur_percent:6.2f}% {self.cur_step:{len(str(self.max_steps))}d} / {self.max_steps}", 
                        f"[{str(total_time)} / {str(final_time)}] {log_dict}", end=', ')

            self.cur_step += 1
        else:
            total_time = now - self.start_time
            cur_percent = self.cur_step / self.max_steps * 100

            if self.cur_step == 0:
                self.start_time = now
                total_time = now - self.start_time
                if self.rank == 0 and self.cur_step % self.logging_steps == 0:
                    print(f"[{str(now)}]   0.00% {self.cur_step:{len(str(self.max_steps))}d} / {self.max_steps}", 
                        f"[{str(total_time)} / {str(total_time)}] validation ", end='')
            else:
                final_time = total_time / self.cur_step * self.max_steps
                if self.rank == 0:
                    print(f"[{str(now)}] {cur_percent:6.2f}% {self.cur_step:{len(str(self.max_steps))}d} / {self.max_steps}", 
                        f"[{str(total_time)} / {str(final_time)}] validation ", end='')
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if f'eval_{self.valid_dataset_abbr}_runtime' in metrics:
            self.valid_total_time += metrics[f'eval_{self.valid_dataset_abbr}_runtime']


class CheckpointingCallback(TrainerCallback):
    def __init__(self, steps_to_save):
        self.steps_to_save = steps_to_save

        # Handle both distributed and single GPU training
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step in self.steps_to_save:
            control.should_save = True
            control.should_evaluate = True
            if self.rank == 0:
                print(f"Saving checkpoint at step {state.global_step}")
        return control
