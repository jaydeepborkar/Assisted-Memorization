import transformers
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM
from accelerate import Accelerator
import csv
import os
import numpy as np
import sys

arg = sys.argv[1]

model_checkpoint = "gpt2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
block_size = 128


def tokenize_function(examples):
    return tokenizer(examples["text"])


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


class CustomTrainer(Trainer):
    """
    saves data seen by the model every 10% interval of total training 
    """
    def __init__(self, *args, tokenizer=None, save_dir=".", **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        self.steps_per_epoch = len(self.get_train_dataloader())
        self.steps_per_interval = self.steps_per_epoch // 10
        self.current_epoch = 0
        self.current_interval = 0
        self.data_seen = []

        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_begin(self):
        self.current_interval = 0

    def training_step(self, model: torch.nn.Module, inputs: dict) -> torch.Tensor:
        if self.state.global_step // self.steps_per_interval > self.current_interval:
            self.save_data_interval()
            self.current_interval += 1

        self.data_seen.append(inputs)
        return super().training_step(model, inputs)

    def on_epoch_end(self):
        self.save_data_interval(final=True)
        self.data_seen = []
        self.current_epoch += 1

    def save_data_interval(self, final=False):
        interval_label = "final" if final else f"{(self.current_interval + 1) * 10}%"
        filename = f"epoch{self.current_epoch + 1}_{interval_label}.txt"
        filepath = os.path.join(self.save_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            for batch in self.data_seen:
                decoded_texts = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                for text in decoded_texts:
                    f.write(text + '\n')

        self.data_seen = []

    
def get_training_args(arg, lm_datasets):
    """
    Returns training arguments  
    """
    
    base_args = {
        "output_dir": "models",
        "evaluation_strategy": "no",
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "push_to_hub": False,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8, #adjust batch size according to your compute 
        "per_device_eval_batch_size": 8,
        "fp16": True,
        "report_to": "none",
        "lr_scheduler_type": "linear",
        "warmup_steps": 500,
        "seed": 42
    }
    
    if arg == "continue_train":
        base_args["save_strategy"] = "steps"
        checkpoint = (len(lm_datasets["train"]) // base_args[ "per_device_train_batch_size"]) * base_args["num_train_epochs"] 
        checkpoint = checkpoint // 30 #save a checkpoint every ~10% increment of total training 
        base_args["save_steps"] = checkpoint
        print(base_args)
    elif arg == "retrain":
        base_args["save_strategy"] = "epoch"
    
    return TrainingArguments(**base_args)    


def get_trainer(arg, model, training_args, lm_datasets, tokenizer=None):
    """
    Use a custom trainer for continued training and vanilla trainer otherwise 
    """
    if arg == "continue_train":
        return CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["validation"],
            tokenizer=tokenizer,
            save_dir="data")
    elif arg == "retrain":
        return Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["validation"])
    

def train(file_path): 
    
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map="auto", low_cpu_mem_usage=True)

    datasets = load_dataset("text", data_files={"train": file_path, "validation": 'datasets/wikivalidation'})
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    
    lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000, num_proc=4,)
    training_args = get_training_args(arg, lm_datasets)
    
    trainer = get_trainer(arg, model, training_args, lm_datasets, tokenizer) 
    trainer.train() 


def main(): 
    if arg == "continue_train":
        print("Continuing training")
        train("datasets/100.txt")
    
    elif arg == "retrain":
        print("Retraining on datasets")
        for i in range(10, 110, 10):
            file_path = f"datasets/{i}.txt"
            if os.path.exists(file_path):
                print(f"Training on {file_path}")
                train(file_path)
                      
    else:
        print("Invalid argument. Valid arguments: 'continue_train' or 'retrain'")
            

if __name__ == "__main__":
    main() 