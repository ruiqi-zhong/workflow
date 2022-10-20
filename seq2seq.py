import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import numpy as np
from torch.utils.data import Dataset
from trainer_utils import PromptCompletionDataset, get_seq2seq_training_args, get_seq2seq_collator
import json
import argparse
from inference import sample_batched

def print_device_place(model):
    for parameters in model.parameters():
        print(parameters.device)

def parallelize_across_device(model):
    num_heads = len(model.encoder.block)
    num_device = torch.cuda.device_count()
    other_device_alloc = num_heads // num_device + 1
    first_device = num_heads - (num_device - 1) * other_device_alloc
    device_map = {}
    cur = 0
    end = max(cur + first_device, 1)
    device_map[0] = list(range(cur, end))
    cur = end
    for i in range(1, num_device):
        end = min(cur + other_device_alloc, num_heads)
        device_map[i] = list(range(cur, end))
        cur += other_device_alloc
    print('device_map', device_map)
    model.parallelize(device_map)

def fit_bit(model):
    for n, parameters in model.named_parameters():
        if 'bias' not in n and 'norm' not in n.lower():
            parameters.requires_grad = False


print('Using %d number of GPUs: ' % torch.cuda.device_count())
mount_dir = 'mount/'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_init_name', type=str, default=None)
    parser.add_argument('--model_init_path', type=str, default=None)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--training_run_name', type=str)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=3000)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--temperature', type=float, default=0.8)
    paraser.add_argument('--n_samples', type=int, default=8)

    args, unknown = parser.parse_known_args()
    
    model_init_path = args.model_init_path
    if model_init_path is None:
        model_init_path = mount_dir + "models/%s" % args.model_init_name

    data_path = mount_dir + "data/%s.json" % args.data_name
    training_run_name = mount_dir + "models/%s" % args.training_run_name
    warmup_steps = args.warmup_steps
    max_steps = args.max_steps
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    save_steps = args.save_steps

    tokenizer_path = mount_dir + "models/t5-small-cp_tokenizer"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_init_path)

    # only finetune the bias and norm layers
    # fit_bit(model)

    # parallelize across devices via device placement
    parallelize_across_device(model)

    # the below lines print reasonable results
    # for name, parameters in model.named_parameters():
    #     print(name, parameters.shape, parameters.device, parameters.requires_grad)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.model_max_length = 1024
    model_tokenizer = (model, tokenizer)

    # defining the prompt completion list
    data = json.load(open(data_path))
    train, test = data['train'], data['eval']
    train_dataset = PromptCompletionDataset(train, model_tokenizer)
    eval_dataset = PromptCompletionDataset(test, model_tokenizer)

    # get a bunch of training arguments
    args = get_seq2seq_training_args(
        training_run_name=training_run_name,
        warmup_steps=warmup_steps, max_steps=max_steps,
        train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, 
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps
    )
    data_collator = get_seq2seq_collator(model_tokenizer)

    # training loop
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    test_prompts = [d['prompt'] for d in test]
    sampled_results = sample_batched(model_tokenizer, test_prompts, args.temperature, args.n_samples)


def sample_batched(
    model_tokenizer, prompts, temperature=0.8, n=8, bsize=8, 
    max_source_length=1024, max_target_length=512, save_score_tok_idx=None, verbose=False, stop_strs=None)