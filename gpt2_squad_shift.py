import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForCausalLM
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch
import numpy as np
import json
import argparse
from inference import sample_batched
from transformers.trainer_callback import TrainerCallback
import random
from itertools import chain
from tqdm import tqdm, trange
from transformers import pipeline


def print_device_place(model):
    for parameters in model.parameters():
        print(parameters.device)

def parallelize_across_device(model, num_heads):
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


print('Using %d GPUs.' % torch.cuda.device_count())
mount_dir = 'mount/'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_init_path', type=str, default=None)
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--max_steps', type=int, default=20000)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--save_steps', type=int, default=100000)
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--eval_steps', type=int, default=2500)

    args, unknown = parser.parse_known_args()
    
    model_init_path = args.model_init_path
    actual_bsize = 32

    models_dir = mount_dir + "models/"
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    warmup_steps = args.warmup_steps
    max_steps = args.max_steps
    train_batch_size = args.train_batch_size
    eval_batch_size = train_batch_size * 2
    gradient_accumulation_steps = actual_bsize // train_batch_size
    print('gradient accumulation steps', gradient_accumulation_steps)
    save_steps = args.save_steps
    no_train = args.no_train
    eval_steps = 4000 * gradient_accumulation_steps
    temperature = args.temperature
    n_samples = args.n_samples

    model_size2num_heads = {
        'gpt2': 12,
        'gpt2-medium': 24,
        'gpt2-large': 36,
        'gpt2-xl': 48
    }

    for data_name in ['squadshifts_experiment_question_left', 'squadshifts_experiment_double', 'squadshifts_experiment_question_right']:
        data_path = mount_dir + "data/%s.json" % data_name
        training_run_name = models_dir + model_init_path.replace('/', '_') + '-' + data_name
        if not os.path.exists(training_run_name):
            os.mkdir(training_run_name)
        else:
            continue
        print('loading model')
        model = AutoModelForCausalLM.from_pretrained(model_init_path)
        parallelize_across_device(model, model_size2num_heads[model_init_path])
        print('model loaded')
        print('saving result to %s' % training_run_name)

        tokenizer = AutoTokenizer.from_pretrained(model_init_path)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.model_max_length = 1024
        model_tokenizer = (model, tokenizer)

        # defining the prompt completion list
        data = json.load(open(data_path))
        train, test = data['train'], data['eval']
        demonstrations = [d['completion'] for d in test]

        for d in chain(train, test):
            d['prompt'] = d['prompt'].strip()
        for d in chain(train, test):
            d['completion'] = d['completion'] + ' <|endoftext|>'
        test_prompts = [d['prompt'] for d in test]

        def eval_model_on_test_prompts(target_model):
            model_tokenizer = (target_model, tokenizer)
            sampled_results = sample_batched(model_tokenizer, test_prompts, temperature=temperature, n=n_samples, bsize=eval_batch_size, max_target_length=20)

            all_results = []
            for (prompt, generations), demonstration, orig_d in zip(sampled_results.items(), demonstrations, test):
                d = {'prompt': prompt, 'generations': generations, 'demonstration': demonstration, 'orig_d': orig_d}
                all_results.append(d)
            return all_results

        def save_result_path(step):
            hyp_str = 'temperature=%.2f_n=%d_step=%d' % (temperature, n_samples, step)
            save_path = os.path.join(training_run_name, '%s.json' % hyp_str)
            return save_path

        if not no_train:
            arg_dict = vars(args)
            json.dump(arg_dict, open(training_run_name + '/training_args.json', 'w'))

            random.shuffle(train)
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optim = AdamW(optimizer_grouped_parameters, lr=1e-5)

            scheduler = get_linear_schedule_with_warmup(optim, warmup_steps, max_steps)
            pbar = trange(max_steps * gradient_accumulation_steps, desc="Training")
            total_epoch_count = (max_steps * actual_bsize) // len(train)
            print('total epoch count', total_epoch_count)
            for step_idx in pbar:
                selected_idxes = [i % len(train) for i in range(step_idx * train_batch_size, (step_idx + 1) * train_batch_size)]
                selected = [train[i] for i in selected_idxes]

                lm_inputs = [d['prompt'] + ' ' + d['completion'] for d in selected]
                lm_labels = [d['prompt'] + ' ' + d['completion'] for d in selected]

                tokenized_inputs = tokenizer(lm_inputs, padding=True, truncation=True, return_tensors="pt").to('cuda')
                tokenized_labels = tokenizer(lm_labels, padding=True, truncation=True, return_tensors="pt").to('cuda')

                # fill -100 for the labels at the prompt
                completion_tokenized_length = [len(tokenizer.encode(d['completion'])) for d in selected]
                completion_tokenized_length = torch.tensor(completion_tokenized_length).to('cuda')
                tokenized_labels['input_ids'] = torch.where(torch.arange(tokenized_labels['input_ids'].shape[1]).to('cuda') < (tokenized_inputs ['input_ids'].shape[1] - completion_tokenized_length.unsqueeze(1)), -100, tokenized_labels['input_ids'])
                labels = tokenized_labels['input_ids'].clone()
                input_ids = tokenized_inputs['input_ids'].clone()
                loss = model(**tokenized_inputs, labels=labels).loss
                loss.backward()
                pbar.set_description('loss: %.4f' % loss.item())

                if (step_idx + 1) % gradient_accumulation_steps == 0:
                    optim.step()
                    scheduler.step()
                    optim.zero_grad()

                if (step_idx + 1) % eval_steps == 0:
                    all_results = eval_model_on_test_prompts(model)
                    save_step_count = (step_idx + 1) // gradient_accumulation_steps
                    save_path = save_result_path(save_step_count)
                    json.dump(all_results, open(save_path, 'w'))

