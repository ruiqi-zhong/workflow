import os

os.environ["CUDA_HOME"] = "/usr/local/cuda"
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoModelForCausalLM,
)
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
import transformers
from typing import Dict


def print_device_place(model):
    for parameters in model.parameters():
        print(parameters.device)


FLOAT_FORMAT = torch.bfloat16

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


print("Using %d GPUs." % torch.cuda.device_count())
hofvarpnir_mount = "model_mount/"
if os.path.exists(hofvarpnir_mount):
    mount_dir = hofvarpnir_mount
else:
    mount_dir = "mount/"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_size", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--max_steps", type=int, default=20001)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--data_name", type=str, default=None)
    parser.add_argument(
        "--model_init_path", type=str, default="../llama_weights_hf/koala_13b_v2/"
    )
    parser.add_argument("--max_target_length", type=int, default=5)
    parser.add_argument("--eval_first", action="store_true")

    args, unknown = parser.parse_known_args()

    if args.model_size is not None:
        model_init_path = f"../llama_weights_hf/llama_{args.model_size}B/"
        if not os.path.exists(model_init_path):
            model_init_path = f"{hofvarpnir_mount}llama_{args.model_size}B/"
    else:
        model_init_path = args.model_init_path
    actual_bsize = 16
    print(model_init_path)

    models_dir = mount_dir + "models/"
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    warmup_steps = args.warmup_steps
    max_steps = args.max_steps
    train_batch_size = args.train_batch_size
    eval_batch_size = train_batch_size
    gradient_accumulation_steps = actual_bsize // train_batch_size
    print("gradient accumulation steps", gradient_accumulation_steps)
    no_train = args.no_train
    eval_steps = args.eval_steps * gradient_accumulation_steps
    save_steps = args.save_steps * gradient_accumulation_steps
    temperature = args.temperature
    n_samples = args.n_samples
    data_name = args.data_name
    data_path = f"mount/data/{data_name}.json"

    model = AutoModelForCausalLM.from_pretrained(
        model_init_path, device_map="balanced"
    ).to(FLOAT_FORMAT)
    print("model loaded")
    training_run_name = (
        models_dir
        + model_init_path.replace("../llama_weights_hf/", "").replace("/", "")
        + "-"
        + data_name
    )
    print("saving result to %s" % training_run_name)
    if os.path.exists(training_run_name):
        #  os.system("rm -rf %s" % training_run_name)
        pass
    else:
        os.mkdir(training_run_name)

    tokenizer = AutoTokenizer.from_pretrained(model_init_path)
    tokenizer.add_special_tokens(
        {
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            "unk_token": tokenizer.convert_ids_to_tokens(
                model.config.pad_token_id
                if model.config.pad_token_id != -1
                else tokenizer.pad_token_id
            ),
        }
    )

    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = 1024
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    model_tokenizer = (model, tokenizer)

    # defining the prompt completion list
    data = json.load(open(data_path))
    train, test = data["train"], data["eval"]
    demonstrations = [d["completion"] for d in test]

    for d in chain(train, test):
        d["prompt"] = tokenizer.bos_token + d["prompt"].strip()
    for d in chain(train, test):
        d["completion"] = d["completion"] + tokenizer.eos_token
    test_prompts = [d["prompt"] for d in test]

    def eval_model_on_test_prompts(target_model):
        model_tokenizer = (target_model, tokenizer)
        sampled_results = sample_batched(
            model_tokenizer,
            test_prompts,
            temperature=temperature,
            n=n_samples,
            bsize=eval_batch_size,
            max_target_length=args.max_target_length,
        )

        all_results = []
        for sampled_d, demonstration, orig_d in zip(
            sampled_results, demonstrations, test
        ):
            d = {
                "prompt": sampled_d["prompt"],
                "generations": sampled_d["generations"],
                "demonstration": demonstration,
                "orig_d": orig_d,
            }
            all_results.append(d)
        return all_results

    def save_result_path(step):
        hyp_str = "temperature=%.2f_n=%d_step=%d" % (temperature, n_samples, step)
        save_path = os.path.join(training_run_name, "%s.json" % hyp_str)
        return save_path

    if not no_train:
        arg_dict = vars(args)
        json.dump(arg_dict, open(training_run_name + "/training_args.json", "w"))

        random.shuffle(train)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optim = AdamW(optimizer_grouped_parameters, lr=1e-5)

        scheduler = get_linear_schedule_with_warmup(optim, warmup_steps, max_steps)
        pbar = trange(max_steps * gradient_accumulation_steps, desc="Training")
        total_epoch_count = (max_steps * actual_bsize) // len(train)
        print("total epoch count", total_epoch_count)

        if args.eval_first:
            all_results = eval_model_on_test_prompts(model)
            save_path = save_result_path(0)
            json.dump(all_results, open(save_path, "w"))

        for step_idx in pbar:
            selected_idxes = [
                i % len(train)
                for i in range(
                    step_idx * train_batch_size, (step_idx + 1) * train_batch_size
                )
            ]
            selected = [train[i] for i in selected_idxes]

            lm_inputs = [d["prompt"] + " " + d["completion"] for d in selected]
            lm_labels = [d["prompt"] + " " + d["completion"] for d in selected]

            tokenized_inputs = tokenizer(
                lm_inputs, padding=True, truncation=True, return_tensors="pt"
            ).to("cuda")
            tokenized_labels = tokenizer(
                lm_labels, padding=True, truncation=True, return_tensors="pt"
            ).to("cuda")

            # fill -100 for the labels at the prompt
            completion_tokenized_length = [
                len(tokenizer.encode(d["completion"])) for d in selected
            ]
            completion_tokenized_length = torch.tensor(completion_tokenized_length).to(
                "cuda"
            )
            tokenized_labels["input_ids"] = torch.where(
                torch.arange(tokenized_labels["input_ids"].shape[1]).to("cuda")
                < (
                    tokenized_inputs["input_ids"].shape[1]
                    - completion_tokenized_length.unsqueeze(1)
                ),
                -100,
                tokenized_labels["input_ids"],
            )
            labels = tokenized_labels["input_ids"].clone()
            input_ids = tokenized_inputs["input_ids"].clone()

            loss = model(
                input_ids=tokenized_inputs["input_ids"],
                attention_mask=tokenized_inputs["attention_mask"],
                labels=labels,
            ).loss
            loss.backward()
            pbar.set_description("loss: %.4f" % loss.item())

            if (step_idx + 1) % gradient_accumulation_steps == 0:
                optim.step()
                scheduler.step()
                optim.zero_grad()

            if (step_idx + 1) % eval_steps == 0:
                all_results = eval_model_on_test_prompts(model)
                save_step_count = (step_idx + 1) // gradient_accumulation_steps
                save_path = save_result_path(save_step_count)
                json.dump(all_results, open(save_path, "w"))

            if (step_idx + 1) % save_steps == 0:
                save_step_count = (step_idx + 1) // gradient_accumulation_steps
                save_path = os.path.join(
                    training_run_name, "checkpoint_%d" % save_step_count
                )
                print("saving model to", save_path)
                model.save_pretrained(save_path)
