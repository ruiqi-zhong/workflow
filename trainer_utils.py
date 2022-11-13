from itertools import accumulate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import numpy as np
from torch.utils.data import Dataset


class PromptCompletionDataset(Dataset):

    def __init__(self, prompt_completion_list, model_tokenizer, random_seed=0):
        num = len(prompt_completion_list)
        self.prompt_completion_list = prompt_completion_list
        r = np.random.RandomState(random_seed)
        perm = r.permutation(num)
        self.ixes = perm
        self.tokenizer = model_tokenizer[1]

    def __len__(self):
        return self.ixes.size

    def __getitem__(self, idx):
        idx = self.ixes[idx]
        prompt_completion = self.prompt_completion_list[idx]
        prompt, completion = prompt_completion['prompt'], prompt_completion['completion']

        return {"input_ids":self.tokenizer(prompt)["input_ids"], "labels":self.tokenizer(completion)["input_ids"]}


def get_seq2seq_training_args(
        training_run_name, warmup_steps=2000, max_steps=10000, 
        train_batch_size=32, eval_batch_size=32, 
        gradient_accumulation_steps=1, save_steps=1000,
        eval_steps=100,
        deepspeed_json_path=None
    ):
    return Seq2SeqTrainingArguments(
        training_run_name,
        evaluation_strategy='steps',
        eval_steps=eval_steps,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        save_total_limit=5,
        warmup_steps=warmup_steps,
        predict_with_generate=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        save_steps=save_steps,
        push_to_hub=False,
        optim='adafactor',
        deepspeed=deepspeed_json_path
    )


def get_seq2seq_collator(model_tokenizer):
    model, tokenizer = model_tokenizer
    return DataCollatorForSeq2Seq(tokenizer, model=model)