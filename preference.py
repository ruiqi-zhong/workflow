from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor, AdafactorSchedule
from torch import nn
from typing import List
import torch
import json
import random
from tqdm import trange
import numpy as np
import os
from argparse import ArgumentParser
import time


def print_device_place(model):
    for parameters in model.parameters():
        print(parameters.device)


def parallelize_across_device(model):
    num_heads = len(model.block)
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
    print("device_map", device_map)
    model.parallelize(device_map)


class T5ValueEstimator(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = T5ForConditionalGeneration.from_pretrained(model_name).encoder
        parallelize_across_device(self.encoder)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.first_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.last_device = (
            "cuda:{}".format(torch.cuda.device_count() - 1)
            if torch.cuda.is_available()
            else "cpu"
        )
        self.linear = nn.Linear(self.encoder.config.d_model, 1).to(self.last_device)

    def forward(self, texts: List[str]):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.first_device)
        outputs = self.encoder(**inputs)
        hidden_states = outputs.last_hidden_state[:, 0, :]
        return self.linear(hidden_states).squeeze(-1)

    @staticmethod
    def from_pretrained(path):
        size = (
            "xxl"
            if "xxl" in path
            else "xl"
            if "xl" in path
            else "large"
            if "large" in path
            else "base"
            if "base" in path
            else "small"
        )
        model = T5ValueEstimator(f"t5-{size}")
        model.load_state_dict(torch.load(path))
        return model


def inference_on_batch(model, batch):
    texts = []
    if "more_preferred" in batch[0]:
        for key in ["more_preferred", "less_preferred"]:
            for example in batch:
                texts.append(example[key])
        values = model(texts)
        value_first_half, value_second_half = (
            values[: len(values) // 2],
            values[len(values) // 2 :],
        )
        loss = torch.log(1 + torch.exp(value_second_half - value_first_half)).mean()
        values = [
            [value_first_half[i].item(), value_second_half[i].item()]
            for i in range(len(value_first_half))
        ]
        values = np.array(values)
        correctness = [1 if v1 > v2 else 0 for v1, v2 in values]
        return values, loss, correctness
    else:
        for example in batch:
            texts.append(example["text"])
        values = model(texts)
        values = [[value.item()] for value in values]
        values = np.array(values)
        return values, None, None


def preference_inference(model, data_dicts, eval_batch_size):
    all_values, all_correctness, all_loss = [], [], []
    for i in range(0, len(data_dicts), eval_batch_size):
        batch = data_dicts[i : i + eval_batch_size]
        values, loss, correctness = inference_on_batch(model, batch)
        all_values.extend(values)
        if loss is not None:
            all_loss.append(loss.item() * len(correctness))
        all_correctness.extend(correctness)

    all_values = np.array(all_values)
    all_correctness = np.array(all_correctness)
    mean_loss = sum(all_loss) / len(data_dicts)
    mean_accuracy = sum(all_correctness) / len(data_dicts)
    return {
        "mean_loss": float(mean_loss),
        "mean_accuracy": float(mean_accuracy),
        "all_values": all_values.tolist(),
        "all_correctness": all_correctness.tolist(),
    }


def train_preference_model(
    model,
    data_path,
    max_steps,
    train_batch_size,
    eval_batch_size,
    gradient_accumulation_steps,
    save_steps,
    eval_steps,
    save_dir,
):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(data_path) as f:
        data = json.load(f)

    optimizer = Adafactor(
        model.parameters(),
        scale_parameter=True,
        relative_step=True,
        warmup_init=True,
        lr=None,
    )
    lr_scheduler = AdafactorSchedule(optimizer)

    total_mini_steps = max_steps * gradient_accumulation_steps
    global_step_finished = 0

    pbar = trange(total_mini_steps)

    for mini_step_idx in pbar:

        # Train
        model.train()
        random.shuffle(data["train"])
        batch = data["train"][:train_batch_size]
        texts = []
        # put the preferred text first and the less preferred text second in the texts list
        for key in ["more_preferred", "less_preferred"]:
            for example in batch:
                texts.append(example[key])

        _, loss, correctness = inference_on_batch(model, batch)
        pbar.set_description(
            "loss: {:.4f}, acc: {:.4f}".format(
                loss.item(), sum(correctness) / len(correctness)
            )
        )
        loss.backward()
        if (mini_step_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step_finished += 1

        # Eval
        if global_step_finished % eval_steps == 0 and global_step_finished > 0:
            model.eval()
            data_dicts = data["eval"]
            eval_results = preference_inference(model, data_dicts, eval_batch_size)
            with open(os.path.join(save_dir, "eval_results.json"), "w") as f:
                json.dump(eval_results, f)

        # Save
        if global_step_finished % save_steps == 0 and global_step_finished > 0:
            checkpoint_path = os.path.join(
                save_dir, "checkpoint_{}".format(global_step_finished)
            )
            torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the data file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model file",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Path to the directory to save the model",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=20000,
        help="Number of training steps",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size for training",
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients",
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=20000,
        help="Number of steps to save the model",
    )

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=5000,
        help="Number of steps to evaluate the model",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Path to the directory to save the model",
    )

    parser.add_argument(
        "--pred_only", action="store_true", help="Whether to run training."
    )

    parser.add_argument(
        "--pred_path",
        type=str,
        default=None,
        help="Path to the directory to save the model",
    )

    args = parser.parse_args()

    if not args.pred_only:

        if args.save_dir is None:
            save_dir = f"mount/models/{os.path.basename(args.model_name)}_{os.path.basename(args.data_path)}_{time.time()}"

        arg_dict = vars(args)
        arg_dict["save_dir"] = save_dir
        os.mkdir(save_dir)
        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump(arg_dict, f)

        if args.model_path is not None:
            model = T5ValueEstimator.from_pretrained(args.model_path)
        else:
            model = T5ValueEstimator(args.model_name)

        train_preference_model(
            model,
            args.data_path,
            args.max_steps,
            args.train_batch_size,
            args.eval_batch_size,
            args.gradient_accumulation_steps,
            args.save_steps,
            args.eval_steps,
            save_dir,
        )
    else:
        if args.pred_path is None:
            args.pred_path = f"preds/{args.model_path.replace('/', '_')}_{args.data_path.replace('/', '_')}"
        if os.path.exists(args.pred_path):
            print("preds already exist")
            exit(0)
        model = T5ValueEstimator.from_pretrained(args.model_path)
        data = json.load(open(args.data_path))
        eval_results = preference_inference(model, data["eval"], args.eval_batch_size)
        with open(args.pred_path, "w") as f:
            json.dump(eval_results, f)

    create_data_flag = False
    debug_training_flag = False

    if create_data_flag:
        example_train_dict = {
            "more_preferred": "A " * 5,
            "less_preferred": "B " * 5,
            "orig_dict": {},
        }
        example_train_data = [example_train_dict] * 100
        data = {
            "train": example_train_data,
            "eval": example_train_data,
        }
        with open("mount/data/preference_debug.json", "w") as f:
            json.dump(data, f)

    if debug_training_flag:
        model = T5ValueEstimator("t5-small")
        train_preference_model(
            model,
            "mount/data/preference_debug.json",
            max_steps=100,
            train_batch_size=2,
            eval_batch_size=2,
            gradient_accumulation_steps=1,
            save_steps=10,
            eval_steps=10,
            save_dir="mount/models/preference_debug_t5-small",
        )
