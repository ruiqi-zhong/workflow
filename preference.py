from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    T5Model,
    T5ForConditionalGeneration,
)
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


SEP_SEGMENT = "\n\nRESPONSE STARTS: "


def parallelize_across_device(model):
    # if the model has the attribute of encoder
    if hasattr(model, "encoder"):
        num_heads = len(model.encoder.block)
    else:
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


class T5EncoderValueEstimator(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = T5ForConditionalGeneration.from_pretrained(
            model_name
        ).encoder.to(torch.bfloat16)
        self.seq2seq_flag = False
        parallelize_across_device(self.encoder)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.first_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.last_device = (
            "cuda:{}".format(torch.cuda.device_count() - 1)
            if torch.cuda.is_available()
            else "cpu"
        )
        self.linear = (
            nn.Linear(self.encoder.config.d_model, 1)
            .to(self.last_device)
            .to(torch.bfloat16)
        )

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
        model = T5EncoderValueEstimator(f"google/flan-t5-{size}")
        model.load_state_dict(torch.load(path))
        return model


class T5Seq2SeqValueEstimator(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = T5Model.from_pretrained(model_name).to(torch.bfloat16)
        self.seq2seq_flag = True
        parallelize_across_device(self.model)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.first_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.last_device = (
            "cuda:{}".format(torch.cuda.device_count() - 1)
            if torch.cuda.is_available()
            else "cpu"
        )
        self.linear = (
            nn.Linear(self.model.encoder.config.d_model, 1)
            .to(self.last_device)
            .to(torch.bfloat16)
        )

    def forward(self, texts: List[str]):
        encoder_texts = [t.split(SEP_SEGMENT)[0] for t in texts]
        decoder_texts = [t.split(SEP_SEGMENT)[1] for t in texts]

        deduped_encoder_texts = list(set(encoder_texts))
        # idx to encoder text idx
        encoder_text_idx = [deduped_encoder_texts.index(t) for t in encoder_texts]

        encoder_inputs = self.tokenizer(
            deduped_encoder_texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.first_device)
        encoder_outputs = self.model.encoder(**encoder_inputs)
        # inflate hidden states
        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state[
            encoder_text_idx
        ]

        # decoder
        decoder_inputs = self.tokenizer(
            decoder_texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.first_device)
        decoder_outputs = self.model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_inputs.input_ids,
            output_hidden_states=True,
        )
        decoder_last_layer_states = decoder_outputs.last_hidden_state
        decoder_lengths = decoder_inputs.attention_mask.sum(-1)
        decoder_last_state = decoder_last_layer_states[
            torch.arange(len(decoder_lengths)), decoder_lengths - 1
        ]
        return self.linear(decoder_last_state).squeeze(-1)

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
        model = T5Seq2SeqValueEstimator(f"google/flan-t5-{size}")
        model.load_state_dict(torch.load(path))
        return model


PROMPT_PREFIX = (
    f"Is the assisstant's response after {SEP_SEGMENT.strip()} good? Answer yes or no."
)


class T5PromptedValueEstimator(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            torch.bfloat16
        )
        self.seq2seq_flag = True
        parallelize_across_device(self.model)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.first_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.last_device = (
            "cuda:{}".format(torch.cuda.device_count() - 1)
            if torch.cuda.is_available()
            else "cpu"
        )
        self.vocab = self.tokenizer.get_vocab()
        self.yes_id = self.vocab["▁yes"]

    def forward(self, texts: List[str]):
        texts = [PROMPT_PREFIX + t for t in texts]
        tokenized_texts = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.first_device)
        num_texts = len(texts)
        starts = torch.tensor(
            [[self.model.config.decoder_start_token_id]] * num_texts
        ).to(self.first_device)
        output = self.model(
            input_ids=tokenized_texts["input_ids"],
            attention_mask=tokenized_texts["attention_mask"],
            decoder_input_ids=starts,
        )
        logits = output.logits[:, 0, self.yes_id]
        return logits

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
        model = T5PromptedValueEstimator(f"google/flan-t5-{size}")
        model.load_state_dict(torch.load(path))
        return model


def inference_on_batch(model, batch, surrogate_loss=False):
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
        if not surrogate_loss:
            loss = torch.log(1 + torch.exp(value_second_half - value_first_half)).mean()
        else:
            surrogate_labels = torch.cat(
                [torch.ones(len(value_first_half)), torch.zeros(len(value_second_half))]
            ).to(value_first_half.device)
            binary_clf_loss_fn = nn.BCEWithLogitsLoss()
            loss = binary_clf_loss_fn(values, surrogate_labels)

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
        if correctness is not None:
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
    constant_lr=False,
    surrogate_fraction=0.0,
    save_best=False,
):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(data_path) as f:
        data = json.load(f)

    if constant_lr:
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=1e-3,
        )
        lr_scheduler = None
    else:
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
    best_accuracy = 0.0

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
        use_surrogate = global_step_finished / total_mini_steps < surrogate_fraction
        _, loss, correctness = inference_on_batch(
            model, batch, surrogate_loss=use_surrogate
        )
        pbar.set_description(
            "loss: {:.4f}, acc: {:.4f}".format(
                loss.item(), sum(correctness) / len(correctness)
            )
        )
        loss.backward()
        if (mini_step_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()
            global_step_finished += 1

        # Eval
        if global_step_finished % eval_steps == 0 and global_step_finished > 0:
            model.eval()
            data_dicts = data["eval"]
            eval_results = preference_inference(model, data_dicts, eval_batch_size)
            accuracy = eval_results["mean_accuracy"]
            if save_best and accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model"))

            with open(
                os.path.join(save_dir, f"{global_step_finished}_eval_results.json"), "w"
            ) as f:
                json.dump(eval_results, f)

        # Save
        if global_step_finished % save_steps == 0 and global_step_finished > 0:
            checkpoint_path = os.path.join(
                save_dir, "checkpoint_{}".format(global_step_finished)
            )
            torch.save(model.state_dict(), checkpoint_path)


choice2class = {
    "encoder": T5EncoderValueEstimator,
    "seq2seq": T5Seq2SeqValueEstimator,
    "prompted": T5PromptedValueEstimator,
}


if __name__ == "__main__":

    DEBUG_FLAG = False

    if not DEBUG_FLAG:
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
            default=10000,
            help="Number of training steps",
        )

        parser.add_argument(
            "--train_batch_size",
            type=int,
            default=8,
            help="Batch size for training",
        )

        parser.add_argument(
            "--eval_batch_size",
            type=int,
            default=8,
            help="Batch size for evaluation",
        )

        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=2,
            help="Number of steps to accumulate gradients",
        )

        parser.add_argument(
            "--save_steps",
            type=int,
            default=None,
            help="Number of steps to save the model",
        )

        parser.add_argument(
            "--eval_steps",
            type=int,
            default=None,
            help="Number of steps to evaluate the model",
        )

        parser.add_argument(
            "--save_dir",
            type=str,
            default=None,
            help="Path to the directory to save the model",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="Random seed",
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

        parser.add_argument(
            "--architecture",
            type=str,
            choices=list(choice2class.keys()),
            help="Architecture to use, whether to put the response into the decoder or not.",
            default=None,
        )

        parser.add_argument(
            "--constant_lr",
            action="store_true",
            help="Whether to use a constant learning rate",
        )

        parser.add_argument(
            "--surrogate_fraction",
            type=float,
            default=0.0,
            help="Fraction of training steps that use the surrogate loss",
        )

        parser.add_argument(
            "--save_best",
            action="store_true",
            help="Whether to save the best model",
        )

        args = parser.parse_args()

        if not os.path.exists(args.data_path):
            raise ValueError("data_path does not exist")

        print("loading model")
        if args.model_path is not None:
            assert (
                "seq2seq" in args.model_path
                or "encoder" in args.model_path
                or "prompted" in args.model_path
            ), "model_path must contain either seq2seq or encoder"
            for key, value in choice2class.items():
                if key in args.model_path:
                    args.architecture = key
                    break
        else:
            assert args.architecture is not None

        if not args.pred_only:
            if args.save_steps is None:
                args.save_steps = args.max_steps
            if args.eval_steps is None:
                args.eval_steps = args.max_steps // 10

        model_class = choice2class[args.architecture]
        if args.model_path is not None:
            model = model_class.from_pretrained(args.model_path)
        else:
            model = model_class(args.model_name)
        print("model loaded")

        if not args.pred_only:

            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

            if args.save_dir is None:
                save_dir = f"mount/models/{os.path.basename(args.model_name)}_{os.path.basename(args.data_path)}_totalbszie={args.train_batch_size * args.gradient_accumulation_steps}_seed={args.seed}_architecture={args.architecture}_lr={args.constant_lr}_surrogate={args.surrogate_fraction:.2f}"

            arg_dict = vars(args)
            arg_dict["save_dir"] = save_dir
            if os.path.exists(save_dir):
                raise ValueError(f"save_dir {save_dir} already exists")
            os.mkdir(save_dir)
            print("saving to", save_dir)
            with open(os.path.join(save_dir, "args.json"), "w") as f:
                json.dump(arg_dict, f)

            train_preference_model(
                model=model,
                data_path=args.data_path,
                max_steps=args.max_steps,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                save_steps=args.save_steps,
                eval_steps=args.eval_steps,
                save_dir=save_dir,
                constant_lr=args.constant_lr,
                surrogate_fraction=args.surrogate_fraction,
                save_best=args.save_best,
            )
        else:
            if args.pred_path is None:
                args.pred_path = f"preds/{args.model_path.replace('/', '_')}_{args.data_path.replace('/', '_')}"
            if os.path.exists(args.pred_path):
                print("preds already exist")
                exit(0)
            with open(args.data_path) as f:
                data = json.load(f)
            with torch.no_grad():
                model.eval()
                eval_results = preference_inference(
                    model, data["eval"], args.eval_batch_size
                )
            with open(args.pred_path, "w") as f:
                json.dump(eval_results, f)

    else:
        create_data_flag = True
        debug_training_flag = True
        cls = T5PromptedValueEstimator

        if create_data_flag:
            example_train_dict = {
                "more_preferred": "X" * 5 + SEP_SEGMENT + "A " * 5,
                "less_preferred": "X" * 5 + SEP_SEGMENT + "B " * 5,
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
            model = cls("t5-small")
            train_preference_model(
                model,
                "mount/data/preference_debug.json",
                max_steps=1000,
                train_batch_size=2,
                eval_batch_size=2,
                gradient_accumulation_steps=1,
                save_steps=500,
                eval_steps=100,
                save_dir="mount/models/preference_debug_t5-small-seq2seq",
            )
