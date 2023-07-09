import random
from transformers.trainer_callback import TrainerCallback
from inference import sample_batched
import argparse
import json
from trainer_utils import (
    PromptCompletionDataset,
    get_seq2seq_training_args,
    get_seq2seq_collator,
)
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
)
import os
from argparse import Namespace

os.environ["CUDA_HOME"] = "/usr/local/cuda"
print("Using %d GPUs." % torch.cuda.device_count())
mount_dir = "/model_mount/"
if not os.path.exists(mount_dir):
    mount_dir = "/scratch/users/ruiqi-zhong/workflow/mount/"


def print_device_place(model):
    for parameters in model.parameters():
        print(parameters.device)


def fit_bit(model):
    for n, parameters in model.named_parameters():
        if "bias" not in n and "norm" not in n.lower():
            parameters.requires_grad = False


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_init_name",
        type=str,
        default=None,
        help="model name relative to mount/models",
    )
    parser.add_argument(
        "--model_init_path",
        type=str,
        default=None,
        help="model path relative to the current directory, or the model name from huggingface like t5-small. NOTICE: in the HOFVARPNIR setup, the model will be downloaded EVERY TIME you launch a new job (rather than loading from a cache), so it might be more efficient to load from a local directory.",
    )
    parser.add_argument(
        "--data_name", type=str, default="data", help="data name relative to mount/data"
    )
    parser.add_argument(
        "--training_run_name",
        type=str,
        default=None,
        help="the folder name to save the model and the evaluation results, relative to mount/models",
    )
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--save_steps", type=int, default=None, help="save the model every save_steps"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.001,
        help="temperature for sampling during evaluation",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="number of samples to generate for each input during evaluation",
    )
    parser.add_argument(
        "--no_train",
        action="store_true",
        help="if set, do not train the model, only evaluate the model",
    )
    parser.add_argument(
        "--debug", action="store_true", help="if set, only train for 10 steps"
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="overwrite output dir"
    )
    parser.add_argument("--num_epochs", type=int, default=4)

    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--pred_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    return parser


def get_args():
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    return args, unknown


def run(args, unknown=None):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model_init_path = args.model_init_path
    if model_init_path is None:
        model_init_path = mount_dir + "models/%s" % args.model_init_name

    data_path = mount_dir + "data/%s.json" % args.data_name
    if not os.path.exists(data_path):
        print("Error: %s does not exist" % data_path)
        exit(0)

    # defining the prompt completion list
    data = json.load(open(data_path))
    train, test = data["train"], data["eval"][: 100 if args.debug else None]
    test_prompts = [d["prompt"] for d in test]
    demonstrations = [d["completion"] for d in test]

    models_dir = mount_dir + "models/"
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    if not args.no_train:
        if args.training_run_name is None:
            if args.model_init_path is not None:
                args.training_run_name = f"{args.model_init_path.replace('/', '_')}_{args.data_name}_{args.seed}"
            else:
                args.training_run_name = (
                    f"{args.model_init_name}_{args.data_name}_{args.seed}"
                )

        training_run_name = models_dir + args.training_run_name
        if os.path.exists(training_run_name):
            if args.overwrite_output_dir:
                print("Warning: %s already exists" % training_run_name)
                os.system("rm -rf %s" % training_run_name)
            else:
                print("Error: %s already exists" % training_run_name)
                exit(0)
        os.mkdir(training_run_name)
    else:
        if args.pred_dir is None:
            if args.model_init_path is not None:
                args.pred_dir = f"{args.model_init_path.replace('/', '_')}_{args.data_name}_{args.seed}"
            else:
                args.pred_dir = f"{args.model_init_name}_{args.data_name}_{args.seed}"
        save_dir = os.path.join("preds", args.pred_dir)
        if os.path.exists(save_dir):
            if args.overwrite_output_dir:
                print("Warning: %s already exists" % save_dir)
                os.system("rm -rf %s" % save_dir)
            else:
                print("Error: %s already exists" % save_dir)
                exit(0)

    num_epochs = args.num_epochs
    warmup_steps = args.warmup_steps
    train_batch_size = args.train_batch_size
    eval_batch_size = args.train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps

    effective_training_batch_size = train_batch_size * gradient_accumulation_steps
    num_training_examples = len(train)
    if args.max_steps is None:
        args.max_steps = (
            num_training_examples * num_epochs // effective_training_batch_size + 1
        )
    max_steps = args.max_steps if not args.debug else 10

    if args.save_steps is None:
        args.save_steps = args.max_steps // num_epochs
    save_steps = args.save_steps

    no_train = args.no_train
    if args.eval_steps is None:
        args.eval_steps = args.max_steps // num_epochs
    temperature = args.temperature
    n_samples = args.n_samples

    tokenizer_path = mount_dir + "models/t5-small-cp_tokenizer"
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "t5-small"
    # tokenizer_path = 't5-small'

    print("loading model from %s" % model_init_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_init_path, device_map="balanced"
    )
    model = model.to(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.model_max_length = 1024
    model_tokenizer = (model, tokenizer)

    def eval_model_on_test_prompts(target_model):
        model_tokenizer = (target_model, tokenizer)
        sampled_results = sample_batched(
            model_tokenizer,
            test_prompts,
            temperature=temperature,
            n=n_samples,
            bsize=eval_batch_size,
        )

        all_results = []
        for result_dict, demonstration, orig_d in zip(
            sampled_results, demonstrations, test
        ):
            d = {
                "prompt": result_dict["prompt"],
                "generations": result_dict["generations"],
                "demonstration": demonstration,
                "orig_d": orig_d,
            }
            all_results.append(d)
        return all_results

    def save_result_path(step):
        hyp_str = "temperature=%.2f_n=%d_step=%d" % (temperature, n_samples, step)
        save_path = os.path.join(training_run_name, "%s.json" % hyp_str)
        return save_path

    class PredAfterEvalCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, model, tokenizer, **kwargs):
            all_results = eval_model_on_test_prompts(model)
            save_path = save_result_path(state.global_step)
            json.dump(all_results, open(save_path, "w"))

    if not no_train:
        arg_dict = vars(args)
        json.dump(arg_dict, open(training_run_name + "/training_args.json", "w"))

        train_dataset = PromptCompletionDataset(train, model_tokenizer)
        eval_dataset = PromptCompletionDataset(test, model_tokenizer)

        # get a bunch of training arguments
        training_args = get_seq2seq_training_args(
            training_run_name=training_run_name,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_steps=save_steps,
            eval_steps=args.eval_steps,
            seed=args.seed,
        )
        data_collator = get_seq2seq_collator(model_tokenizer)

        # training loop
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[PredAfterEvalCallback()],
        )
        trainer.evaluate()
        trainer.train()

    hyp_str = "temperature=%.2f_n=%d_step=%d" % (
        args.temperature,
        args.n_samples,
        max_steps,
    )
    if not no_train:
        save_path = os.path.join(training_run_name, "%s.json" % hyp_str)
    else:
        save_dir = os.path.join("preds", args.pred_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        arg_dict = vars(args)
        json.dump(arg_dict, open(save_dir + "/pred_args.json", "w"))
        save_path = os.path.join(save_dir, "%s.json" % hyp_str)
        if os.path.exists(save_path):
            print(f"{save_path} already exists, skipping")
            exit(0)
        else:
            with open(save_path, "w") as f:
                f.write("")

    sampled_results = sample_batched(
        model_tokenizer,
        test_prompts,
        temperature=args.temperature,
        n=args.n_samples,
        bsize=eval_batch_size,
    )
    all_results = []
    for result_dict, demonstration, orig_d in zip(
        sampled_results, demonstrations, test
    ):
        d = {
            "prompt": result_dict["prompt"],
            "generations": result_dict["generations"],
            "demonstration": demonstration,
            "orig_d": orig_d,
        }
        all_results.append(d)

    print("saving results to %s" % save_path)
    json.dump(
        all_results,
        open(save_path, "w"),
    )


def run_w_kwargs(**kwargs):
    parser = get_parser()
    args = parser.parse_args([])
    for k, v in kwargs.items():
        args.__dict__[k] = v
    print("running with args:")
    print(args)
    run(args, [])


if __name__ == "__main__":
    args, unknown = get_args()
    # testing running with kwargs
    run_w_kwargs(
        model_init_path=args.model_init_path,
        data_name=args.data_name,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        overwrite_output_dir=args.overwrite_output_dir,
    )
    # testing running with args
    run(args, unknown)
