# workflow


## Disclaimer

If your experiments produce wrong results because there is a bug in my code, it's not my responsibility. 

## Onboard 

You need to follow the onboarding instruction of HOFVARPNIR in order to 


## High level overview

This github is mostly built around fine-tuning the T5 series, which is, surprising, still the best open-sourced pre-trained model after more than 3 years of its initial arxiv release and outperforms other public models 17x larger. 
If you want a fine-tunable LM with high performance, your default choice should be ```google/flan-t5-xxl``` (which is mostly instruction-tuned based on AI2's repo with 1600+ tasks + something from Google).


### Prompt Completion

Below is the command I usually used: 

```ctl job run --name 0216synexplain  --container rz2383/deepspeed:1229a --login --high-priority --shared-host-dir /home/rzhong --gpu 4 --command 'bash -c "cd data/workflow; python3 seq2seq.py --model_init_name flan-t5-xxl --training_run_name 0216syn_explain --gradient_accumulation_steps 2 --train_batch_size 8 --max_steps 6001 --data synthetic_explain  --eval_steps 2000 --temperature 0.001 --n_samples 1"'```

It uses 4 x 80 GPUs and a batch size of 16 (8 examples per step and accumuate 2 step for each gradient update). Here are the explanations for each arguments.

The command will 
- indicate that we are using bash with ```bash -c```
- change directory into the ```data/workflow```. notice that I put this repo (```workflow```) in my home directory, but when we launch a job, the home directory will be mounted to the ```data/``` directory, so that's why I need to cd into ```data/workflow``` rather than just ```workflow```
- then run the fine-tuning job with the ```seq2seq.py`` file.

```
optional arguments:
  -h, --help            show this help message and exit
  --model_init_name MODEL_INIT_NAME
                        model name relative to mount/models
  --model_init_path MODEL_INIT_PATH
                        model path relative to the current directory, or the model name from huggingface like t5-small. NOTICE: in the HOFVARPNIR setup, the model will
                        be downloaded EVERY TIME you launch a new job (rather than loading from a cache), so it might be more efficient to load from a local directory.
  --data_name DATA_NAME
                        data name relative to mount/data
  --training_run_name TRAINING_RUN_NAME
                        the folder name to save the model and the evaluation results, relative to mount/models
  --warmup_steps WARMUP_STEPS
  --max_steps MAX_STEPS
  --train_batch_size TRAIN_BATCH_SIZE
  --eval_batch_size EVAL_BATCH_SIZE
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
  --save_steps SAVE_STEPS
                        save the model every save_steps
  --temperature TEMPERATURE
                        temperature for sampling during evaluation
  --n_samples N_SAMPLES
                        number of samples to generate for each input during evaluation
  --no_train            if set, do not train the model, only evaluate the model
  --eval_steps EVAL_STEPS
```

The data format is a .json file containing a dictionary 
```
{'train': [{'prompt': '<prompt>', 'completion': '<completion>'}, ...], 'eval': [{...}, {...}]}
```
Only the key ```'prompt'``` and ```'completion'``` are required for each dictionary to make fine-tuning to work. The file I provided contains other info for logging purposes. The design tries to mirror the openai design as much as possible, which is the level of abstraction I need for most of my current NLP application research. See ```mount/data/data.json``` for an example.

### Binary Classification and Regression

Command that I used to fine-tune T5 to perform classification or regression

```ctl job run --name hardnewsgold --gpus-type "NVIDIA-A100-SXM4-80GB"  --container rz2383/deepspeed:1229a --login --high-priority --shared-host-dir /home/rzhong --gpu 4 --command 'bash -c "cd data/workflow/; python3 clf.py --model_pretrain_name "google/flan-t5-xl" --data_name 1226verifier_ft_all --max_steps 2001 --eval_steps 2000 --save_steps 2000"'```

The data format is of the following format:
```
{'train': [{'input': '<input>', 'target': '<target>'}, ...], 'eval': [{...}, {...}]}
```

where each target is a score between [0, 1] (so you need to renormalize before and after running the model for prediction). I obtained theses scores by taking the softmax for the logits of the ```yes``` and ```no``` tokens, so you might want to wrap your raw input with a template that maps your input to a yes/no question. 
For example, if your goal is to classify whether an ```<article>``` is sports, then the input should be

```
Classify whether the article is sports releated. Output yes or no.

article: <article>
output:
```

See ```mount/data/1226verifier_ft_all.json``` for an example.