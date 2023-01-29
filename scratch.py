from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
import random
import json

MAX_LENGTH = 900
tokenizer = AutoTokenizer.from_pretrained('gpt2')
random.seed(0)

def get_double_prompt(d):
    prompt = 'question: ' + d['question'] + '\ncontext: ' + d['context'] + '\nquestion: ' + d['question'] + '\ncontext: ' + d['context'] + '\nanswer:'
    return prompt

def get_question_left_prompt(d):
    prompt = 'question: ' + d['question'] + '\ncontext: ' + d['context'] + '\nanswer:'
    return prompt

def get_question_right_prompt(d):
    prompt = 'context: ' + d['context'] + '\nquestion: ' + d['question'] + '\nanswer:'
    return prompt


def get_prompt_and_completion(d, config):
    completion = d['answers']['text'][0]
    if config == 'question_left':
        prompt = get_question_left_prompt(d)
    elif config == 'question_right':
        prompt = get_question_right_prompt(d)
    elif config == 'double':
        prompt = get_double_prompt(d)
    return {
        'prompt': prompt.strip(),
        'completion': completion.strip(),
        'config': config,
        'orig_d': d
    }

def filter_data(ds):
    all_ds = []
    for d in ds:
        double_prompt_length = tokenizer(get_double_prompt(d), return_length=True)['length'][0]
        if double_prompt_length > MAX_LENGTH:
            continue
        all_ds.append(d)
    return all_ds

tok = AutoTokenizer.from_pretrained('gpt2')
train_dataset = load_dataset('squad', split='train')
print('orig size', len(train_dataset))
train_dataset = filter_data(train_dataset)
print('filtered size', len(train_dataset))

all_test = []
in_distribution_test = load_dataset('squad', split='validation')
print('orig size', len(in_distribution_test))
in_distribution_test = filter_data(in_distribution_test)
print('filtered size', len(in_distribution_test))
in_distribution_test = random.sample(in_distribution_test, 3000)
for d in in_distribution_test:
    d['from'] = 'squad'
    all_test.append(d)
domains = ['amazon', 'new_wiki', 'nyt', 'reddit']

for domain in domains:
    dataset = load_dataset('squadshifts', domain)['test']
    print('orig size', len(dataset))
    dataset = filter_data(dataset)
    print('filtered size', len(dataset))
    dataset = random.sample(dataset, 3000)
    for d in dataset:
        d['from'] = domain
        all_test.append(d)

for config in ['question_left', 'question_right', 'double']:
    train_dicts = []
    for d in train_dataset:
        train_dicts.append(get_prompt_and_completion(d, config))
    test_dicts = []
    for d in all_test:
        test_dicts.append(get_prompt_and_completion(d, config))
    all_dicts = {
        'train': train_dicts,
        'eval': test_dicts
    }

    json.dump(all_dicts, open('mount/data/squadshifts_experiment_{}.json'.format(config), 'w'))
