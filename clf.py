print('classification script starts running')

import os
import random
from itertools import chain
from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch
import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score
import tqdm
import pickle as pkl
import json
from scipy.stats import pearsonr, spearmanr
import argparse
import sys
sys.path.append('./')
from eval_utils import evaluate_pred_path
import shutil

num_device = torch.cuda.device_count()
first_device_name = 'cuda:0'
last_device_name = 'cuda:%d' % (torch.cuda.device_count() - 1)

# def parallelize_across_device(model):
#     num_heads = len(model.block)
#     if num_device > 1:
#         other_device_alloc = num_heads // num_device + 1
#         first_device_alloc = num_heads - (num_device - 1) * other_device_alloc
#         device_map = {}
#         cur = 0
#         end = max(cur + first_device_alloc, 1)
#         device_map[0] = list(range(cur, end))
#         cur = end
#         for i in range(1, num_device):
#             end = min(cur + other_device_alloc, num_heads)
#             device_map[i] = list(range(cur, end))
#             cur += other_device_alloc
#         print('device_map', device_map)
#         model.parallelize(device_map)
#     else:
#         model.to(first_device_name)

# def load_t5encoder(pretrain_model):
#     model = T5ForConditionalGeneration.from_pretrained(pretrain_model)
#     model = model.encoder
#     parallelize_across_device(model)
#     d_model = model.config.d_model
#     return {
#         'model': model,
#         'd_model': d_model
#     }


# class T5ForSeqClassification_OBSOLETE(nn.Module):

#     def __init__(self, pretrain_model='t5-base'):
#         super().__init__()
#         self.pretrain_model = pretrain_model
#         m_dict = load_t5encoder(pretrain_model)
#         self.model = m_dict['model']
#         self.clf_layer = nn.Linear(m_dict['d_model'], 2).to(last_device_name)
#         self.tok = AutoTokenizer.from_pretrained(pretrain_model)

#     def forward(self, input_ids, attention_mask, labels=None):
#         encoder_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_state = encoder_outputs[0]
#         logits = self.clf_layer(last_hidden_state[:, 0, :])
#         output_dict = {'logits': logits}
#         return output_dict
    
#     def save_pretrained(self, save_path):
#         os.makedirs(save_path, exist_ok=True)
#         torch.save(self.state_dict(), os.path.join(save_path, 'model.pt'))


lsm = torch.nn.LogSoftmax(dim=-1)

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


class T5ForSeqClassification(nn.Module):
    
        def __init__(self, pretrain_model='t5-base'):
            super().__init__()
            self.pretrain_model = pretrain_model
            self.model = T5ForConditionalGeneration.from_pretrained(pretrain_model)
            parallelize_across_device(self.model)
            try:
                self.tok = AutoTokenizer.from_pretrained(pretrain_model)
            except Exception as e:
                print(e)
                if 't5' in pretrain_model:
                    self.tok = T5Tokenizer.from_pretrained('mount/models/t5tok')
            

            self.vocab = self.tok.get_vocab()
            self.yes_id, self.no_id = self.vocab['▁yes'], self.vocab['▁no']
    
        def forward(self, input_ids, attention_mask, labels=None):
            starts = torch.tensor([[self.model.config.decoder_start_token_id]] * len(input_ids)).to(first_device_name)
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=starts)
            logits = lsm(output.logits[:, 0, [self.no_id, self.yes_id]])
            return {'logits': logits}

        def save_pretrained(self, save_path):
            self.model.save_pretrained(save_path)


class SeqClf(nn.Module):
    
    def __init__(self, pretrain_model, max_length=512, load_path=None):
        super().__init__()
        self.pretrain_model = pretrain_model
        if 't5' in pretrain_model:
            self.model = T5ForSeqClassification(pretrain_model=pretrain_model)
            if load_path is not None:
                self.model.model.from_pretrained(load_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(pretrain_model)
            if load_path is not None:
                self.model = self.model.from_pretrained(load_path)
            self.model.to(first_device_name)
        try:
            self.tok = AutoTokenizer.from_pretrained(pretrain_model)
        except Exception as e:
            print(e)
            if 't5' in pretrain_model:
                self.tok = T5Tokenizer.from_pretrained('mount/models/t5tok')
        self.max_length = max_length
        self.loss_fn = nn.KLDivLoss()
        self.tok.model_max_length = max_length
        self.lsm = nn.LogSoftmax(dim=-1)
        
    def forward(self, data_dicts):
        input_texts = [d['input'] for d in data_dicts]
        inputs = self.tok(input_texts, return_tensors='pt', truncation=True, padding=True).to(first_device_name)
        labels = None
        if data_dicts[0].get('target') is not None:
            labels = torch.tensor([[float(1 - d['target']), float(d['target'])] for d in data_dicts]).to(first_device_name)#.to(last_device_name)
        model_outputs = self.model(**inputs)
        if 't5' not in self.pretrain_model:
            logits = model_outputs.logits
        else:
            logits = model_outputs['logits']
        logits = self.lsm(logits)
        model_output_dict = {'logits': logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            model_output_dict['loss'] = loss
        return model_output_dict

    def evaluate_texts(self, texts, eval_bsize=32):
        self.model.eval()
        with torch.no_grad():
            all_logits = []
            cur_start = 0
            pbar = tqdm.tqdm(total=len(texts))
            while cur_start < len(texts):
                input_dicts = [{'input': t} for t in texts[cur_start:cur_start + eval_bsize]]
                model_output_dict = self.forward(input_dicts)
                logits = lsm(model_output_dict['logits'].detach().cpu()).numpy().tolist()
                all_logits.extend(logits)
                cur_start += eval_bsize
                pbar.update(eval_bsize)
            assert len(all_logits) == len(texts)
            
            return {
                'logits': np.array(all_logits),
                'scores': 1 / (1 + np.e ** (np.array(all_logits)[:, 0] - np.array(all_logits)[:, 1]))
            }

    def evaluate_dicts(self, data_dicts, eval_bsize, scores_only=False):
        eval_texts = [d['input'] for d in data_dicts]
        scores = self.evaluate_texts(eval_texts, eval_bsize=eval_bsize)['scores']
        if not scores_only:
            saved_preds = [{'orig_d': data_dicts[i], 'pred_score': scores[i]} for i in range(len(eval_texts))]
        else:
            saved_preds = [{'pred_score': scores[i]} for i in range(len(eval_texts))]
        return saved_preds
    
    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path)


def train_and_eval(
    all_data_dicts, model, max_length=512,
    num_steps=None, save_path_prefix=None, train_bsize=32, accumulate=1, 
    eval_bsize=32, train_last_lyer_only=False, save_every=1000, eval_every=1000, warmup_steps=5000, save_best_metric=None):

    train_data_dicts = all_data_dicts['train']
    total_datapoints = len(train_data_dicts)

    no_decay = ['bias', 'LayerNorm.weight']
    def frozen_param_name(n):
        if train_last_lyer_only:
            if 'roberta' in model.pretrain_model:
                if 'lm_head' in n:
                    return False
                if 'pooler' in n:
                    return False
                return True
            elif 't5' in model.pretrain_model:
                if 'classifier' in n:
                    return False
                return True
            elif 'bert' in model.pretrain_model:
                if 'cls' in n:
                    return False
                return True
        return False

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not frozen_param_name(n)],
        'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not frozen_param_name(n)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)
    if num_steps is None:
        num_steps = max(num_steps, total_datapoints // train_bsize * 3)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_steps)
    
    best_performance = 0
    pbar = tqdm.trange(num_steps * accumulate)
    loss_moving_avg = 0
    for mini_step in pbar:
        pbar.set_description('Step %d' % (mini_step // accumulate))
        random.shuffle(train_data_dicts)
        outputs = model(train_data_dicts[:train_bsize])
        loss = outputs['loss']
        loss.backward()
        loss_moving_avg = loss_moving_avg * 0.95 + loss.item() * 0.05
        pbar.set_postfix(loss='%.4f' % loss_moving_avg)

        global_step_finishes = ((mini_step + 1) % accumulate) == 0

        if global_step_finishes:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step_finished = (mini_step + 1) // accumulate

        if save_path_prefix is not None:
            if save_every is not None and global_step_finishes !=0 and global_step_finished % save_every == 0:
                save_path = save_path_prefix + '-%d' % global_step_finished
                model.model.save_pretrained(save_path)

            if eval_every is not None and global_step_finished % eval_every == 0:
                saved_preds = model.evaluate_dicts(all_data_dicts['eval'], eval_bsize)
                json.dump(saved_preds, open(save_path_prefix + '-%d-preds.json' % global_step_finished, 'w'))
                
                if save_best_metric is not None:
                    result_dict = evaluate_pred_path(save_path_prefix + '-%d-preds.json' % global_step_finished)
                    if result_dict[save_best_metric] > best_performance:
                        best_performance = result_dict[save_best_metric]
                        print('At step %d, best performance is %.4f' % (global_step_finished, best_performance))
                        model_save_path = save_path_prefix + '-best-ckpt'
                        if os.path.exists(model_save_path):
                            shutil.rmtree(model_save_path)
                        model.model.save_pretrained(model_save_path)

        model.train()
    return model

# def evaluate_pred_path(pred_path):
#     preds = json.load(open(pred_path))
#     pred_scores = [p['pred_score'] for p in preds]
#     orig_scores = [p['orig_d']['target'] for p in preds]
#     discrete_gold_label = [1 if s > 0.5 else 0 for s in orig_scores]
#     discrete_pred_label = [1 if s > 0.5 else 0 for s in pred_scores]
#     result_dict = {
#         'accuracy': accuracy_score(discrete_gold_label, discrete_pred_label),
#         'auc': roc_auc_score(discrete_gold_label, pred_scores),
#         'spearman': spearmanr(orig_scores, pred_scores)
#     }
#     return result_dict

        

if __name__ == '__main__':

    # command for debugging
    # training
    # python3 clf.py --model_pretrain_name t5-small --data clf_debug --max_steps=2000 --save_steps=2000
    print(sys.argv)
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_pretrain_name', type=str, required=True)
    parser.add_argument('--model_init_path', type=str, default=None)
    parser.add_argument('--data_name', type=str, default='clf_debug')
    parser.add_argument('--training_run_name', type=str, default='clf_debug')
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=3000)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--save_steps', type=int, default=None)
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--no_ft_head_first', action='store_true')
    parser.add_argument('--eval_steps', type=int, default=3000)
    parser.add_argument('--pred_save_path', type=str, default=None)
    parser.add_argument('--scores_only', action='store_true')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--save_best_metric', type=str, default=None)

    args, unknown = parser.parse_known_args()

    model = SeqClf(args.model_pretrain_name, load_path=args.model_init_path, max_length=args.max_length)
    data_path = 'mount/data/%s.json' % args.data_name
    data_dicts = json.load(open(data_path))

    save_dir = 'mount/models/%s' % args.training_run_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    arg_dict = vars(args)
    json.dump(arg_dict, open(os.path.join(save_dir, 'args.json'), 'w'))

    save_path_prefix = os.path.join(save_dir, 'checkpoint')
    
    # the first few steps only train the last layer
    if not args.no_train:
        if not args.no_ft_head_first:
            model = train_and_eval(data_dicts, model, num_steps=1000, save_path_prefix=None, train_bsize=args.train_batch_size, accumulate=args.gradient_accumulation_steps, save_every=None, eval_every=None, warmup_steps=5000, train_last_lyer_only=True)
        resulting_model = train_and_eval(data_dicts, model, save_path_prefix=save_path_prefix, num_steps=args.max_steps, train_bsize=args.train_batch_size, eval_bsize=args.eval_batch_size, accumulate=args.gradient_accumulation_steps, save_every=args.save_steps, eval_every=args.eval_steps, warmup_steps=args.warmup_steps, save_best_metric=args.save_best_metric)

    print('Evaluating...', 'scores only? ', args.scores_only)
    preds = model.evaluate_dicts(data_dicts['eval'], args.eval_batch_size, scores_only=args.scores_only)

    if args.pred_save_path is not None:
        pred_save_path = args.pred_save_path
    else:
        pred_save_path = os.path.join(save_dir, 'preds.json')
    json.dump(preds, open(pred_save_path, 'w'))
    if all(d['target'] is not None for d in data_dicts['eval']):
        print(evaluate_pred_path(pred_save_path))


# python3 clf.py --model_pretrain_name t5-small --model_init_path mount/models/t5-small_hardnews_iid_0/checkpoint-10000/model.pt --data hardnews_unlabeled_0 --pred_save_path mount/preds/t5smallsplit0step10000.json --no_train
# sbatch -p jsteinhardt -w balrog --gres=gpu:1 --export=model_name='t5-small',split=0,partition=9,step=10000 eval_hardnews.sh
# python3 clf.py --model_pretrain_name $model_name --model_init_path mount/models/$model_name\_hardnews_iid_$split/checkpoint-$step/model.pt --data hardnews_unlabeled_$partition --pred_save_path mount/preds/$model_name\_split$split\_step$step\_partition$partition.json --no_train
