#!/usr/bin/env python
# coding: utf-8 %%

# %%

import numpy as np
import pickle
import glob
import copy
import torch as t
import sys
import re
import torch
import numpy as np
import time
import argparse
from tqdm import tqdm
import torch.nn as nn
import os
import random
from exBert import BertTokenizer, BertAdam
from utils.Logger import Logger


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", required=True, type=int, help='number of training epochs')
    parser.add_argument("-b", "--batchsize", required=True, type=int, help='training batchsize')
    parser.add_argument("-sp", "--save_path", required=True, type=str, help='path to storaget the loss table, stat_dict')
    parser.add_argument('-dv', '--device', required=True, type=int, nargs='+',
                    help='gpu id for the training, ex [-dv 0 1 2 3]')
    parser.add_argument('-lr', '--learning_rate', required=True, type=float, help='learning rate , google use 1e-04')
    parser.add_argument('-str', '--strategy', required=True, type=str,
                    help='choose a strategy from [exBERT]')
    parser.add_argument('-config', '--config', required=True, type=str, nargs='+', help='dir to the config file')
    parser.add_argument('-vocab', '--vocab', required=True, type=str, help='path to the vocab file for tokenization')
    parser.add_argument('-pm_p', '--pretrained_model_path', default=None, type=str,
                    help='path to the pretrained_model stat_dict (torch state_dict)')
    parser.add_argument('-dp', '--data_path', required=True, type=str, help='path to data')
    parser.add_argument('-ls', '--longest_sentence', required=True, type=int,
                    help='set the limit of the sentence lenght, recommand the same to the -dt')
    parser.add_argument('-p', '--percentage', required=True, type=float, help='the percentage used for pretraining')
    parser.add_argument('-vp', '--val_percentage', required=True, type=float, default=0.1,
                        help='the percentage used for validation')
    parser.add_argument('-rd', '--random_seed', type=int, default=10,
                        help='random seed for the training')
    parser.add_argument('-mr', '--mask_rate', type=float, default=0.4, help='mask rate for the training')
    parser.add_argument('-rir', '--random_id_rate', type=float, default=0.1, help='random id rate for the training')
    parser.add_argument('-kr', '--keep_rate', type=float, default=0.05, help='keep rate for the training')

    parser.add_argument('-wp', '--warmup', default=-1, type=float,
                    help='portion of all training itters to warmup, -1 means not using warmup')
    parser.add_argument('-t_ex_only', '--train_extension_only', default=True, type=bool,
                    help='train only the extension module')

    parser.add_argument('-t_with_nlp_word', '--train_with_natural_language_word', action='store_true', default=False,
                        help='train with natural language word')
    args = vars(parser.parse_args())
    return args


class PreTrainPassBertTokenizer(BertTokenizer):
    def __init__(self, vocab_file, mask_rate, random_id_rate, keep_rate, use_nlp_word, **kwargs):
        '''

        '''
        super(PreTrainPassBertTokenizer, self).__init__(vocab_file, do_lower_case=False)
        self.use_nlp_word = use_nlp_word
        if use_nlp_word:
            self.mask_id = self.convert_tokens_to_ids(self.tokenize('[MASK]'))[0]
            self.sep_id = self.convert_tokens_to_ids(self.tokenize('[SEP]'))[0]
        else:
            self.mask_id = self.convert_tokens_to_ids(self.tokenize_without_nlp_subword('[MASK]'))[0]
            self.sep_id = self.convert_tokens_to_ids(self.tokenize_without_nlp_subword('[SEP]'))[0]
        self.mask_rate = mask_rate
        self.random_id_rate = random_id_rate
        self.keep_rate = keep_rate

    def Masking(self, Input_ids, Masked_lm_labels):
        copyInput_ids = copy.deepcopy(Input_ids)
        rd_1 = np.random.random(Input_ids.shape)
        rd_1[:, 0] = 0
        Masked_lm_labels[rd_1 > (1-self.mask_rate)] = Input_ids[rd_1 > (1-self.mask_rate)]
        Input_ids[rd_1 >= (1-self.mask_rate+self.random_id_rate)] = self.mask_id
        Input_ids[(rd_1 >= (1-self.mask_rate+self.keep_rate)) * (rd_1 < (1-self.mask_rate+self.random_id_rate))] = (
                    np.random.rand(((rd_1 >= (1-self.mask_rate+self.keep_rate)) * (rd_1 < (1-self.mask_rate+self.random_id_rate)) * 1).sum()) * len(self.vocab)).astype(int)
        Input_ids[copyInput_ids == 0] = 0
        Masked_lm_labels[copyInput_ids == 0] = -1
        return Input_ids, Masked_lm_labels

    def prepare_batch(self, Train_Data, batch_size=256, longest_sentence=128):
        Input_ids = np.zeros((batch_size, longest_sentence))
        Token_type_ids = np.zeros((batch_size, longest_sentence))
        Attention_mask = np.zeros((batch_size, longest_sentence))
        Masked_lm_labels = (np.ones((batch_size, longest_sentence)) * -1)
        for ii in range(batch_size):
            if not self.use_nlp_word:
                temp = self.convert_tokens_to_ids(self.tokenize_without_nlp_subword(Train_Data[ii]))
            else:
                temp = self.convert_tokens_to_ids(self.tokenize(Train_Data[ii]))
            if len(temp) > longest_sentence:
                sentence_length = longest_sentence
            else:
                sentence_length = len(temp)
            Input_ids[ii, 0:sentence_length] = temp[0:sentence_length]
            if self.sep_id in Input_ids[ii]:
                Token_type_ids[ii, np.where(Input_ids[ii] == self.sep_id)[0][0] + 1:sentence_length] = 1
            else:
                Token_type_ids[ii, :] = 0
            Attention_mask[ii, 0:sentence_length] = 1
        Input_ids, Masked_lm_labels = self.Masking(Input_ids, Masked_lm_labels)
        return Input_ids, Token_type_ids, Attention_mask, Masked_lm_labels


def load_data(data_path, random_seed, percentage, val_percent=0.1):
    with open (data_path, 'r') as f:
        data = f.readlines()

    data = ['[CLS] '+ i.strip() + ' [SEP]' for i in data]
    # for i in range(len(data)):
    #     data[i] = '[CLS] ' + data[i].strip() + ' [SEP]'
    print("add cls and sep done")

    random.seed(random_seed)
    random.shuffle(data)

    data = data[:int(len(data)*percentage)]
    train_data = data[:int(len(data)*(1-val_percent))]
    val_data = data[int(len(data)*(1-val_percent)):]
    return train_data, val_data


def process_batch(INPUT, is_train = True):
    if is_train:
        model.train()
        optimizer.zero_grad()
    Input_ids = t.tensor(INPUT[0]).long().to(device)
    Token_type_ids = t.tensor(INPUT[1]).long().to(device)
    Attention_mask = t.tensor(INPUT[2]).long().to(device)
    Masked_lm_labels = t.tensor(INPUT[3]).long().to(device)
    loss1 = model(Input_ids,
          token_type_ids = Token_type_ids,
          attention_mask = Attention_mask,
          masked_lm_labels = Masked_lm_labels
         )
    if is_train:
        loss1.sum().unsqueeze(0).backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss1.sum().data


if __name__ == '__main__':
    args = setup()
    dataset = args['data_path'].split('/')[-1].split('.')[0]
    vocab_name = args['vocab'].split('/')[-1].split('.')[0]
    sys.stdout = Logger(dataset=dataset, stream=sys.stdout,
                        running_operation='pass_pretrain_{}'.format(vocab_name))
    sys.stderr = Logger(dataset=dataset, stream=sys.stderr,
                        running_operation='pass_pretrain_{}'.format(vocab_name))

    for ii, item in enumerate(args):
        print(item + ': ' + str(args[item]))

    if args['device'] == [-1]:
        device = 'cpu'
        device_ids = 'cpu'
    else:
        device_ids = args['device']
        device = 'cuda:' + str(device_ids[0])
        print('training with GPU: ' + str(device_ids))

    tok = PreTrainPassBertTokenizer(args['vocab'], args['mask_rate'], args['random_id_rate'], args['keep_rate'],
                                    use_nlp_word=args['train_with_natural_language_word'])

    if args['strategy'] == 'exBERT':
        from exBERT import BertForPreTraining, BertConfig

        bert_config_1 = BertConfig.from_json_file(args['config'][0])
        bert_config_2 = BertConfig.from_json_file(args['config'][1])
        print("Building PyTorch model from configuration: {}".format(str(bert_config_1)))
        print("Building PyTorch model from configuration: {}".format(str(bert_config_2)))
        model = BertForPreTraining(bert_config_1, bert_config_2)
    else:
        from exBERT import BertForPreTraining, BertConfig

        bert_config_1 = BertConfig.from_json_file(args['config'][0])
        print("Building PyTorch model from configuration: {}".format(str(bert_config_1)))
        model = BertForPreTraining(bert_config_1)

    ## load pre-trained model
    if args['pretrained_model_path'] is not None:
        stat_dict = t.load(args['pretrained_model_path'], map_location='cpu')
        model.load_state_dict(stat_dict, strict=False)

    sta_name_pos = 0
    if device != 'cpu':
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
            sta_name_pos = 1
        model.to(device)

    if args['strategy'] == 'exBERT':
        if args['train_extension_only']:
            for ii, item in enumerate(model.named_parameters()):
                item[1].requires_grad = False
                if 'ADD' in item[0]:
                    item[1].requires_grad = True
                if 'pool' in item[0]:
                    item[1].requires_grad = True
                if item[0].split('.')[sta_name_pos] != 'bert':
                    item[1].requires_grad = True
    print('The following part of model is goinig to be trained:')
    for ii, item in enumerate(model.named_parameters()):
        if item[1].requires_grad:
            print(item[0])

    lr = args['learning_rate']
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    train_data, val_data = load_data(args['data_path'], args['random_seed'], args['percentage'], args['val_percentage'])

    print('done data preparation')
    print('train data len: {}\t val data len: {}'.format(len(train_data), len(val_data)))

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
        print("create folder: " + args['save_path'])

    num_epoc = args['epochs']
    batch_size = args['batchsize']
    longest_sentence = args['longest_sentence']
    total_batch_num = int(np.ceil(len(train_data) / batch_size))
    total_val_batch_num = int(np.ceil(len(val_data) / batch_size))

    train_loss_table = np.zeros((num_epoc, total_batch_num))
    val_loss_table = np.zeros((num_epoc, total_val_batch_num))

    optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=args['warmup'], t_total=total_batch_num*num_epoc)
    best_loss = float('inf')

    save_id = 0
    print_every_ndata = int(len(train_data) / batch_size / 100)

    print("start training ...")
    try:
        for epoc in range(num_epoc):
            t2 = time.time()
            train_loss = 0
            val_loss = 0

            for batch_ind in range(total_batch_num):
                end_id = min(len(train_data), batch_size * (batch_ind + 1))
                new_batch_size = min(batch_size, end_id - batch_size * batch_ind)

                input = tok.prepare_batch(Train_Data=train_data[batch_ind * batch_size: end_id],
                                          longest_sentence=longest_sentence,  batch_size=new_batch_size)

                train_log = process_batch(input, is_train=True)
                train_loss_table[epoc, batch_ind] = train_log
                train_loss += train_log

                if batch_ind > 0 and batch_ind % print_every_ndata == 0:
                    print(
                        'Train Epoch: {} [{}/{} ({:.0f}%)]\taverage train_Loss: {:.5f} now batch average train_loss: '
                        '{:.5f} \t time: {:.4f} \t lr:{:.6f}'.format(
                            epoc,
                            batch_ind * batch_size , total_batch_num * batch_size,
                            100 * batch_ind * batch_size / len(train_data),
                            train_loss / end_id,
                            train_log / new_batch_size,
                            time.time() - t2,
                            optimizer.get_lr()[0]))

                with open(args['save_path'] + '/loss.pkl', 'wb') as f:
                    pickle.dump([train_loss_table, val_loss_table, args], f)
            if len(device_ids) > 1:
                t.save(model.module.state_dict(), args['save_path'] + '/state_dic_' + args['strategy'] + '_' + str(epoc))
            else:
                t.save(model.state_dict(), args['save_path'] + '/state_dic_' + args['strategy'] + '_' + str(epoc))
            with open(args['save_path'] + '/loss.pkl', 'wb') as f:
                pickle.dump([train_loss_table, val_loss_table, args], f)

            model.eval()
            with t.no_grad():
                for batch_ind in range(total_val_batch_num):
                    end_id = min(len(val_data), batch_size * (batch_ind + 1))
                    new_batch_size = min(batch_size, end_id - batch_size * batch_ind)

                    input = tok.prepare_batch(Train_Data=val_data[batch_ind * batch_size: end_id],
                                                longest_sentence=longest_sentence, batch_size=new_batch_size)

                    val_log = process_batch(input, is_train=False)
                    val_loss_table[epoc, batch_ind] = val_log
                    val_loss += val_log
            with open(args['save_path'] + '/loss.pkl', 'wb') as f:
                pickle.dump([train_loss_table, val_loss_table, args], f)
            print('val_loss: ' + str(val_loss / len(val_data)))

            if val_loss < best_loss:
                if len(device_ids) > 1:
                    t.save(model.module.state_dict(), args['save_path'] + '/Best_stat_dic_' + args['strategy'])
                else:
                    t.save(model.state_dict(), args['save_path'] + '/Best_stat_dic_' + args['strategy'])
                best_loss = val_loss
                print('update!!!!!!!!!!!!')
    except KeyboardInterrupt:
        print('saving stat_dict and loss table')
        with open(args['save_path'] + '/kbstop_loss.pkl', 'wb') as f:
            pickle.dump([train_loss_table, val_loss_table, args], f)
        t.save(model.state_dict(), args['save_path'] + '/kbstop_stat_dict')