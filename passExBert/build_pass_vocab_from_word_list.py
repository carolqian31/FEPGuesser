import os
import time
import argparse
import collections

import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split

from exBert import BertTokenizer
from exBert.tokenization import load_vocab


class VocabProcessor:
    def __init__(self, dataset='csdn', origin_vocab_path=None, word_list_path=None, train_data_path=None):
        self.dataset = dataset
        self.word_list_set = set()
        self.word_list_path = word_list_path
        self.max_rule_length = 0

        self.get_word_list_set()

        self.origin_vocab_path = origin_vocab_path

        self.origin_vocab = load_vocab(self.origin_vocab_path)

        self.origin_vocab_len = len(self.origin_vocab)

        self.train_data_path = train_data_path
        self.clean_train_data_list = self.get_clean_data()

        print("vocabProcessor init finished")


    # def get_raw_dataset_txt_path(self):
    #     return os.path.join('/home/passwordfile/source-file', self.dataset+'.txt')
        # return os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data', self.dataset+'.txt')

    def get_clean_data(self):
        if not os.path.exists(self.train_data_path):
            raise FileNotFoundError("[preporcess] source data {} not exist.".format(self.train_data_path))
        data_list = []
        with open(self.train_data_path, encoding="UTF-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if len(line) < 32 and line.isascii():
                    line = line.strip()
                    if ' ' in line:
                        continue
                    data_list.append(line)
        return data_list


    def get_word_list_set(self):
        with open(self.word_list_path, encoding='UTF-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                self.word_list_set.add(line)
                self.max_rule_length = max(self.max_rule_length, len(line))

    def generate_pass_vocab(self, data_train, output_vocab_path):
        count = 0
        for password in data_train:
            self.generate_one_password_vocab(password)
            count += 1
            if count % 10000 == 0:
                print("processed {} passwords".format(count))
        print("done!")
        print("the origin vocab size is {}, the final vocab size is {}".format(self.origin_vocab_len,
                                                                               len(self.origin_vocab)))

        if not os.path.exists(os.path.dirname(output_vocab_path)):
            os.makedirs(os.path.dirname(output_vocab_path))
            print("create directory {}".format(os.path.dirname(output_vocab_path)))

        with open(output_vocab_path, 'w') as f:
            for key in self.origin_vocab:
                f.write(key+'\n')
        print("write vocab to {}".format(output_vocab_path))

    def generate_one_password_vocab(self, password):
        password = password.strip()
        password_len = len(password)
        is_start = True
        index = self.origin_vocab_len
        while password_len > 0:
            max_cut_length = min(password_len, self.max_rule_length)
            sub_sentence = password[0:max_cut_length]
            while max_cut_length > 0:
                if is_start:
                    if sub_sentence in self.origin_vocab:
                        is_start = False
                        password = password[max_cut_length:]
                        password_len -= max_cut_length
                        break
                    elif sub_sentence in self.word_list_set:
                        self.origin_vocab[sub_sentence] = index
                        index += 1
                        is_start = False
                        password = password[max_cut_length:]
                        password_len -= max_cut_length
                        break
                    else:
                        max_cut_length -= 1
                        sub_sentence = password[0:max_cut_length]
                else:
                    middle_sub_sentence = '##'+sub_sentence
                    if middle_sub_sentence in self.origin_vocab:
                        password = password[max_cut_length:]
                        password_len -= max_cut_length
                        break
                    elif sub_sentence in self.word_list_set:
                        self.origin_vocab[middle_sub_sentence] = index
                        index += 1
                        password = password[max_cut_length:]
                        password_len -= max_cut_length
                        break
                    else:
                        max_cut_length -= 1
                        sub_sentence = password[0:max_cut_length]


if __name__=='__main__':
    dataset = '000webhost'
    word_list_path = '../PSVG/Rules/000webhost_segment/wordlist_segment_200.txt'
    origin_vocab_path = 'bert_base_cased/vocab.txt'
    output_vocab_path = 'config_and_vocab/000webhost/wordlist_segment_200.txt'
    #
    #
    train_data_path = './data/000webhost_bert/train.txt'
    vocabProcessor = VocabProcessor(dataset=dataset, origin_vocab_path=origin_vocab_path, word_list_path=word_list_path,
                                    train_data_path=train_data_path)

    vocabProcessor.generate_pass_vocab(vocabProcessor.clean_train_data_list, output_vocab_path)