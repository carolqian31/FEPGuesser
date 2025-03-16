import os
import torch
import pickle
import random
import collections
import numpy as np

from abc import ABCMeta, abstractmethod
from six.moves import cPickle
from torch.utils.data import DataLoader, Dataset, SequentialSampler, BatchSampler, IterableDataset

from utils.functional import fold, f_and
# from utils.sample_method import get_density_rank
from utils.frequency_method import get_data_dict, get_frequency_data_list, turn_data_list_to_dict
from bert_model.tokenization import BertTokenizer


class PassExBertDataset(Dataset):
    def __init__(self, data_list, bert_tokenizer, max_len=34):
        self.data_list = data_list
        self.bert_tokenizer = bert_tokenizer
        self.max_len = max_len
        self.data_len = len(self.data_list)
        self.cls_idx = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[CLS]'))[0]
        self.sep_idx = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[SEP]'))[0]
        self.pad_idx = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[PAD]'))[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data = self.data_list[idx]
        data = self.bert_tokenizer.tokenize(data)

        end_idx = self.max_len-2 if len(data) > self.max_len-2 else len(data)
        length = self.max_len if len(data) > self.max_len else len(data)+2
        data = data[:end_idx]
        data = self.bert_tokenizer.convert_tokens_to_ids(data)
        data = torch.tensor(data)

        input_ids = torch.ones(self.max_len, dtype=torch.long) * self.pad_idx
        attention_mask = torch.zeros(self.max_len, dtype=torch.long)
        input_ids[1:1+end_idx] = data
        input_ids[0] = self.cls_idx
        input_ids[end_idx+1] = self.sep_idx
        attention_mask[:end_idx+2] = 1

        return input_ids, attention_mask, length      # length-1: remove [CLS]


class PassExBertDynamicDatasetNewCross(Dataset):
    def __init__(self, data_list, bert_tokenizer, max_len=34):
        self.data_list = data_list
        self.bert_tokenizer = bert_tokenizer
        self.max_len = max_len
        self.data_len = len(self.data_list)
        self.cls_idx = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[CLS]'))[0]
        self.sep_idx = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[SEP]'))[0]
        self.pad_idx = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[PAD]'))[0]
        self.new_data = []

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data = self.data_list[idx]
        data = self.bert_tokenizer.tokenize(data)

        end_idx = self.max_len-2 if len(data) > self.max_len-2 else len(data)
        length = self.max_len if len(data) > self.max_len else len(data)+2
        data = data[:end_idx]
        data = self.bert_tokenizer.convert_tokens_to_ids(data)
        data = torch.tensor(data)

        input_ids = torch.ones(self.max_len, dtype=torch.long) * self.pad_idx
        attention_mask = torch.zeros(self.max_len, dtype=torch.long)
        input_ids[1:1+end_idx] = data
        input_ids[0] = self.cls_idx
        input_ids[end_idx+1] = self.sep_idx
        attention_mask[:end_idx+2] = 1

        return input_ids, attention_mask, length      # length-1: remove [CLS]

    def update_dataset(self):
        self.data_list.extend(self.new_data)
        self.data_len = len(self.data_list)
        now_round_new_attacked_data = self.new_data
        self.new_data = []
        return now_round_new_attacked_data

    def add_new_data(self, new_data):
        self.new_data.extend(new_data)


class PassExBertDynamicDataset(IterableDataset):
    def __init__(self, data_list, bert_tokenizer, max_len=34, restart_init_idx=None):
        self.data_list = data_list
        self.bert_tokenizer = bert_tokenizer
        self.max_len = max_len
        self.cls_idx = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[CLS]'))[0]
        self.sep_idx = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[SEP]'))[0]
        self.pad_idx = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[PAD]'))[0]
        self.additional_data = []
        self.restart_init_idx = restart_init_idx

    def add_data(self, data):
        self.additional_data.extend(data)

    def process_data(self, data):
        data = self.bert_tokenizer.tokenize(data)
        end_idx = self.max_len - 2 if len(data) > self.max_len - 2 else len(data)
        length = self.max_len if len(data) > self.max_len else len(data) + 2
        data = data[:end_idx]
        data = self.bert_tokenizer.convert_tokens_to_ids(data)
        data = torch.tensor(data)
        input_ids = torch.ones(self.max_len, dtype=torch.long) * self.pad_idx
        attention_mask = torch.zeros(self.max_len, dtype=torch.long)
        input_ids[1:1 + end_idx] = data
        input_ids[0] = self.cls_idx
        input_ids[end_idx + 1] = self.sep_idx
        attention_mask[:end_idx + 2] = 1
        return input_ids, attention_mask, length

    def data_generator(self):
        if self.restart_init_idx is None:
            for idx in range(len(self.data_list)):
                data = self.data_list[idx]
                input_ids, attention_mask, length = self.process_data(data)
                yield input_ids, attention_mask, length
        else:
            for idx in range(self.restart_init_idx, len(self.data_list)):
                data = self.data_list[idx]
                input_ids, attention_mask, length = self.process_data(data)
                yield input_ids, attention_mask, length

        while self.additional_data:
            data = self.additional_data.pop()
            input_ids, attention_mask, length = self.process_data(data)
            yield input_ids, attention_mask, length

    def __iter__(self):
        return self.data_generator()


class PassExBertDatasetWithFrequency(Dataset):
    def __init__(self, data_list, bert_tokenizer, frequency_value_list, max_len=34):
        self.data_list = data_list
        self.bert_tokenizer = bert_tokenizer
        self.max_len = max_len
        self.frequency_value_list = frequency_value_list
        self.data_len = len(self.data_list)
        self.cls_idx = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[CLS]'))[0]
        self.sep_idx = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[SEP]'))[0]
        self.pad_idx = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[PAD]'))[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data = self.data_list[idx]
        data = self.bert_tokenizer.tokenize(data)

        end_idx = self.max_len-2 if len(data) > self.max_len-2 else len(data)
        length = self.max_len if len(data) > self.max_len else len(data)+2
        data = data[:end_idx]
        data = self.bert_tokenizer.convert_tokens_to_ids(data)
        data = torch.tensor(data)

        input_ids = torch.ones(self.max_len, dtype=torch.long) * self.pad_idx
        attention_mask = torch.zeros(self.max_len, dtype=torch.long)
        input_ids[1:1+end_idx] = data
        input_ids[0] = self.cls_idx
        input_ids[end_idx+1] = self.sep_idx
        attention_mask[:end_idx+2] = 1

        frequency_num = self.frequency_value_list[idx] + 10 if self.frequency_value_list[idx] < 10 else self.frequency_value_list[idx]
        if frequency_num % 2 != 0:
            frequency_num += 1
        return input_ids, attention_mask, length, frequency_num


class BatchSamplerTillEnd:
    def __init__(self, sampler, batch_size: int, total_num_samples=None) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        self.sampler = sampler
        self.batch_size = batch_size
        self.total_num_samples = total_num_samples

    def __iter__(self):
        if self.total_num_samples is None:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                if idx >= self.total_num_samples:
                    break
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class PassExBertDataProvider:
    def __init__(self, dataset, batch_size, vocab_file_path=None, num_workers=0, max_len=34, is_train=True,
                 valid_rate=0.1, use_frequency=False, is_sample=False, use_random=False, head_num=None, dynamic_dataset=False,
                 dynamic_init_list=None, restart_idx=None, test_data_path=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file_path = vocab_file_path
        self.max_len = max_len
        self.is_train = is_train
        self.is_sample = is_sample
        self.use_frequency = use_frequency
        self.use_random = use_random
        self.dynamic_dataset = dynamic_dataset
        self.dynamic_init_list = dynamic_init_list
        self.restart_idx = restart_idx

        if is_sample and ((use_frequency and use_random) or (not use_frequency and not use_random)) and not dynamic_dataset:
            raise ValueError('use_frequency and use_random should be different')

        data_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data')

        self.data_files = [os.path.join(data_path, dataset + '_bert', 'train.txt'),
                           os.path.join(data_path, dataset + '_bert', 'test.txt')]
        self.test_data_path = self.data_files[1]
        if test_data_path is not None:
            self.test_data_path = test_data_path
        print("test data path: {}".format(self.test_data_path))
        if dynamic_init_list is None:
            self.train_data_list = self.load_train_data()

        self.bert_tokenizer = BertTokenizer(vocab_file=self.vocab_file_path, do_lower_case=False)

        if not is_sample:
            self.train_data_list, self.valid_data_list = self.split_train_valid_data(valid_rate=valid_rate)
            self.train_data = PassExBertDataset(self.train_data_list, self.bert_tokenizer, self.max_len)
            self.train_data_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size,
                                                num_workers=self.num_workers, shuffle=True)

            self.valid_data = PassExBertDataset(self.valid_data_list, self.bert_tokenizer, self.max_len)
            self.valid_data_loader = DataLoader(dataset=self.valid_data, batch_size=self.batch_size,
                                                num_workers=self.num_workers, shuffle=True)

        if is_sample:
            if dynamic_dataset:
                # self.train_data_list = ['12345678', 'dearbook']
                self.dynamic_data = PassExBertDynamicDataset(self.dynamic_init_list, self.bert_tokenizer, self.max_len,
                                                             restart_init_idx=self.restart_idx)
                self.dynamic_data_loader = DataLoader(dataset=self.dynamic_data, batch_size=self.batch_size,
                                                        num_workers=self.num_workers)
            if use_frequency:
                self.frequency_data, self.frequency_value = self.get_frequency_data_trainint_password_list()
                self.frequency_data = PassExBertDatasetWithFrequency(self.frequency_data,
                                                                     self.bert_tokenizer,
                                                                     frequency_value_list=self.frequency_value,
                                                                     max_len=self.max_len)
                self.frequency_data_loader = DataLoader(batch_sampler=BatchSamplerTillEnd(
                    sampler=SequentialSampler(self.frequency_data), batch_size=self.batch_size,
                    total_num_samples=head_num), dataset=self.frequency_data, num_workers=self.num_workers)
            if use_random:
                # self.train_data_list = ['12345678', 'dearbook', 'qianqiuyan']
                self.random_data = PassExBertDataset(self.train_data_list, self.bert_tokenizer, self.max_len)
                self.random_data_loader = DataLoader(dataset=self.random_data, num_workers=self.num_workers,
                                                     batch_size=self.batch_size, shuffle=True)

            self.test_data_dict, self.attacked_dict, self.total_num = self.load_test_data_to_dict(test_data_path=self.test_data_path)

        self.sep_token_idx = self.get_sep_idx()

    def get_padding_idx(self):
        return self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[PAD]'))[0]

    def get_cls_idx(self):
        return self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[CLS]'))[0]

    def get_sep_idx(self):
        return self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[SEP]'))[0]

    def load_train_data(self):
        data = open(self.data_files[0], "r", encoding = "UTF-8").read().split('\n')
        return data

    def load_test_data_to_dict(self, test_data_path):
        test_dict = {}
        attacked_dict = {}
        total_line = 0
        with open(test_data_path, "r", encoding = "UTF-8") as f:
            while True:
                line = f.readline()
                total_line += 1
                if not line:
                    break
                line = line.strip()
                if line not in test_dict:
                    test_dict[line] = 1
                    attacked_dict[line] = 0
                else:
                    test_dict[line] += 1
        return test_dict, attacked_dict, total_line

    def get_frequency_data_trainint_password_list(self):
        data_dict = turn_data_list_to_dict(self.train_data_list)
        frequent_passwd_list, frequency_value_list = zip(*get_frequency_data_list(data_dict))
        frequent_passwd_list = list(frequent_passwd_list)
        frequency_value_list = list(frequency_value_list)
        return frequent_passwd_list, frequency_value_list

    def convert_ids_to_tokens(self, ids_list):
        token_list = []
        for ids in ids_list:
            token_list.append(self.bert_tokenizer.convert_ids_to_tokens(ids))
        return token_list

    def split_train_valid_data(self, valid_rate=0.1, random_seed=10):
        random.seed(random_seed)
        random.shuffle(self.train_data_list)
        valid_len = int(len(self.train_data_list) * valid_rate)
        valid_data_list = self.train_data_list[:valid_len]
        train_data_list = self.train_data_list[valid_len:]
        return train_data_list, valid_data_list

    def __iter__(self):
        if not self.is_sample:
            if self.is_train:
                return iter(self.train_data_loader)
            else:
                return iter(self.valid_data_loader)
        else:
            if self.use_frequency:
                return iter(self.frequency_data_loader)
            elif self.use_random:
                return iter(self.random_data_loader)
            elif self.dynamic_dataset:
                return iter(self.dynamic_data_loader)
            else:
                raise ValueError('Please choose mode use_frequency or use_random!')

    def __len__(self):
        if not self.is_sample:
            if self.is_train:
                return len(self.train_data)
            else:
                return len(self.valid_data)
        else:
            return self.total_num

    def convert_ids_to_passwords(self, ids_list):
        password_list = []
        for ids in ids_list:
            tokens = self.bert_tokenizer.convert_ids_to_tokens(ids)
            password = ''
            for token in tokens:
                if token == '[SEP]':
                    break
                if token == '[CLS]':
                    continue
                if token.startswith('##'):
                    token = token[2:]
                password += token
            password_list.append(password)
        return password_list

    def attack(self, password_list):
        attacked_num = 0
        success_attack_passwords = []
        for index, password in enumerate(password_list):
            if password in self.test_data_dict and self.attacked_dict[password] == 0:
                self.attacked_dict[password] = 1
                attacked_num += self.test_data_dict[password]
                success_attack_passwords.append(password)

        return attacked_num, attacked_num/self.total_num, success_attack_passwords

    def add_dynamic_data(self, password_list):
        self.dynamic_data.add_data(password_list)

    def get_success_attack_passwords(self):
        return self.dynamic_data.additional_data


class PassExBertDataProviderFromList:
    def __init__(self, origin_point_list, batch_size=1, vocab_file_path=None, num_workers=0, max_len=34):
        self.origin_point_list = origin_point_list
        self.batch_size = batch_size
        self.vocab_file_path = vocab_file_path
        self.num_workers = num_workers
        self.max_len = max_len

        self.bert_tokenizer = BertTokenizer(vocab_file=self.vocab_file_path, do_lower_case=False)

        self.dynamic_data = PassExBertDynamicDataset(self.origin_point_list, self.bert_tokenizer, self.max_len)
        self.dynamic_data_loader = DataLoader(dataset=self.dynamic_data, batch_size=self.batch_size,
                                              num_workers=self.num_workers)

    def __iter__(self):
        return iter(self.dynamic_data_loader)

    def get_padding_idx(self):
        return self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[PAD]'))[0]

    def get_cls_idx(self):
        return self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[CLS]'))[0]

    def get_sep_idx(self):
        return self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[SEP]'))[0]

    def convert_ids_to_passwords(self, ids_list):
        password_list = []
        for ids in ids_list:
            tokens = self.bert_tokenizer.convert_ids_to_tokens(ids)
            password = ''
            for token in tokens:
                if token == '[SEP]':
                    break
                if token == '[CLS]':
                    continue
                if token.startswith('##'):
                    token = token[2:]
                password += token
            password_list.append(password)
        return password_list


class PassExBertDataProviderFromOneFile:
    def __init__(self, data_file_path, batch_size=1, vocab_file_path=None, num_workers=0, max_len=34):
        self.data_file_path = data_file_path
        self.batch_size = batch_size
        self.vocab_file_path = vocab_file_path
        self.num_workers = num_workers
        self.max_len = max_len

        self.data_list = self.load_data(self.data_file_path)

        self.bert_tokenizer = BertTokenizer(vocab_file=self.vocab_file_path, do_lower_case=False)

        self.random_data = PassExBertDataset(self.data_file_path, self.bert_tokenizer, self.max_len)
        self.random_data_loader = DataLoader(dataset=self.random_data, batch_size=self.batch_size,
                                              num_workers=self.num_workers)

    def __iter__(self):
        return iter(self.random_data_loader)

    def load_data(self, data_path):
        data = open(data_path, "r", encoding="UTF-8").read().split('\n')
        return data

    def get_padding_idx(self):
        return self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[PAD]'))[0]

    def get_cls_idx(self):
        return self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[CLS]'))[0]

    def get_sep_idx(self):
        return self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[SEP]'))[0]

    def convert_ids_to_passwords(self, ids_list):
        password_list = []
        for ids in ids_list:
            tokens = self.bert_tokenizer.convert_ids_to_tokens(ids)
            password = ''
            for token in tokens:
                if token == '[SEP]':
                    break
                if token == '[CLS]':
                    continue
                if token.startswith('##'):
                    token = token[2:]
                password += token
            password_list.append(password)
        return password_list


class PassExBertDataProviderCrossAttack:
    def __init__(self, seed_data_list, batch_size, test_data_path, vocab_file_path=None, num_workers=0, max_len=34):
        self.seed_data_list = seed_data_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file_path = vocab_file_path
        self.max_len = max_len
        self.test_data_path = test_data_path

        print("test data path: {}".format(self.test_data_path))

        self.bert_tokenizer = BertTokenizer(vocab_file=self.vocab_file_path, do_lower_case=False)

        self.dataset = PassExBertDynamicDatasetNewCross(data_list=self.seed_data_list,
                                                        bert_tokenizer=self.bert_tokenizer,
                                                        max_len=self.max_len)

        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers, shuffle=False)

        self.test_data_dict, self.attacked_dict, self.total_num = self.load_test_data_to_dict(
            test_data_path=self.test_data_path)

        self.sep_token_idx = self.get_sep_idx()

    def get_padding_idx(self):
        return self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[PAD]'))[0]

    def get_cls_idx(self):
        return self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[CLS]'))[0]

    def get_sep_idx(self):
        return self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize('[SEP]'))[0]

    def load_test_data_to_dict(self, test_data_path):
        test_dict = {}
        attacked_dict = {}
        total_line = 0
        with open(test_data_path, "r", encoding = "UTF-8") as f:
            while True:
                line = f.readline()
                total_line += 1
                if not line:
                    break
                line = line.strip()
                if line not in test_dict:
                    test_dict[line] = 1
                    attacked_dict[line] = 0
                else:
                    test_dict[line] += 1
        return test_dict, attacked_dict, total_line

    def convert_ids_to_tokens(self, ids_list):
        token_list = []
        for ids in ids_list:
            token_list.append(self.bert_tokenizer.convert_ids_to_tokens(ids))
        return token_list

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataset)

    def convert_ids_to_passwords(self, ids_list):
        password_list = []
        for ids in ids_list:
            tokens = self.bert_tokenizer.convert_ids_to_tokens(ids)
            password = ''
            for token in tokens:
                if token == '[SEP]':
                    break
                if token == '[CLS]':
                    continue
                if token.startswith('##'):
                    token = token[2:]
                password += token
            password_list.append(password)
        return password_list

    def attack(self, password_list):
        """
        passwords in password_list that haven't attack successfully before
        """
        attacked_num = 0
        success_attack_passwords = []
        for index, password in enumerate(password_list):
            if password in self.test_data_dict and self.attacked_dict[password] == 0:
                self.attacked_dict[password] = 1
                attacked_num += self.test_data_dict[password]
                success_attack_passwords.append(password)

        return attacked_num, attacked_num/self.total_num, success_attack_passwords

    def add_dynamic_data(self, password_list):
        self.dataset.add_new_data(password_list)

    def updata_dataset(self):
        print("before seed dataset len: {}".format(len(self.dataset.data_list)))
        new_round_attacked_list = self.dataset.update_dataset()
        print("update seed dataset len: {}".format(len(self.dataset.data_list)))
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers, shuffle=False)
        return new_round_attacked_list

    # def get_all_success_attack_passwords(self):
    #     return self.