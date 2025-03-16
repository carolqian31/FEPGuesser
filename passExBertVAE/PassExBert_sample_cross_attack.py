#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import argparse
import os
import sys
import torch
import random
import time
import inspect
import pickle
import numpy as np
import torch.nn as nn

from utils.passExBert_data_loader import PassExBertDataProviderCrossAttack
from model.passexbert_rvae import PassExBertRVAE
from utils.parameters import PassExBertParameters
from utils.Logger import Logger
# from utils.sample_method import points_sampling_gpu


def setup():
    parser = argparse.ArgumentParser(description='PassExBert_sample_cross_attack')
    parser.add_argument('--origin_data', type=str, default='csdn', help='the data file name')
    parser.add_argument('--use_cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--num_sample', type=int, default=1e7, metavar='NS',
                        help='num samplings (default: 10)')
    parser.add_argument('--batch_size', type=int, default=256, help="the batch size of origin password points")
    parser.add_argument('--vertex_num', type=int, default=None,
                        help='the number of vertex we sample around the origin point')

    parser.add_argument('--num_workers', type=int, default=0, help='the number of workers when loading data')
    parser.add_argument('--vae_path', type=str, default=None, help='the folder of the experiment')
    parser.add_argument('--seed', type=int, default=10, help='the random seed')
    parser.add_argument('--guess_folder', type=str, default='./guess_passExBert', help='the folder of the guess')
    # parser.add_argument('--sample_batch_size', type=int, default=256, help='the batch size of sampling')
    parser.add_argument('--max_seq_len', type=int, default=34, help='the max length of password')
    parser.add_argument('--bert_dict_path', default=None, type=str, help="the bert dict path")
    parser.add_argument('--config', default=None, type=str, nargs='+', help="bert config file path")
    parser.add_argument('--passExBert_embed_type', default=None, type=str,
                        choices=['pooled_output', 'last_layer', 'last_four_layers', 'sixth_layer', 'second_layer',
                                 'first_and_last_layer'],
                        help="passExBert embedding type")
    parser.add_argument("--device", required=True, type=int, nargs="+",
                        help="GPU device ids to use e.g. --device 0 1. If input -1, use CPU.")
    parser.add_argument('--vocab', default=None, type=str, help="the vocab file path")

    parser.add_argument('--step_size', type=float, default=None,
                        help='the length one step can reach when using uniform distribution to sample')
    parser.add_argument('--max_hop', type=int, default=None,
                        help='the max hop when using uniform distribution to sample')
    parser.add_argument('--sigma', type=float, default=1,
                        help='the parameter of covariance when using gaussian distribution to sample')

    parser.add_argument('--beam_size', type=int, default=None, help='the beam num when using beam search')
    parser.add_argument('--temperature', type=float, default=None, help='the temperature when using beam search')
    parser.add_argument('--test_data', type=str, default=None, help='test data path(default is origin test path)')

    parser.add_argument('--init_password_num', type=int, default=None,
                        help='the init password num of original training password')
    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_file_saved_path(args):
    folder_path = os.path.join(args.guess_folder, "{}_attack_{}".format(args.origin_data, args.test_data))
    if not os.path.exists(folder_path):
        print('create {}'.format(folder_path))
        os.makedirs(folder_path)

    folder_path = os.path.join(folder_path, args.vae_path.split('/')[-2])
    if not os.path.exists(folder_path):
        print('create {}'.format(folder_path))
        os.makedirs(folder_path)

    file_name = ""
    file_name += "gaussian_sigma_" + str(args.sigma) + "_" + str(args.num_sample)
    file_name += "_step_size_" + str(args.step_size)
    file_name += "_beam_size_" + str(args.beam_size) + "_temperature_" + str(args.temperature)
    file_name += "_init_password_num_{}".format(args.init_password_num)

    temp_attack_dict_save_path = os.path.join(folder_path, file_name + ".temp_attack_pass_dict.pkl")
    temp_attack_success_path = os.path.join(folder_path, file_name + ".temp_attack_success.pkl")
    temp_samplling_idx_path = os.path.join(folder_path, file_name + ".temp_sampling_idx.pkl")
    args_file_save_path = os.path.join(folder_path, file_name + ".args.txt")
    first_round_stop_pass_idx_path = os.path.join(folder_path, file_name + ".first_round_stop_pass_idx.txt")
    now_round_seed_data_path = os.path.join(folder_path, file_name + ".now_round_seed_data.txt")
    return os.path.join(folder_path, file_name + ".txt"), temp_attack_dict_save_path, temp_attack_success_path, \
           temp_samplling_idx_path, args_file_save_path, first_round_stop_pass_idx_path, now_round_seed_data_path


def generate_now_guess_rate(guess_dict, test_dict, total_test_dict_len):
    now_guess_num = 0
    for password in guess_dict:
        if password in test_dict:
            now_guess_num += test_dict[password]

    return now_guess_num / total_test_dict_len


def get_password_train_path(dataset):
    file_path = '/home/qianqiuyan/FEPGuesser/passExBertVAE/data/{}_bert/train.txt'.format(dataset)
    return file_path


def get_password_test_path(dataset):
    file_path = '/home/qianqiuyan/FEPGuesser/passExBertVAE/data/{}_bert/test.txt'.format(dataset)
    return file_path


def get_password_list(dataset):
    file_path = get_password_train_path(dataset)
    with open(file_path, 'r') as f:
        passwords = f.readlines()

    passwords = [password.strip() for password in passwords]
    return passwords


def get_top_freq_password_list(args):
    dataset = args.origin_data
    passwords = get_password_list(dataset)
    password_dict = {}
    for password in passwords:
        if password not in password_dict:
            password_dict[password] = 1
        else:
            password_dict[password] += 1

    sorted_password_list = sorted(password_dict, key=password_dict.get, reverse=True)

    init_password_num = args.init_password_num

    print("init password num is: {}".format(init_password_num))

    seed_password_list = sorted_password_list[:init_password_num]
    print("origin dataset {}, select top frequency {} passwords, "
          "top password {} frequency is {}, last password {} frequency is {}".format(
        args.origin_data, init_password_num, seed_password_list[0],
        password_dict[seed_password_list[0]], seed_password_list[-1],
        password_dict[seed_password_list[-1]]
    ))

    return seed_password_list


if __name__ == '__main__':
    args = setup()
    running_name = "sample_vae_passExBert_cross_attack"
    sys.stdout = Logger(dataset=args.origin_data+"_attack_"+args.test_data, stream=sys.stdout, running_operation=running_name)
    sys.stderr = Logger(dataset=args.origin_data+"_attack_"+args.test_data, stream=sys.stderr, running_operation=running_name)
    if not os.path.exists(args.guess_folder):
        os.makedirs(args.guess_folder)
        print("create folder {}".format(args.guess_folder))

    set_seed(args.seed)

    vae_path = args.vae_path

    if args.device == -1:
        device = 'cpu'
        device_ids = 'cpu'
    else:
        device_ids = args.device
        device = 'cuda:' + str(args.device[0])
    print(device_ids)


    seed_passwords_list = get_top_freq_password_list(args)
    test_data_path = get_password_test_path(args.test_data)

    data_provider = PassExBertDataProviderCrossAttack(seed_data_list=seed_passwords_list, batch_size=args.batch_size,
                                                      test_data_path=test_data_path, vocab_file_path=args.vocab,
                                                      num_workers=args.num_workers, max_len=args.max_seq_len)
    pad_token_id = data_provider.get_padding_idx()
    test_dict = data_provider.test_data_dict
    total_test_dict_len = data_provider.total_num


    parameters = PassExBertParameters()

    config1 = None

    if len(args.config) == 2:
        config1 = args.config[1]
    passExBertRVAE = PassExBertRVAE(parameters, bert_dict_path=args.bert_dict_path,
                                    passExBert_config_path=args.config[0], passExBert_config1_path=config1,
                                    passExBert_embed_type=args.passExBert_embed_type, device_ids=device_ids,
                                    pad_token_idx=pad_token_id)

    if device != 'cpu':
        if len(device_ids) > 1:
            passExBertRVAE = nn.DataParallel(passExBertRVAE, device_ids=device_ids)
        passExBertRVAE.to(device)

    if len(device_ids) > 1:
        print(vae_path)
        passExBertRVAE.module.load_state_dict(torch.load(vae_path, map_location='cpu'), strict=False)
    else:
        passExBertRVAE.load_state_dict(torch.load(vae_path, map_location='cpu'), strict=False)

    passExBertRVAE.eval()

    guess_dict = {}

    saved_file_path, temp_pass_dict_path, temp_attack_success_path, temp_sampling_idx_path, args_file_path, \
    first_round_stop_pass_idx_path, now_round_seed_data_path = get_file_saved_path(args)
    print("file will be saved to {}".format(saved_file_path))

    with open(args_file_path, 'w') as f:
        f.write(str(args))
        print("args message saved to {}".format(args_file_path))


    if len(device_ids) > 1:
        passExBert_rvae_sampler = passExBertRVAE.module.sampler(data_provider,
                                                                origin_point_strategy='dynamic_beam',
                                                                vertex_num=args.vertex_num, beam_size=args.beam_size,
                                                                    temperature=args.temperature)
    else:
        passExBert_rvae_sampler = passExBertRVAE.sampler(data_provider,
                                                         origin_point_strategy='dynamic_beam',
                                                         vertex_num=args.vertex_num, beam_size=args.beam_size,
                                                         temperature=args.temperature)

    iteration = 0
    reach_max_guess = False

    t_start = time.time()

    dynamic_attacked_list = []
    epoch_time = 0

    sigma = args.sigma
    first_reach_end = True
    now_round_iter = 0
    try:
        while True:
            try:
                generate_passwords_list, new_attacked_list, _, _ = passExBert_rvae_sampler(mode='greedy',
                                                                                           sample_strategy='gaussian',
                                                                                           sigma=sigma, step_size=args.step_size,
                                                                                           max_hop=None)
                dynamic_attacked_list.extend(new_attacked_list)

                for password in generate_passwords_list:
                    if password not in guess_dict:
                        guess_dict[password] = 1
                    else:
                        guess_dict[password] += 1

                if iteration % 100 == 0:
                    print("iteration {}, time {}s, generate {} passwords, {}% of guess task".
                          format(iteration, time.time() - t_start, len(guess_dict),
                                 len(guess_dict) / args.num_sample * 100))

                if iteration % 1000 == 0:
                    now_attacked_rate = generate_now_guess_rate(guess_dict=guess_dict, test_dict=test_dict,
                                                                total_test_dict_len=total_test_dict_len)

                    print("now attacked {}% of guess task".format(now_attacked_rate * 100))

                    with open(temp_pass_dict_path, 'wb') as f:
                        pickle.dump(guess_dict, f)
                        print("save temp file {}".format(temp_pass_dict_path))

                    if len(dynamic_attacked_list) > 0:
                        with open(temp_attack_success_path, 'wb') as f:
                            pickle.dump(dynamic_attacked_list, f)
                            print("save total temp success attack file {}".format(temp_attack_success_path))

                    now_round_data = data_provider.dataset.data_list

                    with open(now_round_seed_data_path, 'wb') as f:
                        pickle.dump(now_round_data, f)
                        print("save now round seed file {}".format(temp_attack_success_path))

                    idx_save_dict = {}
                    sample_idx = now_round_iter * args.batch_size
                    idx_save_dict['sample_idx'] = sample_idx
                    idx_save_dict['sigma'] = sigma
                    idx_save_dict['step_size'] = args.step_size
                    idx_save_dict['epoch_time'] = epoch_time
                    idx_save_dict['now_round_iter'] = now_round_iter

                    with open(temp_sampling_idx_path, 'wb') as f:
                        pickle.dump(idx_save_dict, f)
                        print("save temp sampling idx file {}".format(temp_sampling_idx_path))

                if len(guess_dict) > args.num_sample:
                    print("generate {} passwords, break".format(len(guess_dict)))
                    break
                iteration += 1
                now_round_iter += 1

            except StopIteration:
                print("reach the end of the dataset")
                now_round_iter = 0

                if epoch_time == 0:
                    with open(first_round_stop_pass_idx_path, 'w') as f:
                        f.write(str(len(guess_dict)))
                    print("generate {} passwords, save first round stop idx file {}"
                          .format(len(guess_dict), first_round_stop_pass_idx_path))
                else:
                    sigma += args.step_size
                    print("increase sigma to {}".format(sigma))
                #
                # if sigma == 0.1:
                #     break
                temp_attack_success_path_sigma = temp_attack_success_path.replace(".pkl", "") + "_{}.pkl".format(sigma)
                with open(temp_attack_success_path_sigma, 'wb') as f:
                    pickle.dump(dynamic_attacked_list, f)
                    print("save temp file {}".format(temp_attack_success_path))

                print("attacked {} passwords successfully in total".format(len(dynamic_attacked_list)))
                print("guess {} passwords in all".format(len(guess_dict)))

                new_round_attack_list = data_provider.updata_dataset()
                increase_attacked_rate = generate_now_guess_rate(guess_dict=new_round_attack_list, test_dict=test_dict,
                                                            total_test_dict_len=total_test_dict_len)
                print("round {}, attack {} new passwords, increasing {}% attacking rate!"
                      .format(epoch_time, len(new_round_attack_list), increase_attacked_rate))

                if len(device_ids) > 1:
                    passExBert_rvae_sampler = passExBertRVAE.module.sampler(data_provider,
                                                                            origin_point_strategy='dynamic_beam',
                                                                            vertex_num=args.vertex_num,
                                                                            beam_size=args.beam_size,
                                                                            temperature=args.temperature)
                else:
                    passExBert_rvae_sampler = passExBertRVAE.sampler(data_provider,
                                                                     origin_point_strategy='dynamic_beam',
                                                                     vertex_num=args.vertex_num,
                                                                     beam_size=args.beam_size,
                                                                     temperature=args.temperature)

                epoch_time += 1


        print("done!")
        print(time.time() - t_start)
        print("generate {} passwords".format(len(guess_dict)))

        now_attacked_rate = generate_now_guess_rate(guess_dict=guess_dict, test_dict=test_dict,
                                                    total_test_dict_len=total_test_dict_len)
        print("now attacked {}% of guess task".format(now_attacked_rate * 100))

        with open(saved_file_path, "w", encoding="UTF-8") as f:
            for passwd in guess_dict:
                f.write(passwd + "\n")
        print("file has been saved to {}".format(saved_file_path))

        success_pwd_file = saved_file_path.split('.txt')[0] + "_success_attack_passwords.txt"
        if not os.path.exists(os.path.dirname(success_pwd_file)):
            os.makedirs(os.path.dirname(success_pwd_file))
        print("generate {} success attack passwords".format(len(dynamic_attacked_list)))
        with open(success_pwd_file, "w", encoding="UTF-8") as f:
            for passwd in dynamic_attacked_list:
                f.write(passwd + "\n")

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        print("generate {} passwords".format(len(guess_dict)))

        with open(saved_file_path, "w", encoding="UTF-8") as f:
            for passwd in guess_dict:
                f.write(passwd + "\n")
        print("file has been saved to {}".format(saved_file_path))
