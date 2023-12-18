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

from utils.passExBert_data_loader import PassExBertDataProvider
from model.passexbert_rvae import PassExBertRVAE
from utils.parameters import PassExBertParameters
from utils.Logger import Logger
from utils.sample_method import points_sampling_gpu


def setup():
    parser = argparse.ArgumentParser(description='PassExBert_sample')
    parser.add_argument('--data', type=str, default='csdn', help='the data file name')
    parser.add_argument('--use_cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--num_sample', type=int, default=1e7, metavar='NS',
                        help='num samplings (default: 10)')
    parser.add_argument('--batch_size', type=int, default=256, help="the batch size of origin password points")
    parser.add_argument('--vertex_num', type=int, default=None,
                        help='the number of vertex we sample around the origin point')

    parser.add_argument('--strategy', type=str, default='gaussian', choices=['uniform', 'gaussian'],
                        help='the strategy we use to sample points around origin password points')
    parser.add_argument('--origin_points_strategy', type=str,
                        choices=['frequency_order', 'random_order', 'frequency_head_random', 'gaussian_step',
                                 'frequency_random_gaussian_step_mixed', 'beam_search_random', 'dynamic_beam_random'],
                        default='random_order', help='the strategy we use to choose origin password points')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of workers when loading data')
    parser.add_argument('--vae_path', type=str, default=None, help='the folder of the experiment')
    parser.add_argument('--seed', type=int, default=10, help='the random seed')
    parser.add_argument('--guess_folder', type=str, default='./guess_passExBert', help='the folder of the guess')
    parser.add_argument('--sample_batch_size', type=int, default=256, help='the batch size of sampling')
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
    parser.add_argument('--head_num', default=None, type=int, help="the head number of first part")
    parser.add_argument('--step_size', type=float, default=None,
                        help='the length one step can reach when using uniform distribution to sample')
    parser.add_argument('--max_hop', type=int, default=None,
                        help='the max hop when using uniform distribution to sample')
    parser.add_argument('--sigma', type=float, default=1,
                        help='the parameter of covariance when using gaussian distribution to sample')
    parser.add_argument('--vertex_strategy', type=str, default='fixed',
                        choices=['fixed', 'freqX'], help='the strategy we use to choose vertex')
    parser.add_argument('--beam_size', type=int, default=None, help='the beam num when using beam search')
    parser.add_argument('--temperature', type=float, default=None, help='the temperature when using beam search')
    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_file_saved_path(args):
    folder_path = os.path.join(args.guess_folder, args.data + '_bert')
    if not os.path.exists(folder_path):
        print('create {}'.format(folder_path))
        os.makedirs(folder_path)
    folder_path = os.path.join(folder_path, args.vae_path.split('/')[0])
    if not os.path.exists(folder_path):
        print('create {}'.format(folder_path))
        os.makedirs(folder_path)

    file_name = ""
    if args.strategy == "gaussian":
        file_name += "gaussian_sigma_" + str(args.sigma)
    elif args.strategy == "uniform":
        file_name += "uniform_step_size_" + str(args.step_size) + "_max_hop_" + str(args.max_hop)

    file_name += "_" + str(args.num_sample)

    file_name += "_" + args.origin_points_strategy
    # if "density" in args.origin_points_strategy:
    #     file_name += "_density_" + args.density_file_name.split('.pkl')[0]
    if "frequency_head" in args.origin_points_strategy:
        file_name += "_head_num_" + str(args.head_num)
    if "gaussian_step" in args.origin_points_strategy:
        file_name += "_step_size_" + str(args.step_size) + "_max_hop_" + str(args.max_hop)
    if "dynamic_beam" in args.origin_points_strategy:
        file_name += "_step_size_" + str(args.step_size)

    file_name += "_" + args.vertex_strategy
    if 'fixed' in args.vertex_strategy:
        file_name += "_vertex_num_" + str(args.vertex_num)
    if 'beam' in args.origin_points_strategy:
        file_name += "_beam_size_" + str(args.beam_size) + "_temperature_" + str(args.temperature)

    temp_attack_dict_save_path = os.path.join(folder_path, "temp_attack_pass_dict.pkl")
    temp_attack_success_path = os.path.join(folder_path, "temp_attack_success.pkl")
    temp_samplling_idx_path = os.path.join(folder_path, "temp_sampling_idx.pkl")
    return os.path.join(folder_path, file_name + ".txt"), temp_attack_dict_save_path, temp_attack_success_path, \
           temp_samplling_idx_path


def generate_now_guess_rate(guess_dict, test_dict, total_test_dict_len):
    now_guess_num = 0
    for password in guess_dict:
        if password in test_dict:
            now_guess_num += test_dict[password]

    return now_guess_num / total_test_dict_len


if __name__ == '__main__':
    args = setup()
    running_name = "sample_vae_passExBert"
    sys.stdout = Logger(dataset=args.data, stream=sys.stdout, running_operation=running_name)
    sys.stderr = Logger(dataset=args.data, stream=sys.stderr, running_operation=running_name)
    if not os.path.exists(args.guess_folder):
        os.makedirs(args.guess_folder)
        print("create folder {}".format(args.guess_folder))

    file_path = os.path.join(args.guess_folder, args.data + '_guess.txt')

    set_seed(args.seed)

    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    exbert_data_folder = os.path.join(data_folder, args.data + '_bert')
    vae_path = os.path.join(exbert_data_folder, args.vae_path)

    if args.device == -1:
        device = 'cpu'
        device_ids = 'cpu'
    else:
        device_ids = args.device
        device = 'cuda:' + str(args.device[0])
    print(device_ids)

    frequency_data_provider = None

    random_data_provider = None
    gaussian_step_data_provider = None
    dynamic_data_provider = None
    pad_token_id = None
    total_test_dict_len = 0
    test_dict=None

    if 'frequency' in args.origin_points_strategy:
        frequency_data_provider = PassExBertDataProvider(dataset=args.data, batch_size=args.batch_size,
                                               vocab_file_path=args.vocab, num_workers=args.num_workers,
                                               max_len=args.max_seq_len, use_frequency=True, use_random=False,
                                               is_sample=True, head_num=args.head_num)
        pad_token_id = frequency_data_provider.get_padding_idx()
        total_test_dict_len = frequency_data_provider.total_num
        test_dict = frequency_data_provider.test_data_dict

    if 'random' in args.origin_points_strategy:
        random_data_provider = PassExBertDataProvider(dataset=args.data, batch_size=args.batch_size,
                                                      vocab_file_path=args.vocab, num_workers=args.num_workers,
                                                      max_len=args.max_seq_len, use_frequency=False, use_random=True,
                                                      is_sample=True)
        pad_token_id = random_data_provider.get_padding_idx()
        total_test_dict_len = random_data_provider.total_num
        test_dict = random_data_provider.test_data_dict

    if 'gaussian_step' in args.origin_points_strategy:
        gaussian_step_data_provider = PassExBertDataProvider(dataset=args.data, batch_size=args.batch_size,
                                                      vocab_file_path=args.vocab, num_workers=args.num_workers,
                                                      max_len=args.max_seq_len, use_frequency=False, use_random=True,
                                                      is_sample=True)
        pad_token_id = gaussian_step_data_provider.get_padding_idx()
        total_test_dict_len = gaussian_step_data_provider.total_num
        test_dict = gaussian_step_data_provider.test_data_dict

    # if 'dynamic' in args.origin_points_strategy:
    #     dynamic_data_provider = PassExBertDataProvider(dataset=args.data, batch_size=args.batch_size,
    #                                                   vocab_file_path=args.vocab, num_workers=args.num_workers,
    #                                                   max_len=args.max_seq_len, use_frequency=False, use_random=False,
    #                                                   is_sample=True, dynamic_dataset=True)
    #     pad_token_id = dynamic_data_provider.get_padding_idx()

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
        passExBertRVAE.module.load_state_dict(torch.load(vae_path, map_location='cpu'), strict=False)
    else:
        passExBertRVAE.load_state_dict(torch.load(vae_path, map_location='cpu'), strict=False)


    passExBertRVAE.eval()

    passExBert_rvae_sampler_list = []
    guess_dict = {}

    saved_file_path, temp_pass_dict_path, temp_attack_success_path, temp_sampling_idx_path = get_file_saved_path(args)
    print("file will be saved to {}".format(saved_file_path))


    if frequency_data_provider is not None:
        if len(device_ids) > 1:
            passExBert_rvae_sampler = passExBertRVAE.module.sampler(frequency_data_provider,
                                                             origin_point_strategy='frequency')
        else:
            passExBert_rvae_sampler = passExBertRVAE.sampler(frequency_data_provider,
                                                             origin_point_strategy='frequency')
        passExBert_rvae_sampler_list.append(passExBert_rvae_sampler)

    if random_data_provider is not None and 'beam' in args.origin_points_strategy:
        if len(device_ids) > 1:
            if not 'dynamic' in args.origin_points_strategy:
                passExBert_rvae_sampler = passExBertRVAE.module.sampler(random_data_provider,
                                                                 origin_point_strategy='beam_search_random',
                                                                        vertex_num=args.vertex_num, beam_size=args.beam_size,
                                                                        temperature=args.temperature)
            else:
                passExBert_rvae_sampler = passExBertRVAE.module.sampler(random_data_provider,
                                                                        origin_point_strategy='beam_search_random',
                                                                        vertex_num=args.vertex_num,
                                                                        beam_size=args.beam_size,
                                                                        temperature=args.temperature, dynamic=True)
        else:
            if not 'dynamic' in args.origin_points_strategy:
                passExBert_rvae_sampler = passExBertRVAE.sampler(random_data_provider,
                                                                 origin_point_strategy='beam_search_random',
                                                                 vertex_num=args.vertex_num, beam_size=args.beam_size,
                                                                 temperature=args.temperature)
            else:
                passExBert_rvae_sampler = passExBertRVAE.sampler(random_data_provider,
                                                                 origin_point_strategy='beam_search_random',
                                                                 vertex_num=args.vertex_num,
                                                                 beam_size=args.beam_size,
                                                                 temperature=args.temperature, dynamic=True)

        passExBert_rvae_sampler_list.append(passExBert_rvae_sampler)

    elif random_data_provider is not None:
        if len(device_ids) > 1:
            passExBert_rvae_sampler = passExBertRVAE.module.sampler(random_data_provider,
                                                             origin_point_strategy='random',
                                                                    vertex_num=args.vertex_num)
        else:
            passExBert_rvae_sampler = passExBertRVAE.sampler(random_data_provider,
                                                             origin_point_strategy='random',
                                                             vertex_num=args.vertex_num)

        passExBert_rvae_sampler_list.append(passExBert_rvae_sampler)

    if gaussian_step_data_provider is not None:
        if len(device_ids) > 1:
            passExBert_rvae_sampler = passExBertRVAE.module.sampler(gaussian_step_data_provider,
                                                             origin_point_strategy='gaussian_step',
                                                                    vertex_num=args.vertex_num)
        else:
            passExBert_rvae_sampler = passExBertRVAE.sampler(gaussian_step_data_provider,
                                                             origin_point_strategy='gaussian_step',
                                                             vertex_num=args.vertex_num)
        passExBert_rvae_sampler_list.append(passExBert_rvae_sampler)

    # if dynamic_data_provider is not None:
    #     if len(device_ids) > 1:
    #         passExBert_rvae_sampler = passExBertRVAE.module.sampler(dynamic_data_provider,
    #                                                          origin_point_strategy='dynamic_beam',
    #                                                                 vertex_num=args.vertex_num, beam_size=args.beam_size,
    #                                                                 temperature=args.temperature)
    #     else:
    #         passExBert_rvae_sampler = passExBertRVAE.sampler(dynamic_data_provider,
    #                                                          origin_point_strategy='dynamic_beam',
    #                                                          vertex_num=args.vertex_num, beam_size=args.beam_size,
    #                                                          temperature=args.temperature)
    #     passExBert_rvae_sampler_list.append(passExBert_rvae_sampler)

    iteration = 0
    reach_max_guess = False

    t_start = time.time()

    dynamic_attacked_list = []
    first_init_dynamic = False
    insist_dynamic = True if 'dynamic' in args.origin_points_strategy else False
    if 'dynamic' in args.origin_points_strategy:
        first_init_dynamic = True
    first_reach_end = True
    epoch_time = 0

    sigma = args.sigma
    try:
        for passExBert_rvae_sampler in passExBert_rvae_sampler_list:
            while True:
                try:
                    if args.strategy == 'gaussian':
                        if first_init_dynamic or insist_dynamic:
                            generate_passwords_list, new_attacked_list, _, _ = passExBert_rvae_sampler(mode='greedy',
                                                                        sample_strategy=args.strategy,
                                                                          sigma=sigma, step_size=args.step_size,
                                                                          max_hop=args.max_hop)
                            dynamic_attacked_list.extend(new_attacked_list)
                        else:
                            generate_passwords_list = passExBert_rvae_sampler(mode='greedy', sample_strategy=args.strategy,
                                                                              sigma=sigma, step_size=args.step_size,
                                                                              max_hop=args.max_hop)
                    elif args.strategy == 'uniform':
                        generate_passwords_list = passExBert_rvae_sampler(mode='greedy', sample_strategy=args.strategy,
                                                                          step_size=args.step_size, max_hop=args.max_hop,
                                                                          sample_batch_size=args.sample_batch_size)
                    else:
                        raise ValueError("the sample strategy {} is not supported".format(args.strategy))
                    for password in generate_passwords_list:
                        if password not in guess_dict:
                            guess_dict[password] = 1
                            # print(password)
                        else:
                            guess_dict[password] += 1

                    if iteration % 100 == 0:
                        print("iteration {}, time {}s, generate {} passwords, {}% of guess task".
                              format(iteration, time.time()-t_start, len(guess_dict),
                                     len(guess_dict)/args.num_sample*100))

                    if iteration % 5000 == 0:

                        now_attacked_rate = generate_now_guess_rate(guess_dict=guess_dict, test_dict=test_dict,
                                                                   total_test_dict_len=total_test_dict_len)
                        print("now attacked {}% of guess task".format(now_attacked_rate*100))

                        # temp_attack_success_pass_path = os.path.join(args.save_folder, 'temp_attack_success_pass.txt')
                        with open(temp_pass_dict_path, 'wb') as f:
                            pickle.dump(guess_dict, f)
                            print("save temp file {}".format(temp_pass_dict_path))

                        if len(dynamic_attacked_list) > 0:
                            with open(temp_attack_success_path, 'wb') as f:
                                pickle.dump(dynamic_attacked_list, f)
                                print("save temp success attack file {}".format(temp_attack_success_path))

                        idx_save_dict = {}
                        sample_idx = iteration * args.batch_size
                        idx_save_dict['sample_idx'] = sample_idx
                        idx_save_dict['is_first_init_dynamic'] = first_init_dynamic
                        idx_save_dict['sigma'] = sigma
                        idx_save_dict['step_size'] = args.step_size
                        idx_save_dict['epoch_time'] = epoch_time

                        with open(temp_sampling_idx_path, 'wb') as f:
                            pickle.dump(idx_save_dict, f)
                            print("save temp sampling idx file {}".format(temp_sampling_idx_path))

                    if len(guess_dict) > args.num_sample:
                        reach_max_guess = True
                        print("generate {} passwords, break".format(len(guess_dict)))
                        break
                    iteration += 1
                except StopIteration:
                    print("reach the end of the dataset")
                    if len(dynamic_attacked_list) > 0:
                        if first_reach_end:
                            first_reach_end = False
                        else:
                            sigma += args.step_size
                            print("increase sigma to {}".format(sigma))

                        # success_attack_list_save_folder = os.path.join(args.save_folder,
                        #                                                'success_attack_list_checkpoint_{}'.format(sigma))
                        temp_attack_success_path_sigma = temp_attack_success_path.replace(".pkl", "") + "_{}.pkl".format(sigma)
                        with open(temp_attack_success_path_sigma, 'wb') as f:
                            pickle.dump(dynamic_attacked_list, f)
                            print("save temp file {}".format(temp_attack_success_path))

                        print("attacked {} passwords in total".format(len(dynamic_attacked_list)))
                        dynamic_data_provider = PassExBertDataProvider(dataset=args.data, batch_size=args.batch_size,
                                                      vocab_file_path=args.vocab, num_workers=args.num_workers,
                                                      max_len=args.max_seq_len, use_frequency=False, use_random=False,
                                                      dynamic_dataset=True, is_sample=True,
                                                                            dynamic_init_list=dynamic_attacked_list)
                        if len(device_ids) > 1:
                            passExBert_rvae_sampler = passExBertRVAE.module.sampler(dynamic_data_provider,
                                                                                    origin_point_strategy='dynamic_beam',
                                                                                    vertex_num=args.vertex_num,
                                                                                    beam_size=args.beam_size,
                                                                                    temperature=args.temperature)
                        else:
                            passExBert_rvae_sampler = passExBertRVAE.sampler(dynamic_data_provider,
                                                                                   origin_point_strategy='dynamic_beam',
                                                                                   vertex_num=args.vertex_num,
                                                                                   beam_size=args.beam_size,
                                                                                   temperature=args.temperature)
                        passExBert_rvae_sampler_list.append(passExBert_rvae_sampler)
                        first_init_dynamic = False
                        epoch_time += 1
                    break

            if reach_max_guess:
                break

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

        if dynamic_attacked_list is not None and len(dynamic_attacked_list) > 0:
            new_attack_list = dynamic_data_provider.get_success_attack_passwords()
            dynamic_attacked_list.extend(new_attack_list)
            # success_pwd_file = os.path.join(saved_file_path.split('.txt')[0], "_success_attack_passwords.txt")
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
