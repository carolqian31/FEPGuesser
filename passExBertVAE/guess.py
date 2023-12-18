import multiprocessing
import time
import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
from matplotlib.backends.backend_pdf import PdfPages 

def get_test_set(file = "test.txt"):
    test_set = {}
    total = 0
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace(" ", "").strip()
            if line in test_set:
                test_set[line] += 1
            else:
                test_set[line] = 1
            total += 1
    
    return test_set, total

def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'

def Create_fig(x_list, y_list, name_list, figure_store_path):
    n = len(x_list)
    pp = PdfPages(figure_store_path)
    for i in range(n):
        plt.plot(x_list[i], y_list[i], label = name_list[i])
    plt.grid()
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
    plt.ylabel('Percentage of Cracked Passwords', fontsize = 10)
    plt.xlabel('Number of guesses', fontsize = 10)
    plt.legend()
    pp.savefig(bbox_inches='tight')
    plt.close()
    pp.close()

def attack_(test_data, attack_dic, N):
    TestingSet, total_test = get_test_set(test_data)
    ChunkSize = 100000
    count = 1
    x_ = [0]
    y_ = [0]
    proba_sum = 0
    reader = pd.read_csv(attack_dic, header=None, sep='\t', iterator=True, error_bad_lines=False,
                         quoting=csv.QUOTE_NONE)
    while count < N:
        chunk = reader.get_chunk(ChunkSize)
        AttackSet = chunk[0].to_list()
        index = 0 + count
        for passwd in AttackSet:
            index = index + 1
            if passwd in TestingSet:
                proba_sum = proba_sum + TestingSet[passwd] / total_test
                TestingSet.pop(passwd)
                x_.append(index)
                y_.append(proba_sum)
        count = count + ChunkSize
    return x_, y_


def attack_without_weight(test_data, attack_dic, N):
    TestingSet, total_test = get_test_set(test_data)
    ChunkSize = 100000
    count = 1
    x_ = [0]
    y_ = [0]
    proba_sum = 0
    success_attacked_set = {}
    reader = pd.read_csv(attack_dic, header=None, sep='\t', iterator=True, error_bad_lines=False,
                         quoting=csv.QUOTE_NONE)

    while count < N:
        chunk = reader.get_chunk(ChunkSize)
        AttackSet = chunk[0].to_list()
        index = 0 + count
        for passwd in AttackSet:
            index = index + 1
            if passwd in TestingSet and \
                    (passwd not in success_attacked_set or success_attacked_set[passwd] < TestingSet[passwd]):
                proba_sum = proba_sum + 1 / total_test
                TestingSet.pop(passwd)
                if passwd in success_attacked_set:
                    success_attacked_set[passwd] += 1
                else:
                    success_attacked_set[passwd] = 1
                x_.append(index)
                y_.append(proba_sum)
        count = count + ChunkSize

if __name__ == "__main__":
    # N = 6 * np.power(10, 6)
    # attack_file_list = ["csdn_frequency_order_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_32.txt",
    #                     "csdn_frequency_order_guess_vertex_num_10_n_1000000_sigma_0.1.txt",
    #                     "csdn_frequency_order_guess_vertex_num_10_n_1000000_sigma_0.5.txt",
    #                     "csdn_frequency_order_guess_vertex_num_10_n_1000000_sigma_1.0.txt"
    #                     ]
    # attack_file_list = ['csdn_frequency_head_1000_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_256.txt',
    #                     'csdn_frequency_head_2000_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_256.txt',
    #                     'csdn_frequency_head_3000_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_128.txt',
    #                     'csdn_frequency_head_4000_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_256.txt',
    #                     'csdn_frequency_head_200000_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_256.txt',
    #                      'csdn_origin_order_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_32.txt',
    #                      'csdn_frequency_order_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_32.txt',
    #                     'csdn_frequency_order_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_256_no_ocsvm.txt'
    #                      ]
    # attack_file_list = ['csdn_frequency_head_200000_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_256.txt',
    #                      'csdn_origin_order_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_32.txt',
    #                      'csdn_frequency_order_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_32.txt',
    #                     'csdn_frequency_order_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_256_no_ocsvm.txt',
    #                     'csdn_frequency_head_200000_guess_vertex_num_10_gaussian_sigma_0.05_batch_size_128_no_ocsvm.txt',
    #                     'csdn_density_order_manhattan_density_0.4_guess_vertex_num_density_ln_n_1000000_gaussian_sigma_0.05_batch_size_32_no_ocsvm.txt',
    #                     'csdn_density_order_manhattan_density_0.5_guess_vertex_num_density_ln_n_1000000_gaussian_sigma_0.05_batch_size_32_no_ocsvm.txt',
    #                     'csdn_density_order_manhattan_density_0.3_guess_vertex_num_density_ln_n_1000000_gaussian_sigma_0.05_batch_size_32_no_ocsvm.txt',
    #                     'csdn_density_order_manhattan_density_0.3_guess_vertex_num_density_ln_plus_10_gaussian_sigma_0.05_batch_size_128_no_ocsvm.txt'
    #                      ]
    # attack_file_list = ['csdn_frequency_head_200000_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_256.txt',
    #                     'csdn_origin_order_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_32.txt',
    #                     'csdn_frequency_order_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_32.txt',
    #                     'csdn_frequency_order_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_256_no_ocsvm.txt',
    #                     'csdn_frequency_head_200000_guess_vertex_num_10_gaussian_sigma_0.05_batch_size_128_no_ocsvm.txt'
    #                     ]
    # attack_file_list = ['csdn_origin_order_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_32.txt',
    #                     'csdn_frequency_order_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_256_no_ocsvm.txt',
    #                     'csdn_frequency_head_200000_guess_vertex_num_10_gaussian_sigma_0.05_batch_size_128_no_ocsvm.txt',
    #                     'csdn_density_order_manhattan_density_0.3_guess_vertex_num_density_ln_n_1000000_gaussian_sigma_0.05_batch_size_32_no_ocsvm.txt',
    #                     'csdn_density_order_manhattan_density_0.3_guess_vertex_num_density_ln_plus_10_gaussian_sigma_0.5_batch_size_128_no_ocsvm.txt',
    #                     'csdn_density_order_manhattan_density_0.3_guess_vertex_num_density_ln_plus_10_gaussian_sigma_0.05_batch_size_128_no_ocsvm.txt',
    #                     'csdn_density_order_manhattan_density_0.4_guess_vertex_num_density_ln_gaussian_sigma_0.01_batch_size_128_no_ocsvm.txt',
    #                     'csdn_density_order_manhattan_density_0.4_guess_vertex_num_density_ln_n_1000000_gaussian_sigma_0.05_batch_size_32_no_ocsvm.txt',
    #                     'csdn_density_order_manhattan_density_0.5_guess_vertex_num_density_ln_n_1000000_gaussian_sigma_0.05_batch_size_32_no_ocsvm.txt'
    #                     ]
    # attack_file_list = ['csdn_origin_order_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_32.txt',
    #                     'csdn_frequency_order_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_256_no_ocsvm.txt',
    #                     'csdn_frequency_head_200000_guess_vertex_num_10_gaussian_sigma_0.05_batch_size_128_no_ocsvm.txt',
    #                     'csdn_density_order_manhattan_density_0.4_guess_vertex_num_density_ln_gaussian_sigma_0.01_batch_size_128_no_ocsvm.txt',
    #                     'csdn_density_order_manhattan_density_0.4_guess_vertex_num_density_ln_n_1000000_gaussian_sigma_0.05_batch_size_32_no_ocsvm.txt',
    #                     'csdn_density_order_manhattan_density_0.1_guess_vertex_num_density_ln_plus_5_gaussian_sigma_0.05_batch_size_256_no_ocsvm.txt',
    #                     'csdn_density_order_manhattan_density_0.3_guess_vertex_num_density_ln_plus_5_gaussian_sigma_0.05_batch_size_256_no_ocsvm.txt'
    #                     ]
    attack_file_list = ['csdn_origin_order_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_32.txt',
                        'csdn_frequency_order_guess_vertex_num_10_n_1000000_sigma_0.05_batch_size_256_no_ocsvm.txt',
                        'csdn_frequency_head_200000_guess_vertex_num_10_gaussian_sigma_0.05_batch_size_128_no_ocsvm.txt',
                        'csdn_density_order_manhattan_density_0.4_guess_vertex_num_density_ln_gaussian_sigma_0.01_batch_size_128_no_ocsvm.txt',
                        'csdn_density_order_manhattan_density_0.4_guess_vertex_num_density_ln_n_1000000_gaussian_sigma_0.05_batch_size_32_no_ocsvm.txt'
                        ]
    # attack_file_list = ["frequency_order_guess_vertex_num_10_n_1000000_sigma_0.5.txt"]
    TestSetFileName = "./data/csdn/test.txt"
    FigurePath = "./guess/csdn_gaussian_no_ocsvm_density_0.4_sigma_compare.pdf"
    x_list = []
    y_list = []
    # name_list = ['frequency_sigma=0.05', 'frequency_sigma=0.1', 'frequency_sigma=0.5', 'frequency_sigma=1.0']
    # name_list = [ 'frequency_head_1000_sigma=0.05', 'frequency_head_2000_sigma=0.05', 'frequency_head_3000_sigma=0.05',
    #               'frequency_head_4000_sigma=0.05', 'frequency_head_200000_sigma=0.05', 'origin_sigma=0.05',
    #               'frequency_sigma=0.05', 'frequency_sigma=0.05_no_ocsvm']
    # name_list = ['frequency_head_200000_sigma=0.05', 'origin_sigma=0.05',
    #              'frequency_sigma=0.05', 'frequency_sigma=0.05_no_ocsvm', 'frequency_head_200000_sigma=0.05_no_ocsvm',
    #              'density_0.4_sigma=0.05_no_ocsvm', 'density_0.5_sigma=0.05_no_ocsvm', 'density_0.3_sigma=0.05_no_ocsvm',
    #              'density_0.3_ln_plus_10_sigma=0.05_no_ocsvm']
    # name_list = ['frequency_head_200000_sigma=0.05', 'origin_sigma=0.05',
    #              'frequency_sigma=0.05', 'frequency_sigma=0.05_no_ocsvm', 'frequency_head_200000_sigma=0.05_no_ocsvm']
    name_list = ['origin_sigma=0.05', 'frequency_sigma=0.05_no_ocsvm', 'frequency_head_200000_sigma=0.05_no_ocsvm',
                 'density_0.4_sigma=0.01_no_ocsvm', 'density_0.4_sigma=0.05_no_ocsvm']
    for file in attack_file_list:
        file_path = './guess/'+file
        N = len(open(file_path, 'r').readlines())
        x_, y_ = attack_(TestSetFileName, file_path, N)
        x_list.append(x_)
        y_list.append(y_)
    Create_fig(x_list, y_list, name_list, FigurePath)
    print("done! fig has been saved to {}".format(FigurePath))

'''
Exp 1.
      pcfg      omen
csdn  38.60     27.97
000   23.12     9.48
rock  59.12     32.99
'''
