import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import numpy as np
import torch
import time
import pickle
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from utils.passExBert_data_loader import PassExBertDataProvider
from utils.parameters import Parameters, PassExBertParameters
from model.passexbert_rvae import PassExBertRVAE
from utils.Logger import Logger


def setup():
    parser = argparse.ArgumentParser(description='RVAE')
    parser.add_argument('--epoch', type=int, default=5, help='epoch num')
    parser.add_argument('--iteration_num', type=int, default=None, help='iteration num')
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS',
                        help='batch size (default: 32)')
    parser.add_argument('--use_cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning_rate', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                        help='dropout (default: 0.3)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--data', type=str, default='csdn', help='the data file name')
    parser.add_argument('--experiment_folder', type=str, default=None, help="the experiment saved folder if need")
    parser.add_argument('--num_workers', type=int, default=0, help="the number of workers for data loader")
    parser.add_argument('--is_chunk', default=False, action='store_true', help='whether to use chunk')
    parser.add_argument('--magic_num', default=None, type=int, help="the magic number for vae training")
    parser.add_argument('--experiment_info', default=None, type=str, help="the experiment info")
    parser.add_argument('--embed_type', default='passExBert', type=str, choices=['word2vec', 'passExBert'],
                        help="the embedding type")
    parser.add_argument('--bert_dict_path', default=None, type=str, help="the bert dict path")
    parser.add_argument('--config', default=None, type=str, nargs='+', help="bert config file path")
    parser.add_argument('--passExBert_embed_type', default=None, type=str,
                        choices=['pooled_output', 'last_layer', 'last_four_layers', 'sixth_layer', 'second_layer',
                                  'first_and_last_layer'],
                        help="passExBert embedding type")
    parser.add_argument("--device", required=True, type=int, nargs="+",
                        help="GPU device ids to use e.g. --device 0 1. If input -1, use CPU.")
    parser.add_argument('--vocab', default=None, type=str, help="the vocab file path")
    parser.add_argument('--max_seq_len', default=34, type=int, help="the max sequence length")

    args = parser.parse_args()
    return args


def get_embedding_vae_path(experiment_folder, dataset_name, is_chunk, magic_num=None, dropout=None, learning_rate=None,
                           experiment_info=None):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    dataset_folder = dataset_name if not is_chunk else dataset_name + '_chunk'

    if experiment_folder is not None:
        save_folder = os.path.join(current_folder, 'data', dataset_folder, experiment_folder)
    else:
        save_folder = os.path.join(current_folder, 'data', dataset_folder)

    embedding_path = os.path.join(save_folder, 'word_embeddings.npy')

    experiment_info_str = '' if experiment_info is None else experiment_info
    if magic_num is not None:
        save_folder = os.path.join(save_folder, str(magic_num)+'_'+str(dropout)+'_'+str(learning_rate)) + '_'+ \
                      experiment_info_str

    vae_path = os.path.join(save_folder, 'trained_RVAE')
    file_saved_path = os.path.join(save_folder, 'train_loss.png')

    return embedding_path, vae_path, file_saved_path


def get_loss_fig(x, ce_values, kld_values, file_saved_path):
    plt.plot(x, ce_values, label="Reconstruction Loss")
    plt.plot(x, kld_values, label="KLD Loss")

    plt.xlabel("Train Iteration")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid()

    plt.savefig(file_saved_path, dpi=400)
    plt.close()


def get_weight_fig(x, weight_values, file_saved_path):
    plt.plot(x, weight_values, label="KLD Weight")

    plt.xlabel("Train Iteration")
    plt.ylabel("Weight")

    plt.legend()
    plt.grid()

    plt.savefig(file_saved_path, dpi=400)
    plt.close()


def train():
    ce_result = []
    kld_result = []

    x = []
    ce_values = []
    kld_values = []
    for iteration in range(total_iteration):
        cross_entropy, kld, coef = train_step(iteration, args.batch_size, args.use_cuda, args.dropout)

        if iteration % 20 == 0:
            print('------------TRAIN-------------')
            print('ITERATION :', iteration)
            print('CROSS-ENTROPY :', cross_entropy.data.cpu().numpy())
            print('KLD :', kld.data.cpu().numpy())
            print('KLD-coef :', coef)

            data_provider.is_train = False
            cross_entropy, kld = validate(args.batch_size, args.use_cuda)
            data_provider.is_train = True

            cross_entropy = cross_entropy.data.cpu().numpy()
            kld = kld.data.cpu().numpy()
            print('------------VALID-------------')
            print('CROSS-ENTROPY', cross_entropy)
            print('KLD', kld)

            ce_result += [cross_entropy]
            kld_result += [kld]

            x.append(iteration)
            ce_values.append(args.magic_num * cross_entropy)
            kld_values.append(kld)

        if iteration % 500 == 0:
            # torch.save(rvae.state_dict(), vae_path)
            print("model saved!", "\n")
    # get_loss_fig(x, ce_values, kld_values, file_saved_path)
    print("Train Done!")


if __name__ == '__main__':
    args = setup()

    if args.embed_type == 'word2vec':
        running_name = 'train_vae_chunk' if args.is_chunk else 'train_vae'
        running_name += "_" + args.experiment_folder if args.experiment_folder is not None else ''
    else:
        running_name = "train_vae_passExBert_" + args.passExBert_embed_type
    sys.stdout = Logger(dataset=args.data, stream=sys.stdout, running_operation=running_name)
    sys.stderr = Logger(dataset=args.data, stream=sys.stderr, running_operation=running_name)

    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')



    exbert_data_folder = os.path.join(data_folder, args.data+'_bert')
    vae_path = os.path.join(exbert_data_folder, args.experiment_info)

    if args.device == -1:
        device = 'cpu'
        device_ids = 'cpu'
    else:
        device_ids = args.device
        device = 'cuda:' + str(args.device[0])
    print(device_ids)

    if not os.path.exists(vae_path):
        os.makedirs(vae_path)
        print("Create folder: ", vae_path)

    data_provider = PassExBertDataProvider(dataset=args.data, batch_size=args.batch_size,
                                           vocab_file_path=args.vocab, num_workers=args.num_workers,
                                           max_len=args.max_seq_len)

    valid_data_provider = PassExBertDataProvider(dataset=args.data, batch_size=args.batch_size,
                                                 vocab_file_path=args.vocab, num_workers=args.num_workers,
                                                 max_len=args.max_seq_len, is_train=False)

    parameters = PassExBertParameters()

    if len(args.config) == 1:
        config1 = None
    else:
        config1 = args.config[1]
    passExBertRVAE = PassExBertRVAE(parameters, data_provider=data_provider, bert_dict_path=args.bert_dict_path,
                                    passExBert_config_path=args.config[0], passExBert_config1_path=config1,
                                    passExBert_embed_type=args.passExBert_embed_type, magic_num=args.magic_num,
                                    device_ids=device_ids)

    if device != 'cpu':
        if len(device_ids) > 1:
            passExBertRVAE = nn.DataParallel(passExBertRVAE, device_ids=device_ids)
        passExBertRVAE.to(device)

    if len(device_ids) > 1:
        optimizer = Adam(passExBertRVAE.module.learnable_parameters(), args.learning_rate, eps=1e-4)
    else:
        optimizer = Adam(passExBertRVAE.learnable_parameters(), args.learning_rate)

    train_all_epoch_cross_entropy = []
    train_all_epoch_kld = []

    valid_all_epoch_cross_entropy = []
    valid_all_epoch_kld = []
    all_epoch_iterations = []
    all_epoch_coefs = []

    best_epoch = 0
    best_iteration = 0
    t2 = time.time()
    for epoch in range(args.epoch):
        train_cross_entropy = []
        train_kld = []

        valid_cross_entropy = []
        valid_kld = []
        iterations = []
        coefs = []

        patients = 0

        total_iteration = int(np.ceil(len(data_provider) / args.batch_size)) if args.iteration_num is None \
            else args.iteration_num

        if len(device_ids) > 1:
            train_step = passExBertRVAE.module.trainer(optimizer, data_provider, total_iteration)
            validate = passExBertRVAE.module.validater(valid_data_provider)
        else:
            train_step = passExBertRVAE.trainer(optimizer, data_provider, total_iteration)
            validate = passExBertRVAE.validater(valid_data_provider)

        best_cross_entropy = float('inf')
        best_vae_path = os.path.join(vae_path, 'best_dict_epoch{}.pt'.format(epoch))
        for iteration in range(total_iteration):
            try:
                cross_entropy, kld, coef = train_step(iteration, args.dropout, args.use_cuda)

                if iteration % 50 == 0:
                    cross_entropy = cross_entropy.cpu().detach().numpy()
                    kld = kld.cpu().detach().numpy()
                    # coef = coef.cpu().detach().numpy()
                    print('------------TRAIN-------------')
                    print("epoch {}\t iteration[{}/{}] {}% \t time: {:.4f}\t cross_entropy: {:.4f} \t kld: {:.4f} "
                          "\t coef: {:.4f} \t ".format(epoch, iteration, total_iteration, iteration / total_iteration * 100,
                                                         time.time() - t2, cross_entropy, kld, coef))
                    train_cross_entropy.append(cross_entropy)
                    train_kld.append(kld)
                    coefs.append(coef)

                    passExBertRVAE.eval()
                    with torch.no_grad():
                        cross_entropy, kld = validate(args.dropout, args.use_cuda)

                    passExBertRVAE.train()
                    cross_entropy = cross_entropy.cpu().detach().numpy()
                    kld = kld.cpu().detach().numpy()
                    print('------------VALID-------------')
                    print('CROSS-ENTROPY', cross_entropy)
                    print('KLD', kld)
                    valid_cross_entropy.append(cross_entropy)
                    valid_kld.append(kld)
                    iterations.append(iteration)
                    all_epoch_iterations.append(epoch*total_iteration+iteration)

                if iteration % 200 == 0:
                    if cross_entropy < best_cross_entropy:
                        best_cross_entropy = cross_entropy
                        patients = 0
                        best_epoch = epoch
                        best_iteration = iteration
                        if len(device_ids) > 1:
                            torch.save(passExBertRVAE.module.state_dict(), best_vae_path)
                        else:
                            torch.save(passExBertRVAE.state_dict(), best_vae_path)
                        print("epoch {}, iteration {}, the now best model saved in {}".
                              format(epoch, iteration, best_vae_path))
                    else:
                        patients += 1
                        if patients < 6:
                            print("epoch {}, iteration {}, not best model!".format(epoch, iteration))
                        else:
                            print("epoch {}, iteration {}, early stop!".format(epoch, iteration))
                            break
            except Exception as e:
                print(e)
                break


        last_vae_path = os.path.join(vae_path, 'last_iter_dict_epoch{}.pt'.format(epoch))
        if len(device_ids) > 1:
            torch.save(passExBertRVAE.module.state_dict(), last_vae_path)
        else:
            torch.save(passExBertRVAE.state_dict(), last_vae_path)
        print("epoch {}, the last model saved in {}".format(epoch, last_vae_path))

        print("the best model is in epoch {}, iteration {}".format(best_epoch, best_iteration))

        train_all_epoch_cross_entropy.extend(train_cross_entropy)
        train_all_epoch_kld.extend(train_kld)
        valid_all_epoch_kld.extend(valid_kld)
        valid_all_epoch_cross_entropy.extend(valid_cross_entropy)
        all_epoch_coefs.extend(coefs)

    train_fig_path = os.path.join(vae_path, 'train_loss.png')
    valid_fig_path = os.path.join(vae_path, 'valid_loss.png')
    kl_weight_fig_path = os.path.join(vae_path, 'kl_weight.png')

    train_valid_log_dict = {'iteration': all_epoch_iterations,
                            'train_cross_entropy': train_all_epoch_cross_entropy,
                            'train_kld': train_all_epoch_kld,
                            'valid_cross_entropy': valid_all_epoch_cross_entropy,
                            'valid_kld': valid_all_epoch_kld,
                            'kl_weight': all_epoch_coefs}
    with open(os.path.join(vae_path, 'train_valid_loss_log_dict.pkl'), 'wb') as f:
        pickle.dump(train_valid_log_dict, f)

    get_loss_fig(x=all_epoch_iterations, ce_values=train_all_epoch_cross_entropy,
                 kld_values=train_all_epoch_kld, file_saved_path=train_fig_path)
    get_loss_fig(x=all_epoch_iterations, ce_values=valid_all_epoch_cross_entropy,
                 kld_values=valid_all_epoch_kld, file_saved_path=valid_fig_path)
    get_weight_fig(x=all_epoch_iterations, weight_values=all_epoch_coefs, file_saved_path=kl_weight_fig_path)

    print("done!")