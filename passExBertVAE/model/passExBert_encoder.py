import torch as t
import torch.nn as nn
import torch.nn.functional as F

from utils.functional import parameters_allocation_check


class PassExBertEncoder(nn.Module):
    def __init__(self, params):
        super(PassExBertEncoder, self).__init__()

        self.params = params

        self.rnn = nn.LSTM(input_size=self.params.encoder_rnn_size,
                           hidden_size=self.params.encoder_rnn_size,
                           num_layers=self.params.encoder_num_layers,
                           batch_first=True,
                           bidirectional=True)

    def forward(self, input, input_lengths):
        """
        :param input: [batch_size, seq_len, embed_size] tensor
        :return: context of input sentenses with shape of [batch_size, latent_variable_size]
        """

        batch_size = input.size(0)
        packed_input = nn.utils.rnn.pack_padded_sequence(input, input_lengths.cpu(),
                                                         batch_first=True, enforce_sorted=False)

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'

        ''' Unfold rnn with zero initial state and get its final state from the last layer
        '''

        _, (_, final_state) = self.rnn(packed_input)

        final_state = final_state.view(self.params.encoder_num_layers, 2, -1, self.params.encoder_rnn_size)
        final_state = final_state[-1]
        h_1, h_2 = final_state[0], final_state[1]
        final_state = t.cat([h_1, h_2], 1)

        return final_state
