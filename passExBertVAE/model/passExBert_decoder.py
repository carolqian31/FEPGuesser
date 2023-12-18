import torch as t
import torch.nn as nn
import torch.nn.functional as F

from utils.functional import parameters_allocation_check


class PassExBertDecoder(nn.Module):
    def __init__(self, params, word_vocab_size):
        super(PassExBertDecoder, self).__init__()

        self.params = params

        # self.rnn = nn.LSTM(input_size=self.params.latent_variable_size + self.params.encoder_rnn_size,
        #                    hidden_size=self.params.decoder_rnn_size,
        #                    num_layers=self.params.decoder_num_layers,
        #                    batch_first=True)

        self.rnn = nn.LSTM(input_size=self.params.latent_variable_size + self.params.encoder_rnn_size,
                          hidden_size=self.params.decoder_rnn_size,
                          num_layers=self.params.decoder_num_layers,
                          batch_first=True, dtype=t.float32)

        self.fc = nn.Linear(self.params.decoder_rnn_size, word_vocab_size)
        self.vocab_size = word_vocab_size

    def forward(self, decoder_input, z, drop_prob, lengths, padding_value=0, initial_state=None, is_valid=False):
        """
        :param decoder_input: tensor with shape of [batch_size, seq_len, embed_size]
        :param z: sequence context with shape of [batch_size, latent_variable_size]
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout
        :param initial_state: initial state of decoder rnn

        :return: unnormalized logits of sentense words distribution probabilities
                    with shape of [batch_size, seq_len, word_vocab_size]
                 final rnn state with shape of [num_layers, batch_size, decoder_rnn_size]
        """

        assert parameters_allocation_check(self), \
            'Invalid CUDA options. Parameters should be allocated in the same memory'

        [batch_size, seq_len, _] = decoder_input.size()

        '''
            decoder rnn is conditioned on context via additional bias = W_cond * z to every input token
        '''
        decoder_input = F.dropout(decoder_input, drop_prob)

        z = t.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.latent_variable_size)
        decoder_input = t.cat([decoder_input, z], 2)
        if not is_valid:
            decoder_input = t.nn.utils.rnn.pack_padded_sequence(decoder_input, lengths=lengths.cpu(),
                                                                       batch_first=True, enforce_sorted=False)
        rnn_out, final_state = self.rnn(decoder_input, initial_state)
        # rnn_out = t.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, total_length=total_padding_length)
        if not is_valid:
            rnn_out, _ = t.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, padding_value=padding_value,
                                                                    total_length=seq_len)

        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)
        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.vocab_size)

        return result, final_state
