import torch
from torch import nn

from bert_model.modeling import BertModel


class BertForPassEmbedding(nn.Module):
    def __init__(self, config, config1=None, embed_type=None):
        super(BertForPassEmbedding, self).__init__()
        self.bert = BertModel(config, config1)
        if embed_type is None:
            raise ValueError("embed_type must be specified! Please choose from "
                             "[pooled_output, last_layer, last_four_layers, sixth_layer, second_layer, "
                             "first_and_last_layer]")
        self.embed_type = embed_type
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = None
        flat_attention_mask = None
        if token_type_ids is not None:
            flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        if attention_mask is not None:
            flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        encoded_layers, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask,
                                     output_all_encoded_layers=True)

        if self.embed_type == 'last_layer':
            return encoded_layers[-1]
        elif self.embed_type == 'pooled_output':
            return pooled_output
        elif self.embed_type == 'last_four_layers':
            return torch.cat(encoded_layers[-4:], dim=-1)
        elif self.embed_type == 'sixth_layer':
            return encoded_layers[5]
        elif self.embed_type == 'second_layer':
            return encoded_layers[1]
        elif self.embed_type == 'first_and_last_layer':
            return torch.cat((encoded_layers[0], encoded_layers[-1]), dim=-1)
        else:
            raise ValueError("embed_type must be specified! Please choose from "
                             "[pooled_output, last_layer, last_four_layers]")
