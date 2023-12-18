import torch
import torch.nn as nn
import numpy as np

from selfModules.bert_embedding import BertForPassEmbedding
from bert_model.modeling import BertConfig
from model.passExBert_encoder import PassExBertEncoder
from model.passExBert_decoder import PassExBertDecoder
from utils.functional import kld_coef, frange_cycle_linear
from utils.sample_method import points_sampling_gpu
from utils.beam_search import BeamHypotheses


class PassExBertRVAE(nn.Module):
    def __init__(self, params, device_ids, data_provider=None, max_len=34, bert_dict_path=None, passExBert_config_path=None,
                 passExBert_config1_path=None, passExBert_embed_type=None, magic_num=None, pad_token_idx=None):
        super(PassExBertRVAE, self).__init__()

        self.params = params
        self.passExBert_embed_type = passExBert_embed_type
        self.magic_num = magic_num
        self.device_ids = device_ids
        self.max_len = max_len
        if data_provider is not None:
            self.data_provider = data_provider
            self.pad_token_idx = data_provider.get_padding_idx()
            self.go_token_idx = data_provider.get_cls_idx()
            self.sep_token_idx = data_provider.get_sep_idx()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        if pad_token_idx is not None:
            self.pad_token_idx = pad_token_idx

        passExBert_config = BertConfig.from_json_file(passExBert_config_path)
        if passExBert_config1_path is not None:
            passExBert_config1 = BertConfig.from_json_file(passExBert_config1_path)
            self.vocab_size = passExBert_config.vocab_size + passExBert_config1.vocab_size
            print("Building PyTorch model from configuration: {}".format(str(passExBert_config)))
            print("Building PyTorch model from configuration: {}".format(str(passExBert_config1)))
        else:
            passExBert_config1 = None
            self.vocab_size = passExBert_config.vocab_size
            print("Building PyTorch model from configuration: {}".format(str(passExBert_config)))

        self.embedding = BertForPassEmbedding(passExBert_config, passExBert_config1,
                                              embed_type=passExBert_embed_type)

        state_dict = torch.load(bert_dict_path, map_location='cpu')
        # model_dict = self.embedding.state_dict()
        #
        # state_dict = {key: value for key, value in state_dict.items() if 'cls' not in key}
        # model_dict.update(state_dict)
        self.embedding.load_state_dict(state_dict, strict=False)
        print("Load passExBert model successfully!")

        if len(self.device_ids) > 1:
            self.embedding = nn.DataParallel(self.embedding, device_ids=self.device_ids)

            for name, param in self.embedding.module.named_parameters():
                param.requires_grad = False
            print("Freeze passExBert model successfully!")
        else:
            for name, param in self.embedding.named_parameters():
                param.requires_grad = False
            print("Freeze passExBert model successfully!")

        if passExBert_embed_type == 'last_four_layers':
            # self.dimension_reduction_layer = nn.Linear(passExBert_config.hidden_size * 4,
            #                                            self.params.encoder_rnn_size)

            self.context_to_mu = nn.Linear(passExBert_config.hidden_size * 4, self.params.latent_variable_size)
            self.context_to_logvar = nn.Linear(passExBert_config.hidden_size * 4, self.params.latent_variable_size)
            # self.decoder_reduce_dim_linear = nn.Linear(passExBert_config.hidden_size * 4, self.params.encoder_rnn_size)

        elif passExBert_embed_type == 'pooled_output':
            # self.bert_as_encoder_dim_reduction = nn.Linear(passExBert_config.hidden_size,
            #                                                self.params.encoder_rnn_size * 2)
            self.context_to_mu = nn.Linear(passExBert_config.hidden_size, self.params.latent_variable_size)
            self.context_to_logvar = nn.Linear(passExBert_config.hidden_size, self.params.latent_variable_size)
            # self.decoder_reduce_dim_linear = nn.Linear(passExBert_config.hidden_size, self.params.encoder_rnn_size)

            # self.dimension_reduction_layer = nn.Linear(passExBert_config.hidden_size,
            #                                            self.params.encoder_rnn_size)

        elif passExBert_embed_type == 'first_and_last_layer':
            self.context_to_mu = nn.Linear(passExBert_config.hidden_size * 2, self.params.latent_variable_size)
            self.context_to_logvar = nn.Linear(passExBert_config.hidden_size * 2, self.params.latent_variable_size)
            self.decoder_reduce_dim_linear = nn.Linear(passExBert_config.hidden_size * 2, self.params.encoder_rnn_size)
        else:
            # self.dimension_reduction_layer = nn.Linear(passExBert_config.hidden_size,
            #                                            self.params.encoder_rnn_size)
            self.context_to_mu = nn.Linear(passExBert_config.hidden_size, self.params.latent_variable_size)
            self.context_to_logvar = nn.Linear(passExBert_config.hidden_size, self.params.latent_variable_size)
            # self.decoder_reduce_dim_linear = nn.Linear(passExBert_config.hidden_size , self.params.encoder_rnn_size)

        self.decoder_embedding = nn.Embedding(self.vocab_size, self.params.encoder_rnn_size, padding_idx=self.pad_token_idx)


        # nn.init.normal_(self.decoder_embedding.weight, mean=0, std=1)
        # self.encoder_rnn = PassExBertEncoder(self.params)
        self.decoder_rnn = PassExBertDecoder(self.params, self.vocab_size)
        if len(device_ids) > 1:
            # self.encoder_rnn = nn.DataParallel(self.encoder_rnn, device_ids=device_ids)
            self.decoder_rnn = nn.DataParallel(self.decoder_rnn, device_ids=device_ids)
        # self.context_to_mu = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)
        # self.context_to_logvar = nn.Linear(self.params.encoder_rnn_size * 2, self.params.latent_variable_size)

    def forward(self, input_ids, attention_mask, input_lengths, drop_prob, initial_state=None):
        batch_size = input_ids.shape[0]
        if len(self.device_ids) > 1:
            use_cuda = self.embedding.module.bert.embeddings.word_embeddings.weight.is_cuda
        else:
            use_cuda = self.embedding.bert.embeddings.word_embeddings.weight.is_cuda
        if self.passExBert_embed_type == 'pooled_output':
            bert_embedding = self.embedding(input_ids=input_ids, attention_mask=attention_mask)
            context = bert_embedding
            # bert_embedding = self.dimension_reduction_layer(bert_embedding)
            # bert_embedding = torch.cat([bert_embedding] * (self.max_len-1), 1)\
            #     .view(batch_size, self.max_len-1, bert_embedding.shape[-1])

        # elif self.passExBert_embed_type == 'last_four_layers':
        #     bert_embedding = self.embedding(input_ids, attention_mask=attention_mask)
        #     # bert_embedding = bert_embedding[:, 1:, :]
        #     attention_mask[:, 0] = 0
        #
        #     for i, length in enumerate(input_lengths):
        #         attention_mask[i, length-1] = 0
        #
        #     attention_mask = attention_mask.float()
        #     masked_embedding = bert_embedding * attention_mask.unsqueeze(-1)
        #     context = masked_embedding.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

            # bert_embedding = self.dimension_reduction_layer(bert_embedding)
            # packed_input = nn.utils.rnn.pack_padded_sequence(bert_embedding, input_lengths,
            #                                                  batch_first=True, enforce_sorted=False)
            # context = self.encoder_rnn(input=bert_embedding, input_lengths=input_lengths)
        else:
            bert_embedding = self.embedding(input_ids, attention_mask=attention_mask)
            # attention_mask[:, 0] = 0
            # for i, length in enumerate(input_lengths):
            #     attention_mask[i, length-1] = 0
            # attention_mask = attention_mask.float()
            attention_mask = attention_mask.float()
            masked_embedding = bert_embedding * attention_mask.unsqueeze(-1)
            context = masked_embedding.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            # context = bert_embedding.sum(dim=1) / self.max_len.__float__()

        mu = self.context_to_mu(context)
        logvar = self.context_to_logvar(context)
        std = torch.exp(0.5 * logvar)

        z = torch.randn([batch_size, self.params.latent_variable_size])
        if use_cuda:
            z = z.to(self.device_ids[0])
        z = z * std + mu
        kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()

        # self.decoder_rnn.flatten_parameters()
        decoder_input_ids = input_ids.clone()
        decoder_input_ids[decoder_input_ids == self.sep_token_idx] = self.pad_token_idx
        decoder_input = self.decoder_embedding(decoder_input_ids)
        # decoder_input = self.decoder_reduce_dim_linear(bert_embedding)
        output, final_state = self.decoder_rnn(decoder_input=decoder_input, z=z, drop_prob=drop_prob,
                                               lengths=input_lengths, initial_state=initial_state)
        # output, final_state = self.decoder_rnn(decoder_input=bert_embedding, z=z, drop_prob=drop_prob,
        #                                        lengths=input_lengths, initial_state=initial_state)

        return output, final_state, kld

    def learnable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainer(self, optimizer, data_provider, total_iteration):
        train_data_provider = iter(data_provider)
        kld_coef_list = frange_cycle_linear(start=0, stop=1, n_epoch=total_iteration, ratio=1)
        def train(iteration, dropout, use_cuda):
            input_ids, attention_masks, lengths = next(train_data_provider)
            if use_cuda:
                input_ids = input_ids.to(self.device_ids[0])
                attention_masks = attention_masks.to(self.device_ids[0])
                # lengths = lengths.cuda()
            lengths = lengths.int()
            # targets = input_ids.clone().detach()[:, 1:]
            targets = input_ids.clone().detach()
            if use_cuda:
                targets = torch.cat([targets[:, 1:], torch.zeros(targets.shape[0], 1).to(self.device_ids[0])], dim=1).long()
            else:
                targets = torch.cat([targets[:, 1:], torch.zeros(targets.shape[0], 1)], dim=1).long()

            logits, _, kld = self(input_ids, attention_masks, lengths, dropout)
            logits = logits.view(-1, self.vocab_size)
            targets = targets.contiguous().view(-1)
            reconstruction_loss = nn.functional.cross_entropy(logits, targets, ignore_index=self.pad_token_idx)

            loss = self.magic_num * reconstruction_loss + kld_coef(iteration) * kld
            # loss = self.magic_num * reconstruction_loss + kld_coef_list[iteration] * kld

            optimizer.zero_grad()
            loss.backward()

            # for name, param in self.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad)

            if loss != loss:
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        print(name, param.grad)
                raise Exception('NaN in loss, crack!')

            optimizer.step()
            return reconstruction_loss, kld, kld_coef(iteration)

        return train

    def validater(self, data_provider):
        val_data_provider = iter(data_provider)

        def validate(dropout, use_cuda=True):
            input_ids, attention_masks, lengths = next(val_data_provider)
            if use_cuda:
                input_ids = input_ids.to(self.device_ids[0])
                attention_masks = attention_masks.to(self.device_ids[0])
            lengths = lengths.int()
            # targets = input_ids.clone().detach()[:, 1:]
            targets = input_ids.clone().detach()
            if use_cuda:
                targets = torch.cat([targets[:, 1:], torch.zeros(targets.shape[0], 1).to(self.device_ids[0])], dim=1).long()
            else:
                targets = torch.cat([targets[:, 1:], torch.zeros(targets.shape[0], 1)], dim=1).long()
            logits, _, kld = self(input_ids, attention_masks, lengths, dropout)

            # print_input_ids = []
            # print_predicted_targets = []
            # logits_max = logits.argmax(-1)
            # total_num = 0
            # count = 0
            # count_except_go_end = 0
            #
            # for i in range(len(input_ids)):
            #     print_input_ids.append(input_ids[i, :lengths[i]])
            #     print_predicted_targets.append(logits_max[i, :lengths[i]])
            #     count += sum(1 for x, y in zip(input_ids[i, :lengths[i]], logits_max[i, :lengths[i]]) if x == y)
            #     count_except_go_end += sum(1 for x, y in zip(input_ids[i, :lengths[i]], logits_max[i, :lengths[i]])
            #                                if x == y and x != self.go_token_idx and x != self.sep_token_idx)
            #     total_num += lengths[i]
            #
            # # count = sum(1 for x, y in zip(print_input_ids, print_predicted_targets) if torch.equal(x, y))
            # # count_except_go_end = sum(1 for x, y in zip(print_input_ids, print_predicted_targets) if torch.equal(x, y)
            # #                           and x != self.go_token_idx and x != self.sep_token_idx)
            #
            # print("true targets: {}".format(print_input_ids))
            # print("predicted targets: {}".format(print_predicted_targets))
            # print("predicted right num:{}, predict right except [cls], [sep] num: {} , total num:{}, "
            #       "predict right probility :{:.2f}%, predict right except [cls], [sep] probility :{:.2f}%"
            #       .format(count, count_except_go_end, total_num, count/total_num*100,
            #               count_except_go_end/(total_num-len(input_ids)*2)*100))

            logits = logits.view(-1, self.vocab_size)

            targets = targets.contiguous().view(-1)
            reconstruction_loss = nn.functional.cross_entropy(logits, targets, ignore_index=self.pad_token_idx)
            return reconstruction_loss, kld

        # def validate(use_cuda):
        #     input_ids, attention_masks, lengths = next(val_data_provider)
        #
        #     if use_cuda:
        #         input_ids = input_ids.cuda()
        #         attention_masks = attention_masks.cuda()
        #     lengths = lengths.int()
        #     targets = input_ids.clone().detach()
        #
        #     z = self.calcu_z_from_input(input_ids, attention_masks, lengths)
        #     samples, z = self.sample(batch_size=len(input_ids), z=z, mode='greedy')
        #     reconstruction_loss = nn.functional.cross_entropy(logits, targets, ignore_index=self.pad_token_idx)

        return validate



    def sample(self, batch_size, z=None, mode='greedy'):
        if z is None:
            z = torch.randn([batch_size, self.params.latent_variable_size])
        else:
            batch_size = z.size(0)

        initial_state = None
        input_sequence = torch.zeros(batch_size).fill_(self.go_token_idx).long()

        # required for dynamic stopping of sentence generation
        # sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()  # all idx of batch
        # all idx of batch which are still generating
        # sequence_running = torch.arange(0, batch_size, out=self.tensor()).long()
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
        # idx of still generating sequences with respect to current loop
        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long()

        generations = self.tensor(batch_size, self.max_len).fill_(self.pad_token_idx).long()

        decoder_attention_mask = torch.ones(batch_size, out=self.tensor()).long()
        decoder_attention_mask = decoder_attention_mask.unsqueeze(1)

        t = 0
        while t < self.max_len and len(running_seqs) > 0:
            if t == 0:
                input_sequence = torch.Tensor(batch_size).fill_(self.go_token_idx).long()
                if torch.cuda.is_available():
                    input_sequence = input_sequence.to(self.device_ids[0])
                    decoder_attention_mask = decoder_attention_mask.to(self.device_ids[0])

            input_sequence = input_sequence.unsqueeze(1)
            input_embedding = self.decoder_embedding(input_sequence)

            # input_embedding = self.embedding(input_sequence, attention_mask=decoder_attention_mask)
            # input_embedding = self.decoder_reduce_dim_linear(input_embedding)
            lengths = torch.ones(batch_size, out=self.tensor()).long()
            logits, initial_state = self.decoder_rnn(decoder_input=input_embedding, z=z, lengths=lengths,
                                                        initial_state=initial_state, drop_prob=0.0)

            logits = logits.squeeze(1)
            prediction = nn.functional.softmax(logits, dim=-1)
            input_sequence = self._sample(prediction.data)
            # save next input
            generations = self._save_sample(generations, input_sequence, t)
            # update global running sequence
            sequence_mask = (input_sequence != self.sep_token_idx)
            if torch.all(sequence_mask == False):
                break
            # sequence_running = sequence_idx.masked_select(sequence_mask)

            # if len(running_seqs) > 0:
            #     # input_sequence = input_sequence[running_seqs]
            #     # initial_state = initial_state[:, running_seqs]
            #
            #     running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1
        return generations, z

    def beam_search_sample(self, batch_size, z=None, num_beams=2, temperature=1.0):
        if z is None:
            z = torch.randn([batch_size, self.params.latent_variable_size])
        else:
            batch_size = z.size(0)

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=z.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)
        done = [False for _ in range(batch_size)]

        initial_state = None

        generated_hyps = [
            BeamHypotheses(num_beams, self.max_len, early_stopping=True) for _ in range(batch_size)
        ]
        input_ids = torch.full((batch_size * num_beams, 1), self.go_token_idx, dtype=torch.long).to(z.device)
        next_input_ids = input_ids

        z = z.unsqueeze(1).expand(batch_size, num_beams, self.params.latent_variable_size)
        z = z.contiguous().view(batch_size * num_beams, self.params.latent_variable_size)
        lengths = torch.ones(batch_size*num_beams, out=self.tensor()).long()
        cur_len = 1
        while cur_len < self.max_len:
            # model_inputs = self.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask)

            next_input_ids = input_ids[:, -1].unsqueeze(-1)
            input_embedding = self.decoder_embedding(next_input_ids)

            logits, initial_state = self.decoder_rnn(decoder_input=input_embedding, z=z, lengths=lengths,
                                                        initial_state=initial_state, drop_prob=0.0)
            logits = logits[:, -1, :]
            logits = logits / temperature
            scores = nn.functional.log_softmax(logits, dim=-1)
            next_scores = scores + beam_scores[:, None].expand_as(scores)
            next_scores = next_scores.view(
                batch_size, num_beams * self.vocab_size
            )  # 转成(batch_size, num_beams * vocab_size)
            next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=1, largest=True, sorted=True)

            next_batch_beam = []
            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, self.pad_token_idx, 0)] * num_beams)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_size  # 1
                    token_id = beam_token_id % self.vocab_size  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * num_beams + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if (self.sep_token_idx is not None) and (token_id.item() == self.sep_token_idx):
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == num_beams:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
            # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            # 准备下一次循环(下一层的解码)
            # beam_scores: (num_beams * batch_size)
            # beam_tokens: (num_beams * batch_size)
            # beam_idx: (num_beams * batch_size)
            # 这里beam idx shape不一定为num_beams * batch_size，一般是小于等于
            # 因为有些beam id对应的句子已经解码完了 (下面假设都没解码完)
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            bool_idx = torch.zeros(input_ids.size(0), dtype=torch.bool)
            bool_idx[beam_idx] = True

            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            # input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # z = z[beam_idx, :]
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids[beam_idx], beam_tokens.unsqueeze(1)], dim=-1)

            cur_len = cur_len + 1
            # 注意有可能到达最大长度后，仍然有些句子没有遇到eos token，这时done[batch_idx]是false

        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(num_beams):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
                # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
                # 下面选择若干最好的序列输出
                # 每个样本返回几个句子
        output_num_return_sequences_per_batch = num_beams
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad


        sent_lengths = input_ids.new(output_batch_size)
        best = []
        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.pad_token_idx)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_len:
                    decoded[i, sent_lengths[i]] = self.sep_token_idx
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded


    def _sample(self, dist, k=1, mode='greedy'):
        if mode == 'greedy':
            # _, sample = torch.topk(dist, k=1, dim=-1)
            # print(dist.shape)
            sample = torch.multinomial(dist, num_samples=k)
            sample = sample.reshape(-1)
            # if k != 1:
            #     raise ValueError('Greedy sampling does not support k > 1')
        #     sample = torch.multinomial(dist, num_samples=k)
        #     sample = sample.reshape(-1)
        # elif mode == 'beam_search':
        #     _, topk = torch.topk(dist, k=k, dim=-1)
        #     sample = torch.multinomial(topk, num_samples=1).reshape(-1)
        return sample

    def _save_sample(self, save_to, sample, t):
        # select only still running
        running_latest = save_to
        # update token at position t
        running_latest[:, t] = sample.data
        # save back
        save_to = running_latest

        return save_to

    def calcu_z_from_input(self, input_ids, attention_masks, lengths):
        batch_size = input_ids.size(0)
        if len(self.device_ids) > 1:
            use_cuda = self.embedding.module.bert.embeddings.word_embeddings.weight.is_cuda
        else:
            use_cuda = self.embedding.bert.embeddings.word_embeddings.weight.is_cuda
        if use_cuda:
            input_ids = input_ids.to(self.device_ids[0])
            attention_masks = attention_masks.to(self.device_ids[0])
        lengths = lengths.int()

        if self.passExBert_embed_type == 'pooled_output':
            bert_embedding = self.embedding(input_ids=input_ids, attention_mask=attention_masks)
            context = self.bert_as_encoder_dim_reduction(bert_embedding)

        else:
            # bert_embedding = self.embedding(input_ids, attention_mask=attention_masks)
            # bert_embedding = self.dimension_reduction_layer(bert_embedding)
            # context = self.encoder_rnn(input=bert_embedding, input_lengths=lengths)

            bert_embedding = self.embedding(input_ids, attention_mask=attention_masks)
            # attention_masks[:, 0] = 0
            #
            # for i, length in enumerate(lengths):
            #     attention_masks[i, length - 1] = 0

            attention_mask = attention_masks.float()
            masked_embedding = bert_embedding * attention_mask.unsqueeze(-1)
            context = masked_embedding.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            # context = bert_embedding.sum(dim=1) / lengths.unsqueeze(-1).float()

        mu = self.context_to_mu(context)
        logvar = self.context_to_logvar(context)
        std = torch.exp(0.5 * logvar)

        z = torch.randn([batch_size, self.params.latent_variable_size])
        if use_cuda:
            z = z.to(self.device_ids[0])
        z = z * std + mu
        return z

    def sampler(self, data_provider, origin_point_strategy='frequency', vertex_num=None, beam_size=None, temperature=1.0,
                dynamic=False):
        if origin_point_strategy not in ['frequency', 'random', 'gaussian_step', 'gaussian_step_success',
                                         'beam_search_random', 'dynamic_beam', 'dynamic_beam_random']:
            raise ValueError('origin_point_strategy must be frequency or random')
        self.go_token_idx = data_provider.get_cls_idx()
        self.sep_token_idx = data_provider.get_sep_idx()
        # convert_ids_to_passwords = data_provider.convert_ids_to_passwords
        # attack_func = data_provider.attack
        self.beam_size = beam_size

        data_provider_iterator = iter(data_provider)
        if len(self.device_ids) > 1:
            self.use_cuda = self.embedding.module.bert.embeddings.word_embeddings.weight.is_cuda
        else:
            self.use_cuda = self.embedding.bert.embeddings.word_embeddings.weight.is_cuda

        def one_batch_sample(mode='greedy', sample_strategy='gaussian',
                             sigma=None, step_size=None, max_hop=None, sample_batch_size=256):
            input_ids, attention_masks, lengths = next(data_provider_iterator)

            # origin_input = data_provider.convert_ids_to_tokens(input_ids.tolist())
            # print(origin_input)
            z = self.calcu_z_from_input(input_ids, attention_masks, lengths)
            generate_passwords = []

            vertex_num_list = [vertex_num] * z.shape[0]
            if sample_strategy == 'gaussian':
                points_sampled = points_sampling_gpu(points=z, vertex_num=vertex_num_list, sigma=sigma, strategy=sample_strategy)
            elif sample_strategy == "uniform":
                points_sampled = points_sampling_gpu(points=z, vertex_num=vertex_num_list, step_size=step_size, max_hop=max_hop,
                                    strategy=sample_strategy)
            else:
                raise ValueError('sample_strategy must be gaussian or uniform')

            init_batch_size = points_sampled.shape[0] if points_sampled.shape[0] < sample_batch_size \
                else sample_batch_size
            if init_batch_size % 2 == 1:
                init_batch_size -= 1
            for idx in range(0, points_sampled.shape[0], init_batch_size):
                cur_batch = points_sampled[idx: idx + init_batch_size]
                generations, z = self.sample(batch_size=cur_batch.shape[0], z=cur_batch, mode=mode)
                generations = generations.cpu().numpy()
                generations = list(generations)
                new_passwords = data_provider.convert_ids_to_passwords(generations)
                generate_passwords.extend(new_passwords)

                if idx + init_batch_size * 2 >= points_sampled.shape[0]:
                    init_batch_size = points_sampled.shape[0] - (idx + init_batch_size)
                if init_batch_size % 2 == 1:
                    init_batch_size -= 1
                if init_batch_size == 0 or init_batch_size == 1:
                    break

            return generate_passwords

        def one_batch_freq_sample(mode='greedy', sample_strategy='gaussian', sigma=None, step_size=None, max_hop=None,
                                  sample_batch_size=256):
            input_ids, attention_masks, lengths, frequency_value_list = next(data_provider_iterator)

            frequency_value_list = frequency_value_list.numpy().tolist()

            # origin_input = data_provider.convert_ids_to_tokens(input_ids.tolist())
            # print(origin_input)
            z = self.calcu_z_from_input(input_ids, attention_masks, lengths)

            generate_passwords = []

            if sample_strategy == 'gaussian':
                points_sampled = points_sampling_gpu(points=z, vertex_num=frequency_value_list, sigma=sigma, strategy=sample_strategy)
            elif sample_strategy == "uniform":
                points_sampled = points_sampling_gpu(points=z, vertex_num=frequency_value_list, step_size=step_size, max_hop=max_hop,
                                    strategy=sample_strategy)
            else:
                raise ValueError('sample_strategy must be gaussian or uniform')

            init_batch_size = points_sampled.shape[0] if points_sampled.shape[0] < sample_batch_size \
                else sample_batch_size
            for idx in range(0, points_sampled.shape[0], init_batch_size):
                cur_batch = points_sampled[idx: idx + init_batch_size]
                generations, z = self.sample(batch_size=cur_batch.shape[0], z=cur_batch, mode=mode)

                generations = generations.cpu().numpy()
                generations = list(generations)
                new_passwords = data_provider.convert_ids_to_passwords(generations)
                generate_passwords.extend(new_passwords)

                if idx + init_batch_size * 2 >= points_sampled.shape[0]:
                    init_batch_size = points_sampled.shape[0] - (idx + init_batch_size)
                if init_batch_size == 0:
                    break

            return generate_passwords

        def one_batch_gaussian_step(mode='greedy', sample_strategy='gaussian', sigma=None, max_hop=None,
                                       step_size=None, needs_z=False):
            input_ids, attention_masks, lengths, = next(data_provider_iterator)
            z = self.calcu_z_from_input(input_ids, attention_masks, lengths)
            # sigma_list = [sigma] * z.shape[0]
            origin_input = data_provider.convert_ids_to_passwords(input_ids.tolist())
            existed_passwords = set(origin_input)
            generate_passwords = []
            vertex_num_list = [vertex_num] * z.shape[0]
            # sigma_list = [sigma] * z.shape[0]
            sigma_list = torch.ones(z.shape[0], device=z.device) * sigma

            return_z_list = []

            for hop_num in range(max_hop):
                points_sampled = points_sampling_gpu(points=z, vertex_num=vertex_num_list, sigma_list=sigma_list,
                                                     strategy=sample_strategy)
                if needs_z:
                    return_z_list.extend(points_sampled)
                generations, _ = self.sample(batch_size=points_sampled.shape[0], z=points_sampled, mode=mode)
                generations = generations.cpu().numpy()
                generations = list(generations)
                new_passwords = data_provider.convert_ids_to_passwords(generations)
                generate_passwords.extend(new_passwords)
                new_password_num = torch.zeros(z.shape[0]).to(z.device)
                new_z_tensor = torch.zeros_like(z)
                for idx in range(len(new_passwords)):
                    if new_passwords[idx] not in existed_passwords:
                        new_origin_idx = int(idx / vertex_num)
                        existed_passwords.add(new_passwords[idx])
                        new_password_num[new_origin_idx] += 1
                        new_z_tensor[new_origin_idx].add_(points_sampled[idx])
                sigma_list[new_password_num == 0] += step_size
                new_password_num = new_password_num.unsqueeze(1)
                z = torch.where(new_password_num != 0, new_z_tensor / new_password_num, z)

            if needs_z:
                return generate_passwords, return_z_list
            return generate_passwords

        # def one_batch_gaussian_step_attack_success(mode='greedy', sample_strategy='gaussian', sigma=None, max_hop=None,
        #                                            step_size=None):
        #     input_ids, attention_masks, lengths, = next(data_provider)
        #     z = self.calcu_z_from_input(input_ids, attention_masks, lengths)
        #     origin_input = convert_ids_to_passwords(input_ids.tolist())
        #     existed_passwords = set(origin_input)
        #     generate_passwords = []
        #     vertex_num_list = [vertex_num] * z.shape[0]
        #     sigma_list = torch.ones(z.shape[0], device=z.device) * sigma
        #
        #     for hop_num in range(max_hop):
        #         points_sampled = points_sampling_gpu(points=z, vertex_num=vertex_num_list, sigma_list=sigma_list,
        #                                              strategy=sample_strategy)
        #         generations, _ = self.sample(batch_size=points_sampled.shape[0], z=points_sampled, mode=mode)
        #         generations = generations.cpu().numpy()
        #         generations = list(generations)
        #         new_passwords = convert_ids_to_passwords(generations)
        #         generate_passwords.extend(new_passwords)
        #         new_password_num = torch.zeros(z.shape[0]).to(z.device)
        #         new_z_tensor = torch.zeros_like(z)
        #         for idx in range(len(new_passwords)):
        #             if new_passwords[idx] not in existed_passwords:
        #                 new_origin_idx = int(idx / vertex_num)
        #                 existed_passwords.add(new_passwords[idx])
        #                 new_password_num[new_origin_idx] += 1
        #                 new_z_tensor[new_origin_idx].add_(points_sampled[idx])
        #         sigma_list[new_password_num == 0] += step_size
        #         new_password_num = new_password_num.unsqueeze(1)
        #         z = torch.where(new_password_num != 0, new_z_tensor / new_password_num, z)
        #
        #     return generate_passwords

        def one_batch_beam_search_generate(mode='beam_search', sample_strategy='gaussian', sigma=None, max_hop=None, step_size=None,
                                           sample_batch_size=5012):
            input_ids, attention_masks, lengths = next(data_provider_iterator)

            # origin_input = data_provider.convert_ids_to_tokens(input_ids.tolist())
            # print(origin_input)
            z = self.calcu_z_from_input(input_ids, attention_masks, lengths)
            generate_passwords = []

            vertex_num_list = [vertex_num] * z.shape[0]
            if sample_strategy == 'gaussian':
                points_sampled = points_sampling_gpu(points=z, vertex_num=vertex_num_list, sigma=sigma,
                                                     strategy=sample_strategy)
            elif sample_strategy == "uniform":
                points_sampled = points_sampling_gpu(points=z, vertex_num=vertex_num_list, step_size=step_size,
                                                     max_hop=max_hop,
                                                     strategy=sample_strategy)
            else:
                raise ValueError('sample_strategy must be gaussian or uniform')

            init_batch_size = points_sampled.shape[0] if points_sampled.shape[0] < sample_batch_size \
                else sample_batch_size
            if init_batch_size % 2 == 1:
                init_batch_size -= 1
            for idx in range(0, points_sampled.shape[0], init_batch_size):
                cur_batch = points_sampled[idx: idx + init_batch_size]
                try:
                    generations = self.beam_search_sample(batch_size=cur_batch.shape[0], z=cur_batch, num_beams=self.beam_size,
                                                      temperature=temperature)

                except Exception as e:
                    print(e)
                    continue

                generations = generations.cpu().numpy()
                generations = list(generations)
                new_passwords = data_provider.convert_ids_to_passwords(generations)
                generate_passwords.extend(new_passwords)

                if idx + init_batch_size * 2 >= points_sampled.shape[0]:
                    init_batch_size = points_sampled.shape[0] - (idx + init_batch_size)
                if init_batch_size % 2 == 1:
                    init_batch_size -= 1
                if init_batch_size == 0 or init_batch_size == 1:
                    break

            if dynamic:
                attack_num, attack_rate, new_attacked_passwords = data_provider.attack(generate_passwords)
                return generate_passwords, new_attacked_passwords, attack_num, attack_rate

            return generate_passwords

        def one_batch_beam_search_attack_success_generate(mode='beam_search', sample_strategy='gaussian', sigma=None, max_hop=None, step_size=None,
                                           sample_batch_size=10240):
            input_ids, attention_masks, lengths = next(data_provider_iterator)

            # origin_input = data_provider.convert_ids_to_tokens(input_ids.tolist())
            # print(origin_input)
            z = self.calcu_z_from_input(input_ids, attention_masks, lengths)
            generate_passwords = []

            vertex_num_list = [vertex_num] * z.shape[0]
            if sample_strategy == 'gaussian':
                points_sampled = points_sampling_gpu(points=z, vertex_num=vertex_num_list, sigma=sigma,
                                                     strategy=sample_strategy)
            elif sample_strategy == "uniform":
                points_sampled = points_sampling_gpu(points=z, vertex_num=vertex_num_list, step_size=step_size,
                                                     max_hop=max_hop,
                                                     strategy=sample_strategy)
            else:
                raise ValueError('sample_strategy must be gaussian or uniform')

            init_batch_size = points_sampled.shape[0] if points_sampled.shape[0] < sample_batch_size \
                else sample_batch_size
            if init_batch_size % 2 == 1:
                init_batch_size -= 1
            for idx in range(0, points_sampled.shape[0], init_batch_size):
                cur_batch = points_sampled[idx: idx + init_batch_size]
                generations = self.beam_search_sample(batch_size=cur_batch.shape[0], z=cur_batch, num_beams=self.beam_size,
                                                      temperature=temperature)

                generations = generations.cpu().numpy()
                generations = list(generations)
                new_passwords = data_provider.convert_ids_to_passwords(generations)
                generate_passwords.extend(new_passwords)

                if idx + init_batch_size * 2 >= points_sampled.shape[0]:
                    init_batch_size = points_sampled.shape[0] - (idx + init_batch_size)
                if init_batch_size % 2 == 1:
                    init_batch_size -= 1
                if init_batch_size == 0 or init_batch_size == 1:
                    break

            attack_num, attack_rate, new_attacked_passwords = data_provider.attack(generate_passwords)
            data_provider.add_dynamic_data(new_attacked_passwords)

            return generate_passwords, new_attacked_passwords, attack_num, attack_rate


        if origin_point_strategy == 'frequency':
            return one_batch_freq_sample
        elif origin_point_strategy == 'random':
            return one_batch_sample
        elif origin_point_strategy == 'gaussian_step':
            return one_batch_gaussian_step
        elif origin_point_strategy == 'beam_search_random' or origin_point_strategy == 'dynamic_beam_random':
            return one_batch_beam_search_generate
        elif origin_point_strategy == 'dynamic_beam':
            return one_batch_beam_search_attack_success_generate
