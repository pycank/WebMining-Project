import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as Models


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim=None,
        out_dim=None,
        n_head=1,
        score_function="scaled_dot_product",
        dropout=0,
    ):
        """Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        """
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_kx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.w_qx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == "mlp":
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == "bi_linear":
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        self.w_kx.data.uniform_(-stdv, stdv)
        self.w_qx.data.uniform_(-stdv, stdv)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head, ?*k_len, embed_dim) -> (n_head*?, k_len, hidden_dim)
        # qx: (n_head, ?*q_len, embed_dim) -> (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = k.repeat(self.n_head, 1, 1).view(
            self.n_head, -1, self.embed_dim
        )  # (n_head, ?*k_len, embed_dim)
        qx = q.repeat(self.n_head, 1, 1).view(
            self.n_head, -1, self.embed_dim
        )  # (n_head, ?*q_len, embed_dim)
        kx = torch.bmm(kx, self.w_kx).view(
            -1, k_len, self.hidden_dim
        )  # (n_head*?, k_len, hidden_dim)
        qx = torch.bmm(qx, self.w_qx).view(
            -1, q_len, self.hidden_dim
        )  # (n_head*?, q_len, hidden_dim)
        if self.score_function == "scaled_dot_product":
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == "mlp":
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == "bi_linear":
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError("invalid score_function")
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(
            torch.split(output, mb_size, dim=0), dim=-1
        )  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output


import numpy as np
import torch.nn as nn


class DynamicLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=False,
        only_use_last_hidden_state=False,
        rnn_type="LSTM",
    ):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == "LSTM":
            self.RNN = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif self.rnn_type == "GRU":
            self.RNN = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif self.rnn_type == "RNN":
            self.RNN = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional,
            )

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = np.argsort(-x_len)
        x_unsort_idx = torch.LongTensor(np.argsort(x_sort_idx))
        x_len = x_len[x_sort_idx]
        x = x[torch.LongTensor(x_sort_idx)]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(
            x, x_len, batch_first=self.batch_first
        )

        # process using the selected RNN
        if self.rnn_type == "LSTM":
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx
        ]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(
                out_pack, batch_first=self.batch_first
            )  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type == "LSTM":
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx
                ]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)


class MIMN(nn.Module):
    def locationed_memory(self, memory, memory_len):
        # here we just simply calculate the location vector in Model2's manner
        for i in range(memory.size(0)):
            for idx in range(memory_len[i]):
                memory[i][idx] *= 1 - float(idx) / int(memory_len[i])
        return memory

    def __init__(self, embedding_matrix, opt):
        super(MIMN, self).__init__()
        self.opt = opt
        self.img_extractor = nn.Sequential(
            *list(Models.resnet18(pretrained=True).children())[:-1]
        )

        self.embed = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float), freeze=False
        )
        self.bi_lstm_context = DynamicLSTM(
            opt.embed_dim,
            opt.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bi_lstm_aspect = DynamicLSTM(
            opt.embed_dim,
            opt.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bi_lstm_img = DynamicLSTM(
            opt.embed_dim_img,
            opt.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attention_text = Attention(opt.hidden_dim * 2, score_function="mlp")
        self.attention_img = Attention(opt.hidden_dim * 2, score_function="mlp")
        self.attention_text2img = Attention(opt.hidden_dim * 2, score_function="mlp")
        self.attention_img2text = Attention(opt.hidden_dim * 2, score_function="mlp")

        self.gru_cell_text = nn.GRUCell(opt.hidden_dim * 2, opt.hidden_dim * 2)
        self.gru_cell_img = nn.GRUCell(opt.hidden_dim * 2, opt.hidden_dim * 2)

        self.bn = nn.BatchNorm1d(opt.hidden_dim * 2, affine=False)
        self.fc = nn.Linear(opt.hidden_dim * 4, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices, imgs, num_imgs = (
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
        )
        text_memory_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        imgs_memory_len = torch.tensor(num_imgs).to(self.opt.device)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(
            self.opt.device
        )

        text_raw = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)

        text_memory, (_, _) = self.bi_lstm_context(text_raw, text_memory_len)
        # memory = self.locationed_memory(memory, memory_len)
        aspect, (_, _) = self.bi_lstm_aspect(aspect, aspect_len)
        img_memory, (_, _) = self.bi_lstm_img(imgs, imgs_memory_len)

        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))

        et_text = aspect
        et_img = aspect

        for _ in range(self.opt.hops):
            it_al_text2text = self.attention_text(text_memory, et_text).squeeze(dim=1)
            it_al_img2text = self.attention_img2text(text_memory, et_img).squeeze(dim=1)
            it_al_text = (it_al_text2text + it_al_img2text) / 2
            # it_al_text = it_al_text2text

            it_al_img2img = self.attention_img(img_memory, et_img).squeeze(dim=1)
            it_al_text2img = self.attention_text2img(img_memory, et_text).squeeze(dim=1)
            it_al_img = (it_al_img2img + it_al_text2img) / 2
            # it_al_img = it_al_img2img

            et_text = self.gru_cell_text(it_al_text, et_text)
            et_img = self.gru_cell_img(it_al_img, et_img)
        et = torch.cat((et_text, et_img), dim=-1)
        out = self.fc(et)
        return out
