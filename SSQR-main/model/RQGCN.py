import math
import torch
from torch import nn
import dgl
from model.model_layers import CompGCNCov
import torch.nn.functional as F
from model.RQCodebook import RQCodebook


class KGGCN(nn.Module):  # GCN表示学习
    def __init__(self, num_ent, num_rel, gcn_layers, init_dim, gcn_dim, edge_type, edge_norm,
                 gcn_drop=0., opn='mult', act=None):
        super(KGGCN, self).__init__()
        self.act = act
        self.num_ent, self.num_rel = num_ent, num_rel
        self.init_dim, self.gcn_dim = init_dim, gcn_dim
        self.gcn_drop = gcn_drop
        self.opn = opn
        self.edge_type = edge_type  # [E]
        self.edge_norm = edge_norm  # [E]

        self.init_embed = self.get_param([self.num_ent, self.init_dim])  # initial embedding for entities
        self.init_rel = self.get_param([self.num_rel * 2, self.init_dim])  # relation

        layers = []
        for i in range(gcn_layers):
            if i == 0:
                layer = CompGCNCov(self.init_dim, self.gcn_dim, self.act, gcn_drop, opn)
            else:
                layer = CompGCNCov(self.gcn_dim, self.gcn_dim, self.act, gcn_drop, opn)
            layers.append(layer)
        self.gcn_layers = nn.ModuleList(layers)
        self.drop = torch.nn.Dropout(self.gcn_drop)

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def forward(self, g, subj, rel):
        x, r = self.init_embed, self.init_rel  # embedding of relations
        for layer in self.gcn_layers:
            x, r = layer(g, x, r, self.edge_type, self.edge_norm)
            x = self.drop(x)
            r = self.drop(r)
        sub_emb = torch.index_select(x, 0, subj)  # filter out embeddings of subjects in this batch
        rel_emb = torch.index_select(r, 0, rel)  # filter out embeddings of relations in this batch
        return sub_emb, rel_emb, x

    def cal_ent_embeds(self, g):
        x, r = self.init_embed, self.init_rel  # embedding of relations
        for layer in self.gcn_layers:
            x, r = layer(g, x, r, self.edge_type, self.edge_norm)
            x = self.drop(x)
            r = self.drop(r)
        return x


class RQGCN(nn.Module):
    def __init__(self, num_ent, num_rel, gcn_layers, tf_layers, init_dim, gcn_dim, edge_type, edge_norm,
                 gcn_drop=0., act='tanh', opn='mult', seq_len=16, num_code=1024, num_quantizers=4, att_head=2):
        super(RQGCN, self).__init__()
        if act == 'tanh':
            self.act = torch.tanh
        elif act == 'relu':
            self.act = torch.relu
        self.seq_len = seq_len
        self.gcn_dim = gcn_dim
        self.num_quantizers = num_quantizers
        
        self.kggcn = KGGCN(num_ent, num_rel, gcn_layers, init_dim, gcn_dim, edge_type, edge_norm, gcn_drop, opn, self.act)
        self.encoder = TransformerBlock(gcn_dim, num_heads=att_head, transformer_layers=tf_layers)
        self.entagg = TransformerBlock(gcn_dim, num_heads=att_head, transformer_layers=tf_layers)
        self.position_enc = PositionalEncoding(gcn_dim)
        
        # 使用RQ-VAE替代VQ-VAE
        self.rq_codebook = RQCodebook(num_codes=num_code, latent_dim=gcn_dim, 
                                    num_quantizers=num_quantizers)

        self.bias = nn.Parameter(torch.zeros(num_ent))  # 预测结果时实体bias
        self.cls_emb_head = self.get_param([1, gcn_dim])  # CLS初始化 1, D
        self.loss = nn.BCELoss()
        self.drop = nn.Dropout(gcn_drop)

        # convE
        self.conv_drop = nn.Dropout(gcn_drop)
        self.line_conv1 = nn.Linear(2*gcn_dim, 400)
        self.line_conv2 = nn.Linear(200*14*14, gcn_dim)
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=200,
                                      kernel_size=(7, 7), stride=1, padding=0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)  # one channel, do bn on initial embedding
        self.bn1 = torch.nn.BatchNorm2d(200)  # do bn on output of conv
        self.bn2 = torch.nn.BatchNorm1d(gcn_dim)

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def concat(self, ent_embed, rel_embed):
        ent_embed = ent_embed.view(-1, 1, self.gcn_dim)
        rel_embed = rel_embed.view(-1, 1, self.gcn_dim)
        stack_input = torch.cat([ent_embed, rel_embed], 1)  # [batch_size, 2, embed_dim]
        stack_input = stack_input.reshape(-1, 1, 20, 20)  # reshape to 2D [batch, 1, 2*k_h, k_w]
        return stack_input

    def forward(self, g, subj, rel, ent_text_emd=None, stage=1):
        sub_emb, rel_emb, all_ent = self.kggcn(g, subj, rel)  # B D, B D, M D

        batch_size = sub_emb.size(0)

        sub_exp = sub_emb.unsqueeze(1).repeat(1, self.seq_len-1, 1)  # B L-1 D

        # 添加一个CLS token
        cls_emb_head = self.cls_emb_head.unsqueeze(0).repeat(batch_size, 1, 1)  # B 1 D
        sub_trans = torch.cat([cls_emb_head, sub_exp], dim=1)  # B L D
        pos_emds = self.position_enc(sub_trans)  # 1 L D
        sub_trans = sub_trans + pos_emds
        sub_trans = self.encoder(sub_trans)  # 实体表示  之后进行RQ
        sub_trans = self.drop(self.act(sub_trans))

        # 使用RQ-VAE进行残差量化
        sub_trans_q, all_codebook_indices, rq_loss = self.rq_codebook(sub_trans)

        sub_trans = self.entagg(sub_trans)  # 重新使用
        sub_trans = self.drop(self.act(sub_trans))
        sub_final = sub_trans[:, 0, :]  # B D 最终实体表示

        # convE
        subrel = self.concat(sub_final, rel_emb)
        x = self.bn0(subrel)
        x = self.conv2d(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv_drop(x)
        x = x.view(batch_size, -1)
        x = self.line_conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.conv_drop(x)
        x = torch.mm(x, all_ent.transpose(1, 0))  # [batch_size, ent_num]

        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)

        return score, rq_loss, rq_loss

    def cal_allent_codes(self, g):
        """生成所有实体的RQ代码"""
        all_ent = self.kggcn.cal_ent_embeds(g)  # M D
        sub_exp = all_ent.unsqueeze(1).repeat(1, self.seq_len, 1)  # M L D

        sub_trans = sub_exp
        pos_emds = self.position_enc(sub_trans)  # 1 L D
        sub_trans = sub_trans + pos_emds
        sub_trans = self.encoder(sub_trans)  # 实体表示  之后进行RQ
        
        # 生成所有量化器的代码索引
        all_codebook_indices = self.rq_codebook.cal_codes(sub_trans)  # M L num_quantizers
        return all_codebook_indices

    def cal_allent_codes_emds(self, g):
        """生成所有实体的RQ嵌入表示"""
        all_ent = self.kggcn.cal_ent_embeds(g)  # M D
        sub_exp = all_ent.unsqueeze(1).repeat(1, self.seq_len, 1)  # M L D

        sub_trans = sub_exp
        pos_emds = self.position_enc(sub_trans)  # 1 L D
        sub_trans = sub_trans + pos_emds
        sub_trans = self.encoder(sub_trans)  # 实体表示  之后进行RQ
        
        # 生成量化后的嵌入表示
        codes_emds = self.rq_codebook.cal_code_emds(sub_trans)  # M L D
        return codes_emds


class TransformerBlock(nn.Module):  # decoder聚合结果
    def __init__(self, embed_dim, num_heads=4, transformer_layers=2):
        super(TransformerBlock, self).__init__()
        layers = []
        for i in range(transformer_layers):
            layers.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):  # B L D
        for layer in self.layers:
            x, attn_weights = layer(query=x, key=x, value=x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=1024):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = nn.Parameter(pe, requires_grad=False)
        self.register_buffer('pe', pe)

    def forward(self, x):
        L = x.size(1)
        emds = self.pe[:L, :].unsqueeze(0)  # 1 L D
        return emds
