import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()

        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim # 임베딩 차원
        self.n_heads = n_heads # 헤드(head)의 개수: 서로 다른 어텐션(attention) 컨셉의 수
        self.head_dim = hidden_dim // n_heads # 각 헤드(head)에서의 임베딩 차원

        self.fc_q = nn.Linear(hidden_dim, hidden_dim) # Query 값에 적용될 FC 레이어
        self.fc_k = nn.Linear(hidden_dim, hidden_dim) # Key 값에 적용될 FC 레이어
        self.fc_q_s = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k_s = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim) # Value 값에 적용될 FC 레이어

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, query_side, key_side, value, mask = None):

        batch_size = query.shape[0]

        # query: [batch_size, query_len, hidden_dim]
        # key: [batch_size, key_len, hidden_dim]
        # value: [batch_size, value_len, hidden_dim]
 
        Q = self.fc_q(query)
        Q_s = self.fc_q_s(query_side)
        K = self.fc_k(key)
        K_s = self.fc_k_s(key_side)
        V = self.fc_v(value)

        # Q: [batch_size, query_len, hidden_dim]
        # K: [batch_size, key_len, hidden_dim]
        # V: [batch_size, value_len, hidden_dim]

        # hidden_dim → n_heads X head_dim 형태로 변형
        # n_heads(h)개의 서로 다른 어텐션(attention) 컨셉을 학습하도록 유도
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        Q_s = Q_s.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K_s = K_s.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q: [batch_size, n_heads, query_len, head_dim]
        # K: [batch_size, n_heads, key_len, head_dim]
        # V: [batch_size, n_heads, value_len, head_dim]

        # Attention Energy 계산
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        energy_side = torch.matmul(Q_s, K_s.permute(0, 1, 3, 2)) / self.scale

        # energy: [batch_size, n_heads, query_len, key_len]

        # 마스크(mask)를 사용하는 경우
        if mask is not None:
            # 마스크(mask) 값이 0인 부분을 -1e10으로 채우기
            #energy = energy.masked_fill(mask==0, -1e10)
            energy = energy.masked_fill(mask, 0) + energy_side.masked_fill(mask==0,0)

            
        
        # 어텐션(attention) 스코어 계산: 각 단어에 대한 확률 값
        attention = torch.softmax(energy, dim=-1)

        # attention: [batch_size, n_heads, query_len, key_len]

        # 여기에서 Scaled Dot-Product Attention을 계산
        x = torch.matmul(self.dropout(attention), V)

        # x: [batch_size, n_heads, query_len, head_dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x: [batch_size, query_len, n_heads, head_dim]

        x = x.view(batch_size, -1, self.hidden_dim)

        # x: [batch_size, query_len, hidden_dim]

        x = self.fc_o(x)

        # x: [batch_size, query_len, hidden_dim]

        return x, attention

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num,side1num,side2num,side3num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.side1num = side1num
        self.side2num = side2num
        self.side3num = side3num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.side1_emb = torch.nn.Embedding(self.side1num+1, args.hidden_units, padding_idx=0)
        self.side2_emb = torch.nn.Embedding(self.side2num+1, args.hidden_units, padding_idx=0)
        self.side3_emb = torch.nn.Embedding(self.side3num+1, args.hidden_units, padding_idx=0)

        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layernorms_side = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layernorms_side = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.forward_layers_side = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            self.attention_layernorms_side.append(new_attn_layernorm)

            # 이 부분을 변경
            # new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
            #                                                 args.num_heads,
            #                                                 args.dropout_rate)
            new_attn_layer = MultiHeadAttentionLayer(args.hidden_units, args.num_heads, args.dropout_rate,self.dev)


            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            self.forward_layernorms_side.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
            self.forward_layers_side.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs, side1_seqs, side2_seqs, side3_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5

        seqs_s1 = self.side1_emb(torch.LongTensor(side1_seqs).to(self.dev))
        seqs_s1 *= self.side1_emb.embedding_dim ** 0.5

        seqs_s2 = self.side2_emb(torch.LongTensor(side2_seqs).to(self.dev))
        seqs_s2 *= self.side2_emb.embedding_dim ** 0.5

        seqs_s3 = self.side3_emb(torch.LongTensor(side3_seqs).to(self.dev))
        seqs_s3 *= self.side3_emb.embedding_dim ** 0.5


        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        
        seqs = seqs + seqs_s1 + seqs_s2 + seqs_s3
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
        ######
        seqs_side = seqs_s1 + seqs_s2 + seqs_s3
        seqs_side += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs_side = self.emb_dropout(seqs_side)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim
        seqs_side *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        for i in range(len(self.attention_layers)):
            # seqs = torch.transpose(seqs, 0, 1)
            # seqs_side = torch.transpose(seqs_side, 0, 1)
            
            Q = self.attention_layernorms[i](seqs)
            Q_side = self.attention_layernorms_side[i](seqs_side)

            
            mha_outputs, _ = self.attention_layers[i](Q, seqs, Q_side, seqs_side, seqs,
                                            mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs_side = Q_side + mha_outputs
            # seqs = torch.transpose(seqs, 0, 1)
            # seqs_side = torch.transpose(seqs_side, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs_side = self.forward_layernorms_side[i](seqs_side)

            seqs = self.forward_layers[i](seqs)
            seqs_side = self.forward_layers_side[i](seqs_side)

            seqs *=  ~timeline_mask.unsqueeze(-1)
            seqs_side *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, log_seqs_side1, pos_seq_side1, neg_seqs_side1, 
    log_seqs_side2, pos_seqs_side2, neg_seqs_side2, log_seqs_side3, pos_seqs_side3, neg_seqs_side3): # for training     
        log_feats = self.log2feats(log_seqs, log_seqs_side1, log_seqs_side2, log_seqs_side3) # user_ids hasn't been used yet

        
        # import pickle
        # with open('user.pickle','wb') as fw:
        #     pickle.dump(a, fw)

        pos_embs = (self.item_emb(torch.LongTensor(pos_seqs).to(self.dev)) + 
        self.side1_emb(torch.LongTensor(pos_seq_side1).to(self.dev)) + 
        self.side2_emb(torch.LongTensor(pos_seqs_side2).to(self.dev)) + 
        self.side3_emb(torch.LongTensor(pos_seqs_side3).to(self.dev)))      

        neg_embs = (self.item_emb(torch.LongTensor(neg_seqs).to(self.dev)) + 
        self.side1_emb(torch.LongTensor(neg_seqs_side1).to(self.dev))+ 
        self.side2_emb(torch.LongTensor(neg_seqs_side2).to(self.dev)) + 
        self.side3_emb(torch.LongTensor(neg_seqs_side3).to(self.dev)))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)



        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices
    , log_seqs_side1, item_indices_side1
    , log_seqs_side2, item_indices_side2
    , log_seqs_side3, item_indices_side3): # for inference
        
        log_feats = self.log2feats(log_seqs, log_seqs_side1, log_seqs_side2, log_seqs_side3)

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = (self.item_emb(torch.LongTensor(item_indices).to(self.dev))+
        self.side1_emb(torch.LongTensor(item_indices_side1).to(self.dev))+
        self.side2_emb(torch.LongTensor(item_indices_side2).to(self.dev))+

        self.side3_emb(torch.LongTensor(item_indices_side3).to(self.dev))) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        import pickle

        return logits # preds # (U, I)
