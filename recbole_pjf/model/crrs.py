# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.model.general_recommender.bpr import BPR
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class BPR_base(BPR):
    def __init__(self, config, dataset):
        super(BPR_base, self).__init__(config, dataset)
        self.neg_user_id = self.NEG_USER_ID = config['NEG_PREFIX'] + self.USER_ID
        path = config['pretrained_path']
        self.load_pretrain_weight(path)
    
    def load_pretrain_weight(self, path):
        state_dict = torch.load(path)
        self.user_embedding.weight.data = state_dict['state_dict']['user_embedding.weight']
        self.item_embedding.weight.data = state_dict['state_dict']['item_embedding.weight']
    
    def calculate_loss(self, interaction, direct):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        neg_user = interaction[self.NEG_USER_ID]

        u_embeddings = self.user_embedding(user)
        pos_embeddings = self.item_embedding(pos_item)
        neg_embeddings = self.item_embedding(neg_item)
        neg_u_embeddings = self.user_embedding(neg_user)
        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        neg_u_scores = torch.mul(neg_u_embeddings, pos_embeddings).sum(dim=1)
        loss = self.loss(pos_scores * (direct == 1), neg_scores * (direct == 1)) \
                        + self.loss(pos_scores * (direct == 2), neg_u_scores * (direct == 2))
        
        # user_e, pos_e = self.forward(user, pos_item)
        # neg_e = self.get_item_embedding(neg_item)
        
        # pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        # loss = self.loss(pos_item_score, neg_item_score)
        # print(loss)
        return loss



class LightGCN_base(LightGCN):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCN_base, self).__init__(config, dataset)
        self.neg_user_id = self.NEG_USER_ID = config['NEG_PREFIX'] + self.USER_ID
        path = config['pretrained_path']
        self.load_pretrain_weight(path)

    def load_pretrain_weight(self, path):
        state_dict = torch.load(path)
        self.user_embedding.weight.data = state_dict['state_dict']['user_embedding.weight']
        self.item_embedding.weight.data = state_dict['state_dict']['item_embedding.weight']

    def calculate_loss(self, interaction, direct):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        neg_user = interaction[self.NEG_USER_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]
        neg_u_embeddings = user_all_embeddings[neg_user]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        neg_u_scores = torch.mul(neg_u_embeddings, pos_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores * (direct == 1), neg_scores * (direct == 1)) \
                        + self.mf_loss(pos_scores * (direct == 2), neg_u_scores * (direct == 2))

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss

        return loss


class CRRS(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(CRRS, self).__init__(config, dataset)
        self.base_model = config['base_model']
        self.DIRECT_FIELD = config["DIRECT_FIELD"]
        self.sample_users = config['sample_users']
        self.sigmoid = nn.Sigmoid()
        self.alpha = config['alpha']

        if self.base_model == 'BPR':
            self.model10 = BPR_base(config, dataset)
            self.model01 = BPR_base(config, dataset)
            self.model11 = BPR_base(config, dataset)
        elif self.base_model == 'LightGCN':
            self.model11 = LightGCN_base(config, dataset)
            self.model01 = LightGCN_base(config, dataset)
            self.model10 = LightGCN_base(config, dataset)
        else:
            raise ValueError(f"The parameter \'base_model\' should be BPR or LightGCN") 
            

    def calculate_loss(self, interaction):
        direct = interaction[self.DIRECT_FIELD] # 1 / 2
        biexposure = interaction['biexposure'] # 1 / 2
        label = interaction['label']
        mask = (label == 1)

        u_active = ((direct == 1) + (biexposure == 2)).long()
        i_active = ((direct == 2) + (biexposure == 2)).long()
        bi = (biexposure == 2).long()
        
        add_idx = 3 * (biexposure == 2) + (biexposure == 1) * (u_active + i_active * 2)

        loss = 0
        loss += self.model11.calculate_loss(interaction[(mask == 1) * (add_idx == 3)], direct[(mask == 1) * (add_idx == 3)])
        loss += self.model11.calculate_loss(interaction[(mask == 1) * (add_idx == 3)], (3 - direct)[(mask == 1) * (add_idx == 3)])
        loss += self.model01.calculate_loss(interaction[(mask == 1) * (add_idx == 1)], direct[(mask == 1) * (add_idx == 1)])
        loss += self.model10.calculate_loss(interaction[(mask == 1) * (add_idx == 2)], direct[(mask == 1) * (add_idx == 2)])

        loss += self.model01.calculate_loss(interaction[add_idx == 3], direct[add_idx == 3])
        loss += self.model01.calculate_loss(interaction[add_idx == 3], (3 - direct)[add_idx == 3])
        loss += self.model10.calculate_loss(interaction[add_idx == 3], direct[add_idx == 3])
        loss += self.model10.calculate_loss(interaction[add_idx == 3], (3 - direct)[add_idx == 3])
        
        return loss

    def get_avg_score(self, user, item):
        user_e = self.model11.user_embedding(user)
        item_e = self.model11.item_embedding(item)

        user_index = torch.randperm(self.n_users)[:self.sample_users]
        item_index = torch.randperm(self.n_items)[:self.sample_users]

        item_score_thre = torch.max(torch.matmul(item_e, self.model11.user_embedding.weight[user_index].t()), dim=1).values
        user_score_thre = torch.max(torch.matmul(user_e, self.model11.item_embedding.weight[item_index].t()), dim=1).values
 
        return self.sigmoid(user_score_thre), self.sigmoid(item_score_thre)

    def predict(self, interaction, direct = 0):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_score_thre, item_score_thre = self.get_avg_score(user, item)

        score_11 = self.sigmoid(self.model11.predict(interaction))
        score_01 = self.sigmoid(self.model01.predict(interaction)) + user_score_thre
        score_10 = self.sigmoid(self.model10.predict(interaction)) + item_score_thre

        rec_to_a = score_10 > self.alpha * score_01
        rec_to_b = self.alpha * score_10 < score_01

        rec_to_both = (score_11 > score_01) * (score_11 > score_10) 
        # rec_to_a = (score_10 > score_11) * (score_10 > score_01)
        # rec_to_b = (score_01 > score_11) * (score_01 > score_10)

        rec_to_b_score = rec_to_both * score_11 + rec_to_b * score_10
        rec_to_a_score = rec_to_both * score_11 + rec_to_a * score_01

        if direct == 0:
            direct = interaction[self.DIRECT_FIELD]
        return (direct == 1) * rec_to_a_score + (direct == 2) * rec_to_b_score
    
    