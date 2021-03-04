from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import RegLoss
from recbole.utils import InputType
import torch.nn as nn
import torch
from torch.nn.init import normal_

class MFSimple(GeneralRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(MFSimple, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        self.embedding_dim = config['embedding_dimension']
        self.reg_weight = config['reg_weight']
        self.logger.info(f"embedding_dimension = {self.embedding_dim}")
        self.logger.info(f"reg_weight = {self.reg_weight}")

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim, )
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.user_bias = nn.Parameter(torch.zeros(self.n_users))
        self.item_bias = nn.Parameter(torch.zeros(self.n_items))
        self.bias = nn.Parameter(torch.zeros(1))

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.reg_loss = RegLoss()
        
        self.apply(self._init_weights) # initialize embeddings

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        pred = torch.sum(torch.mul(user_emb, item_emb).squeeze(), dim=1) # or torch.diag(torch.matmul(user_emb, item_emb.T))
        pred = pred + self.item_bias[item] + self.user_bias[user]
        pred = pred + self.bias
        pred = self.sigmoid(pred)
        
        reg = self.reg_loss([self.user_embedding.weight, self.user_bias, self.item_embedding.weight, self.item_bias, self.bias])
        return pred, reg 

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output, reg = self.forward(user, item)
        loss = self.loss(output, label) + (self.reg_weight * reg)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        output, reg = self.forward(user, item)
        return output

