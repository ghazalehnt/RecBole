from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import RegLoss
from recbole.utils import InputType
import torch.nn as nn
import torch
from torch.nn.init import normal_

class GMF(GeneralRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(GMF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        self.embedding_dim = config['embedding_dimension']
        self.logger.info(f"embedding_dimension = {self.embedding_dim}")

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim, )
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.w = torch.nn.Linear(in_features=self.embedding_dim, out_features=1)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.apply(self._init_embedding_weights) # initialize embeddings

    def _init_embedding_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        pred = torch.mul(user_emb, item_emb) # element-wise product of these
        pred = self.w(pred) # gives m number, that summed the weighted elements of each row (dims). if w=1, then it would be simple MF
        pred = self.sigmoid(pred).squeeze()
        return pred

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output = self.forward(user, item)
        pred_loss = self.loss(output, label)
        return pred_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        output = self.forward(user, item)
        return output

