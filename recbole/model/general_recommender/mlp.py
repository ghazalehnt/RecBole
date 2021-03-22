from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import RegLoss
from recbole.utils import InputType
import torch.nn as nn
import torch
from torch.nn.init import normal_

class MLP(GeneralRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(MLP, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        self.embedding_dim = config['embedding_dimension']
        # self.reg_weight = config['reg_weight'] # TODO: either remove or we have to implement the regloss function as the RegLoss here does not work for Linear
        # self.dropout = config["dropout"] #TODO do we want dropout?
        self.n_layers = config["mlp_n_layers"]
        self.logger.info(f"embedding_dimension = {self.embedding_dim}")
        self.logger.info(f"mlp_n_layers = {self.n_layers}")

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)

        mlp_layers = []
        input_size = self.embedding_dim * 2
        for i in range(0, self.n_layers):
            # mlp_layers.append(nn.Dropout(p=self.dropout)) # here?
            mlp_layers.append(nn.Linear(input_size, input_size // 2))
            mlp_layers.append(nn.ReLU())
            input_size = input_size // 2
        self.fc_layers = nn.Sequential(*mlp_layers)
        self.affine_output = torch.nn.Linear(in_features=input_size, out_features=1)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.apply(self._init_weights) # initialize embeddings

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        pred = self.fc_layers(torch.cat([user_emb, item_emb], dim=-1))
        pred = self.affine_output(pred)
        pred = self.sigmoid(pred).squeeze()
        return pred

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output = self.forward(user, item)
        loss = self.loss(output, label)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        output = self.forward(user, item)
        return output

