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
        self.dropout = config["dropout"]
        self.ff_layers = config["ff_layers"]
        self.mlp_variant = config["mlp_variant"]
        self.logger.info(f"embedding_dimension = {self.embedding_dim}")
        self.logger.info(f"ff_layers = {self.ff_layers}")
        self.logger.info(f"dropout = {self.dropout}")
        self.logger.info(f"mlp_variant = {self.mlp_variant}")

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)

        # there are different implementations, I chose one to input the layer sizes directly as appose to calculating them
        mlp_layers = []
        if self.mlp_variant == "cat":
            input_size = self.embedding_dim * 2
        elif self.mlp_variant == "mul":
            input_size = self.embedding_dim
        else:
            raise ValueError(f"mlp_variant = {self.mlp_variant} is not implemented")
        for i in range(len(self.ff_layers)):
            if i > 0:
                input_size = self.ff_layers[i-1]
            mlp_layers.append(nn.Dropout(p=self.dropout))
            mlp_layers.append(nn.Linear(input_size, self.ff_layers[i]))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(torch.nn.Linear(in_features=self.ff_layers[-1], out_features=1))
        mlp_layers.append(nn.Sigmoid())
        self.fc_layers = nn.Sequential(*mlp_layers)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.apply(self._init_weights) # initialize embeddings

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        if self.mlp_variant == "cat":
            pred = self.fc_layers(torch.cat([user_emb, item_emb], dim=-1))
        elif self.mlp_variant == "mul":
            pred = torch.mul(user_emb, item_emb)
            pred = self.fc_layers(pred)
        return pred.squeeze()

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

