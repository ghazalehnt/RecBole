from torch.nn.init import normal_

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType
import torch.nn as nn
import torch

# This is the same as NeuMF, but in my own (GH) implementation.
# this implementation does not implement loading pretrained embeddings now.
class NCF(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NCF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        self.gmf_embedding_dim = config['gmf_embedding_dimension']
        self.mlp_embedding_dim = config['mlp_embedding_dimension']
        self.mlp_n_layers = config["mlp_n_layers"]
        self.logger.info(f"gmf_embedding_dimension = {self.gmf_embedding_dim}")
        self.logger.info(f"mlp_embedding_dimension = {self.mlp_embedding_dim}")
        self.logger.info(f"mlp_n_layers = {self.mlp_n_layers}")

        self.mlp_user_embedding = nn.Embedding(self.n_users, self.mlp_embedding_dim)
        self.mlp_item_embedding = nn.Embedding(self.n_items, self.mlp_embedding_dim)
        mlp_layers = []
        input_size = self.mlp_embedding_dim * 2
        for i in range(0, self.mlp_n_layers):
            mlp_layers.append(nn.Linear(input_size, input_size // 2))
            mlp_layers.append(nn.ReLU())
            input_size = input_size // 2
        self.mlp_fc_layers = nn.Sequential(*mlp_layers)

        self.gmf_user_embedding = nn.Embedding(self.n_users, self.gmf_embedding_dim)
        self.gmf_item_embedding = nn.Embedding(self.n_items, self.gmf_embedding_dim)

        self.affine_output = torch.nn.Linear(in_features=input_size+self.gmf_embedding_dim, out_features=1)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.apply(self._init_weights)  # initialize embeddings

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item):
        mlp_user_emb = self.mlp_user_embedding(user)
        mlp_item_emb = self.mlp_item_embedding(item)
        mlp_output = self.mlp_fc_layers(torch.cat([mlp_user_emb, mlp_item_emb], dim=-1))

        gmf_user_emb = self.gmf_user_embedding(user)
        gmf_item_emb = self.gmf_item_embedding(user)
        gmf_item_user_mul = torch.mul(gmf_user_emb, gmf_item_emb)

        pred = self.affine_output(torch.cat((mlp_output, gmf_item_user_mul), dim=1))
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

