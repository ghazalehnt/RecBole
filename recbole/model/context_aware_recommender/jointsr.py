from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType
import torch.nn as nn
import torch
from torch.nn.init import normal_
import gensim
import gensim.downloader as api
import os


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

class JOINTSR(GeneralRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(JOINTSR, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        self.embedding_dim = config['embedding_dimension']
        self.dropout = config["dropout"]
        self.ff_layers = config["ff_layers"]
        self.logger.info(f"embedding_dimension = {self.embedding_dim}")
        self.logger.info(f"ff_layers = {self.ff_layers}")
        self.logger.info(f"dropout = {self.dropout}")

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.apply(self._init_weights)

        # I saw different implementations of this!
        mlp_layers = []
        input_size = self.embedding_dim
        for i in range(len(self.ff_layers)):
            if i > 0:
                input_size = self.ff_layers[i-1]
            mlp_layers.append(nn.Dropout(p=self.dropout))
            mlp_layers.append(nn.Linear(input_size, self.ff_layers[i]))
            mlp_layers.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*mlp_layers)
        self.affine_output = torch.nn.Linear(in_features=input_size, out_features=1)

        gensim_cache = ""
        os.environ['GENSIM_DATA_DIR'] = str(gensim_cache)
        # pretrained_embedding_name = "conceptnet-numberbatch-17-06-300"
        pretrained_embedding_name = "glove-twitter-50" # because the size must be 50 the same as the embedding
        model_path = api.load(pretrained_embedding_name, return_path=True)
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path)
        weights = torch.FloatTensor(model.vectors)  # formerly syn0, which is soon deprecated
        self.logger.info(f"pretrained_embedding shape: {weights.shape}")
        self.word_embedding = nn.Embedding.from_pretrained(weights)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.loss_rec = nn.BCELoss()
        self.loss_ml = cross_entropy

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward_rec(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        pred = torch.mul(user_emb, item_emb)
        pred = self.fc_layers(pred)
        pred = self.affine_output(pred)
        pred = self.sigmoid(pred).squeeze()
        return pred

    def forward_ml(self, item):
        item_emb = self.item_embedding(item)
        pred = torch.matmul(item_emb, self.word_embedding.weight.T)
        pred = self.softmax(pred).squeeze()
        return pred


    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output_rec = self.forward_rec(user, item)
        loss_rec = self.loss_rec(output_rec, label)

        output_ml = self.forward_ml(item)
        fake = torch.rand(output_ml.shape)
        label_lm = torch.softmax(fake, dim=1)  # get the language model for itam!!!!
        loss_ml = self.loss_ml(output_ml, label_lm)
        # loss = loss_rec + loss_ml
        # return loss
        return loss_rec, loss_ml

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        output = self.forward(user, item)
        return output

