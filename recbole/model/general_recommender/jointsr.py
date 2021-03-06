from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import SoftCrossEntropyLoss#, HierarchicalSoftmax
from recbole.utils import InputType
import torch.nn as nn
import torch
from torch.nn.init import normal_
import gensim
import gensim.downloader as api
import os


class JOINTSR(GeneralRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(JOINTSR, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        self.embedding_dim = config['embedding_dimension']
        self.dropout = config["dropout"]
        self.ff_layers = config["ff_layers"]
        self.alpha = config["alpha"]
        item_description_fields = config['item_description_fields']

        self.logger.info(f"embedding_dimension = {self.embedding_dim}")
        self.logger.info(f"ff_layers = {self.ff_layers}")
        self.logger.info(f"dropout = {self.dropout}")
        self.logger.info(f"alpha = {self.alpha}")
        self.logger.info(f"item_description_fields = {item_description_fields}")

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
        mlp_layers.append(torch.nn.Linear(in_features=self.ff_layers[-1], out_features=1))
        mlp_layers.append(nn.Sigmoid())
        self.fc_layers = nn.Sequential(*mlp_layers)

        gensim_cache = open('gensim_cache_path', 'r').read().strip()
        os.environ['GENSIM_DATA_DIR'] = str(gensim_cache)
        # pretrained_embedding_name = "conceptnet-numberbatch-17-06-300"
        pretrained_embedding_name = "glove-wiki-gigaword-50" # because the size must be 50 the same as the embedding
        model_path = api.load(pretrained_embedding_name, return_path=True)
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path)
        weights = torch.FloatTensor(model.vectors)  # formerly syn0, which is soon deprecated
        self.logger.info(f"pretrained_embedding shape: {weights.shape}")
        self.word_embedding = nn.Embedding.from_pretrained(weights, freeze=True)

        # getting the lms:
        # TODO: this should be changed if we could load fields from other atomic files as well
        item_features = dataset.get_item_feature()
        self.lm_gt_keys = [[] for i in range(len(item_features))]
        self.lm_gt_values = [[] for i in range(len(item_features))]
        for item_description_field in item_description_fields:
            item_descriptions = item_features[item_description_field]  # [0] is PAD
            for i in range(1, len(item_descriptions)):
                for termid in item_descriptions[i]:
                    if termid > 0: # termid=0 is reserved for padding
                        term = dataset.id2token(item_description_field, termid)
                        term = str(term)
                        term = term.lower()
                        if model.vocab.__contains__(term):
                            wv_term_index = model.vocab.get(term).index
                            if wv_term_index not in self.lm_gt_keys[i]:
                                self.lm_gt_keys[i].append(wv_term_index)
                                self.lm_gt_values[i].append(1)
                            else:
                                idx = self.lm_gt_keys[i].index(wv_term_index)
                                self.lm_gt_values[i][idx] += 1
        self.logger.info(f"Done with lm_gt construction!")

        self.loss_rec = nn.BCELoss()
        self.loss_lm = SoftCrossEntropyLoss()
        # self.loss_lm_gs = HierarchicalSoftmax(len(model.vocab), self.embedding_dim)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    @staticmethod
    def get_entries(array, keys):
        ret = []
        for k in keys:
            ret.append(array[k])
        return ret

    def forward_rec(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        pred = torch.mul(user_emb, item_emb)
        pred = self.fc_layers(pred)
        return pred.squeeze()

    def forward_lm(self, item):
        item_emb = self.item_embedding(item)
        pred = torch.matmul(item_emb, self.word_embedding.weight.T)
        return pred.squeeze()

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output_rec = self.forward_rec(user, item)
        loss_rec = self.loss_rec(output_rec, label)

        output_lm = self.forward_lm(item) # output should be unnormalized counts
        item_term_keys = self.get_entries(self.lm_gt_keys, item)
        item_term_vals = self.get_entries(self.lm_gt_values, item)
        label_lm = torch.zeros(len(item), output_lm.shape[1], device=self.device)
        for i in range(len(item_term_keys)):
            item_desc_len = 0
            for j in range(len(item_term_keys[i])):
                k = item_term_keys[i][j]
                v = item_term_vals[i][j]
                label_lm[i][k] = v
                item_desc_len += v
            if item_desc_len > 0:
                label_lm[i] /= item_desc_len  # labels should be probability distribution
        loss_lm = self.loss_lm(output_lm, label_lm)

        # output_lm_hs, loss_lm_hs = self.loss_lm_gs(self.item_embedding(item), label_lm) # it doesn;t really make sense, we are not using the w2v like this!

        # return loss_rec, loss_lm, loss_lm_hs
        return loss_rec, self.alpha * loss_lm

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        output = self.forward_rec(user, item)
        return output

