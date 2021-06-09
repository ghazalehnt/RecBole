import time

from torch.nn import AdaptiveLogSoftmaxWithLoss

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import SoftCrossEntropyLoss, SoftAdaptiveSoftmaxWithLoss  # , HierarchicalSoftmax
from recbole.utils import InputType
import torch.nn as nn
import torch
from torch.nn.init import normal_
import gensim
import gensim.downloader as api
import os

class JOINTSRMFADAP(GeneralRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(JOINTSRMFADAP, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        self.embedding_dim = config['embedding_dimension']
        self.alpha = config["alpha"]
        item_description_fields = config['item_description_fields']

        self.logger.info(f"embedding_dimension = {self.embedding_dim}")
        self.logger.info(f"alpha = {self.alpha}")
        self.logger.info(f"item_description_fields = {item_description_fields}")

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.user_bias = nn.Parameter(torch.zeros(self.n_users))
        self.item_bias = nn.Parameter(torch.zeros(self.n_items))
        self.bias = nn.Parameter(torch.zeros(1))
        self.apply(self._init_weights)

        gensim_cache = open('gensim_cache_path', 'r').read().strip()
        os.environ['GENSIM_DATA_DIR'] = str(gensim_cache)
        # pretrained_embedding_name = "conceptnet-numberbatch-17-06-300"
        pretrained_embedding_name = "glove-wiki-gigaword-50" # because the size must be 50 the same as the embedding
        model_path = api.load(pretrained_embedding_name, return_path=True)
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path)
        self.vocab_size = len(model.key_to_index)
        weights = torch.FloatTensor(model.vectors)  # formerly syn0, which is soon deprecated
        self.logger.info(f"pretrained_embedding shape: {weights.shape}")
        self.word_embedding = nn.Embedding.from_pretrained(weights, freeze=True)

        self.lm_gt_keys = [[] for i in range(self.n_items)]
        self.lm_gt_values = [[] for i in range(self.n_items)]
        item_LM_file = os.path.join(dataset.dataset.dataset_path, f"{dataset.dataset.dataset_name}.item")
        item_desc_fields = []
        if "item_description" in item_description_fields:
            item_desc_fields.append(3)
        if "item_genres" in item_description_fields:
            item_desc_fields.append(4)
        # TODO other fields? e.g. review? have to write another piece of code
        with open(item_LM_file, 'r') as infile:
            next(infile)
            for line in infile:
                split = line.split("\t")
                item_id = dataset.token2id_exists("item_id", split[0])
                if item_id == -1:
                    continue
                for fi in item_desc_fields:
                    desc = split[fi]
                    for term in desc.split():
                        if term in model.key_to_index:
                            wv_term_index = model.key_to_index[term]
                            if wv_term_index not in self.lm_gt_keys[item_id]:
                                self.lm_gt_keys[item_id].append(wv_term_index)
                                self.lm_gt_values[item_id].append(1)
                            else:
                                idx = self.lm_gt_keys[item_id].index(wv_term_index)
                                self.lm_gt_values[item_id][idx] += 1
        self.logger.info(f"Done with lm_gt construction!")

        self.sigmoid = nn.Sigmoid()
        self.loss_rec = nn.BCELoss()
        self.loss_lm = SoftAdaptiveSoftmaxWithLoss(self.embedding_dim, self.vocab_size)

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
        pred = torch.sum(torch.mul(user_emb, item_emb).squeeze(), dim=1)
        pred = pred + self.item_bias[item] + self.user_bias[user]
        pred = pred + self.bias
        pred = self.sigmoid(pred)
        return pred

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

        s = time.time()
        output_lm = self.forward_lm(item) # output should be unnormalized counts
        item_term_keys = self.get_entries(self.lm_gt_keys, item)
        item_term_vals = self.get_entries(self.lm_gt_values, item)
        # when using the softmax loss, we need to have probability distrubution for labels:
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
        e = time.time()
        self.logger.info(f"{e - s}s lm_output and loss_lm")

        # # when using negative sampling loss:
        # loss_lm = self.loss_lm(output_lm, item_term_keys, item_term_vals)

        return loss_rec, self.alpha * loss_lm

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        output = self.forward_rec(user, item)
        return output

