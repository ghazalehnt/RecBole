from torch.autograd import profiler

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import SoftCrossEntropyLoss#, HierarchicalSoftmax
from recbole.utils import InputType
import torch.nn as nn
import torch
from torch.nn.init import normal_
import gensim
import gensim.downloader as api
import os
import time


class JOINTSRMFFULL(GeneralRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(JOINTSRMFFULL, self).__init__(config, dataset)
        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        self.embedding_dim = config['embedding_dimension']
        self.alpha = config["alpha"]
        item_description_fields = config['item_description_fields']
        if "number_of_reviews_to_use" in config:
            max_number_of_reviews = config['number_of_reviews_to_use']
        else:
            max_number_of_reviews = 1

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
        weights = torch.FloatTensor(model.vectors)  # formerly syn0, which is soon deprecated
        self.logger.info(f"pretrained_embedding shape: {weights.shape}")
        self.word_embedding = nn.Embedding.from_pretrained(weights, freeze=True)
        self.vocab_size = len(model.key_to_index)

        s = time.time()
        self.lm_gt = torch.zeros((self.n_items, self.vocab_size), dtype=torch.uint8)
        self.lm_gt_len = torch.ones(self.n_items, dtype=torch.int16)
        item_desc_fields = []
        if "item_description" in item_description_fields:
            item_desc_fields.append(3)
        if "item_genres" in item_description_fields:
            item_desc_fields.append(4)
        if "tags" in item_description_fields:
            item_desc_fields.append(4)
        if len(item_desc_fields) > 0:
            item_LM_file = os.path.join(dataset.dataset.dataset_path, f"{dataset.dataset.dataset_name}.item")
            with open(item_LM_file, 'r') as infile:
                next(infile)
                for line in infile:
                    split = line.split("\t")
                    item_id = dataset.token2id_exists("item_id", split[0])
                    if item_id == -1:
                        continue
                    if item_id == 0:
                        print("Isnt that padding?")
                    for fi in item_desc_fields:
                        desc = split[fi]
                        for term in desc.split():
                            if term in model.key_to_index:
                                wv_term_index = model.key_to_index[term]
                                self.lm_gt[item_id][wv_term_index] += 1
                                self.lm_gt_len[item_id] += 1
        if "review" in item_description_fields:
            num_of_used_revs = {}
            item_desc_fields = [3]
            item_LM_file = os.path.join(dataset.dataset.dataset_path, f"{dataset.dataset.dataset_name}.inter")
            with open(item_LM_file, 'r') as infile:
                next(infile)
                for line in infile:
                    split = line.split("\t")
                    item_id = dataset.token2id_exists("item_id", split[1])
                    if item_id == -1:
                        continue
                    if item_id == 0:
                        print("Isnt that padding?")
                    if item_id not in num_of_used_revs:
                        num_of_used_revs[item_id] = 0
                    elif num_of_used_revs[item_id] >= max_number_of_reviews:
                        continue
                    for fi in item_desc_fields:
                        desc = split[fi]
                        if len(desc.split()) > 0:
                            num_of_used_revs[item_id] += 1
                        for term in desc.split():
                            if term in model.key_to_index:
                                wv_term_index = model.key_to_index[term]
                                if term in model.key_to_index:
                                    wv_term_index = model.key_to_index[term]
                                    self.lm_gt[item_id][wv_term_index] += 1
                                    self.lm_gt_len[item_id] += 1
        self.lm_gt_len[(self.lm_gt_len == 0).nonzero(as_tuple=True)] = 1
        e = time.time()
        self.logger.info(f"{e - s}s")
        self.logger.info(f"Done with lm_gt construction!")

        self.sigmoid = nn.Sigmoid()
        self.loss_rec = nn.BCELoss()
        self.loss_lm = SoftCrossEntropyLoss()

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

        with profiler.record_function("REC output and loss"):
            output_rec = self.forward_rec(user, item)
            loss_rec = self.loss_rec(output_rec, label)

        with profiler.record_function("LM output"):
            output_lm = self.forward_lm(item)

        if self.variant == 1:
            with profiler.record_function("LM making tensor on GPU"):
                label_lm_k = self.lm_gt[item].to(device=self.device)
                label_lm_len = self.lm_gt_len[item].to(device=self.device)
                label_lm = (label_lm_k.T / label_lm_len).T
        elif self.variant == 2:
            with profiler.record_function("LM making tensor on CPU"):
                label_lm_k = self.lm_gt[item]
                label_lm_len = self.lm_gt_len[item]
                label_lm_k = label_lm_k.T / label_lm_len
                label_lm = label_lm_k.to(device=self.device).T

        with profiler.record_function("LM loss"):
            loss_lm = self.loss_lm(output_lm, label_lm)

        return loss_rec, self.alpha * loss_lm

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        output = self.forward_rec(user, item)
        return output
