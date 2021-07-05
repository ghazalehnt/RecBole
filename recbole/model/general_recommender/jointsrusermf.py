from torch_sparse import SparseTensor

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import SoftCrossEntropyLoss#, HierarchicalSoftmax
from recbole.utils import InputType
import torch.nn as nn
import torch
from torch.nn.init import normal_
import os
import time
import numpy as np
import torch.autograd.profiler as profiler

class JOINTSRUSERMF(GeneralRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(JOINTSRUSERMF, self).__init__(config, dataset)
        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        self.embedding_dim = config['embedding_dimension']
        self.alpha1 = config["alpha_item"]
        self.alpha2 = config["alpha_user"]
        item_description_fields = config['item_description_fields']
        user_description_fields = config['user_description_fields']
        max_number_of_reviews_item = config['number_of_reviews_to_use_item']
        max_number_of_reviews_user = config['number_of_reviews_to_use_user']
        self.variant = config["debug_variant"]

        self.logger.info(f"embedding_dimension = {self.embedding_dim}")
        self.logger.info(f"alpha item = {self.alpha1}")
        self.logger.info(f"alpha user = {self.alpha2}")
        self.logger.info(f"item_description_fields = {item_description_fields}")
        self.logger.info(f"user_description_fields = {user_description_fields}")

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.user_bias = nn.Parameter(torch.zeros(self.n_users))
        self.item_bias = nn.Parameter(torch.zeros(self.n_items))
        self.bias = nn.Parameter(torch.zeros(1))
        self.apply(self._init_weights)

        gensim_cache = open('gensim_cache_path', 'r').read().strip()
        os.environ['GENSIM_DATA_DIR'] = str(gensim_cache)
        import gensim
        import gensim.downloader as api
        # pretrained_embedding_name = "conceptnet-numberbatch-17-06-300"
        pretrained_embedding_name = "glove-wiki-gigaword-50" # because the size must be 50 the same as the embedding
        model_path = api.load(pretrained_embedding_name, return_path=True)
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path)
        weights = torch.FloatTensor(model.vectors)  # formerly syn0, which is soon deprecated
        self.logger.info(f"pretrained_embedding shape: {weights.shape}")
        self.word_embedding = nn.Embedding.from_pretrained(weights, freeze=True)
        self.vocab_size = len(model.key_to_index)

        s = time.time()
        item_lms = {}
        item_lm_len = {}
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
                    if item_id not in item_lms:
                        item_lms[item_id] = {}
                        item_lm_len[item_id] = 0
                    for fi in item_desc_fields:
                        if fi >= len(split):
                            print(split)
                            continue
                        desc = split[fi]
                        for term in desc.split():
                            if term in model.key_to_index:
                                wv_term_index = model.key_to_index[term]
                                if wv_term_index not in item_lms[item_id]:
                                    item_lms[item_id][wv_term_index] = 1
                                else:
                                    item_lms[item_id][wv_term_index] += 1
                                item_lm_len[item_id] += 1
        # Do reviews as well
        # inter: user_id:token   item_id:token   rating:float    review:token_seq
        num_of_used_revs = {}
        if "review" in item_description_fields:
            item_desc_fields = [3]
            inter_file = os.path.join(dataset.dataset.dataset_path, f"{dataset.dataset.dataset_name}.inter")
            with open(inter_file, 'r') as infile:
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
                    if max_number_of_reviews_item is not None and num_of_used_revs[item_id] >= max_number_of_reviews_item:
                        continue
                    if item_id not in item_lms:
                        item_lms[item_id] = {}
                        item_lm_len[item_id] = 0
                    for fi in item_desc_fields:
                        desc = split[fi]
                        if len(desc) > 1:
                            num_of_used_revs[item_id] += 1
                        for term in desc.split():
                            if term in model.key_to_index:
                                wv_term_index = model.key_to_index[term]
                                if wv_term_index not in item_lms[item_id]:
                                    item_lms[item_id][wv_term_index] = 1
                                else:
                                    item_lms[item_id][wv_term_index] += 1
                                item_lm_len[item_id] += 1
        indices = [[], []]
        values = []
        for item_id in item_lms.keys():
            for k, v in item_lms[item_id].items():
                indices[0].append(item_id)
                indices[1].append(k)
                values.append(v / item_lm_len[item_id])
        self.item_lm_gt = SparseTensor(row=torch.tensor(indices[0]), col=torch.tensor(indices[1]), value=torch.tensor(values), sparse_sizes=(self.n_items, len(model.key_to_index)))
        if self.variant == 1:
            self.item_lm_gt = self.item_lm_gt.to(self.device)

        # user LM:
        user_lms = {} # TODO all users? only KITT users?
        user_lm_len = {}
        num_of_used_revs = {}
        if "review" in user_description_fields:
            user_desc_fields = [3]
            inter_file = os.path.join(dataset.dataset.dataset_path, f"{dataset.dataset.dataset_name}.inter")
            with open(inter_file, 'r') as infile:
                next(infile)
                for line in infile:
                    split = line.split("\t")
                    user_id = dataset.token2id_exists("item_id", split[0])
                    if user_id == -1:
                        continue
                    if user_id == 0:
                        print("Isnt that padding? user_id")
                    if user_id not in num_of_used_revs:
                        num_of_used_revs[user_id] = 0
                    if max_number_of_reviews_user is not None and num_of_used_revs[user_id] >= max_number_of_reviews_user:
                        continue
                    if user_id not in user_lms:
                        user_lms[user_id] = {}
                        user_lm_len[user_id] = 0
                    for fi in user_desc_fields:
                        desc = split[fi]
                        if len(desc) > 1:
                            num_of_used_revs[user_id] += 1
                        for term in desc.split():
                            if term in model.key_to_index:
                                wv_term_index = model.key_to_index[term]
                                if wv_term_index not in user_lms[user_id]:
                                    user_lms[user_id][wv_term_index] = 1
                                else:
                                    user_lms[user_id][wv_term_index] += 1
                                user_lm_len[user_id] += 1
        # TODO: KITT users add profiles here...
        indices = [[], []]
        values = []
        for user_id in user_lms.keys():
            for k, v in user_lms[user_id].items():
                indices[0].append(user_id)
                indices[1].append(k)
                values.append(v / user_lm_len[user_id])
        self.user_lm_gt = SparseTensor(row=torch.tensor(indices[0]), col=torch.tensor(indices[1]),
                                       value=torch.tensor(values), sparse_sizes=(self.n_users, len(model.key_to_index)))
        if self.variant == 1:
            self.user_lm_gt = self.user_lm_gt.to(self.device)
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

    def forward_lm_item(self, items):
        item_embs = self.item_embedding(items)
        pred = torch.matmul(item_embs, self.word_embedding.weight.T)
        return pred.squeeze()

    def forward_lm_user(self, users):
        user_embs = self.item_embedding(users)
        pred = torch.matmul(user_embs, self.word_embedding.weight.T)
        return pred.squeeze()

    def calculate_loss(self, interaction):
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]
        labels = interaction[self.LABEL]

        with profiler.record_function("REC output and loss"):
            output_rec = self.forward_rec(users, items)
            loss_rec = self.loss_rec(output_rec, labels)

        with profiler.record_function("LM output"):
            output_lm_item = self.forward_lm_item(items)

        with profiler.record_function("LM output user"):
            output_lm_user = self.forward_lm_user(users)

        if self.variant == 2:
            label_lm_items = self.item_lm_gt[items].to(self.device).to_dense()
            label_lm_users = self.user_lm_gt[users].to(self.device).to_dense()

        if self.variant == 1:
            with profiler.record_function("LM making label on GPU"):
                label_lm_items = self.item_lm_gt[items].to_dense()
                label_lm_users = self.user_lm_gt[users].to_dense()

        with profiler.record_function("LM loss item"):
            loss_lm_item = self.loss_lm(output_lm_item, label_lm_items)

        with profiler.record_function("LM loss user"):
            loss_lm_user = self.loss_lm(output_lm_user, label_lm_users)

        return loss_rec, self.alpha1 * loss_lm_item, self.alpha2 * loss_lm_user

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        output = self.forward_rec(user, item)
        return output