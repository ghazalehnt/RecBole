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


class JOINTSRMF(GeneralRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(JOINTSRMF, self).__init__(config, dataset)
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
        lm_gt_keys_t = [[] for i in range(self.n_items)]
        lm_gt_values_t = [[] for i in range(self.n_items)]
        item_desc_fields = []
        if "item_description" in item_description_fields:
            item_desc_fields.append(3)
        if "item_genres" in item_description_fields:
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
                                if wv_term_index in lm_gt_keys_t[item_id]:
                                    key_idx = lm_gt_keys_t[item_id].index(wv_term_index)
                                    lm_gt_values_t[item_id][key_idx] += 1
                                else:
                                    lm_gt_keys_t[item_id].append(wv_term_index)
                                    lm_gt_values_t[item_id].append(1)

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
                        if len(desc) > 1:
                            num_of_used_revs[item_id] += 1
                        for term in desc.split():
                            if term in model.key_to_index:
                                wv_term_index = model.key_to_index[term]
                                if term in model.key_to_index:
                                    wv_term_index = model.key_to_index[term]
                                    if wv_term_index in lm_gt_keys_t[item_id]:
                                        key_idx = lm_gt_keys_t[item_id].index(wv_term_index)
                                        lm_gt_values_t[item_id][key_idx] += 1
                                    else:
                                        lm_gt_keys_t[item_id].append(wv_term_index)
                                        lm_gt_values_t[item_id].append(1)

        keys_sum = 0
        zeros = 0
        max_len = 0
        for lm in lm_gt_keys_t:
            if len(lm) == 0:
                zeros += 1
            else:
                keys_sum += len(lm)
                if len(lm) > max_len:
                    max_len = len(lm)
        print(keys_sum)
        print(zeros)
        print(self.n_items - 1 - zeros)
        print(keys_sum / (self.n_items - zeros))
        print(max_len)
        e = time.time()
        self.logger.info(f"{e - s}s")
        self.logger.info(f"Done with lm_gt construction!")

        MAX_LM_SIZE = max_len + 1
        s = time.time()
        self.lm_gt_keys = -1 * torch.ones((self.n_items, MAX_LM_SIZE))
        self.lm_gt_values = torch.zeros((self.n_items, MAX_LM_SIZE))
        for item_id in range(len(lm_gt_keys_t)):
            for j in range(len(lm_gt_keys_t[item_id])):
                self.lm_gt_keys[item_id][j] = lm_gt_keys_t[item_id][j]
                self.lm_gt_values[item_id][j] = lm_gt_values_t[item_id][j]
        # values should be probabilities
        self.lm_gt_values = (self.lm_gt_values.T / self.lm_gt_values.sum(1)).T
        e = time.time()
        self.logger.info(f"Done making tensors and probabilities!: {e - s}s")

        self.sigmoid = nn.Sigmoid()
        self.loss_rec = nn.BCELoss()
        self.loss_lm = SoftCrossEntropyLoss()
        self.label_lm = torch.zeros(config["train_batch_size"]-1, self.vocab_size, device=self.device) # why is it -1?

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

#        s = time.time()
        output_lm = self.forward_lm(item) # output should be unnormalized counts
#        e = time.time()
#        self.logger.info(f"{e - s}s output_lm")

#        s = time.time()
        item_term_keys = self.lm_gt_keys[item]
        item_term_vals = self.lm_gt_values[item]
#        e = time.time()
#        self.logger.info(f"{e - s}s get entries")

        s = time.time()
        label_lm = self.label_lm.detach().clone()
        for i in range(len(item_term_keys)):
            for j in range(len(item_term_keys[i])):
                k = int(item_term_keys[i][j])
                if k == -1:
                    break
                v = item_term_vals[i][j]
                label_lm[i][k] = v
        e = time.time()
        self.logger.info(f"{e - s}s make tensor lm")
        
#        s = time.time()
        loss_lm = self.loss_lm(output_lm, label_lm)
#        e = time.time()
#        self.logger.info(f"{e - s}s loss_lm")

        return loss_rec, self.alpha * loss_lm

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        output = self.forward_rec(user, item)
        return output
