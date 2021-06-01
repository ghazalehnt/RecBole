from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import SoftCrossEntropyLoss, SoftCrossEntropyLossByNegSampling  # , HierarchicalSoftmax
from recbole.utils import InputType
import torch.nn as nn
import torch
from torch.nn.init import normal_
import gensim
import gensim.downloader as api
import os

class JOINTSRMFNEGS(GeneralRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(JOINTSRMFNEGS, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config['LABEL_FIELD']

        self.embedding_dim = config['embedding_dimension']
        self.alpha = config["alpha"]
        item_description_fields = config['item_description_fields']

        LM_neg_samples = config["LM_neg_samples"]

        self.logger.info(f"embedding_dimension = {self.embedding_dim}")
        self.logger.info(f"alpha = {self.alpha}")
        self.logger.info(f"item_description_fields = {item_description_fields}")
        self.logger.info(f"LM_neg_samples = {LM_neg_samples}")

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

        # getting the lms:
        # TODO: this should be changed if we could load fields from other atomic files as well
        # noise_dist = {} # This is the noise distribution!
        # item_features = dataset.get_item_feature()
        # self.lm_gt = torch.zeros((len(item_features), len(model.vocab)), device=self.device)
        # self.lm_gt_cnt = torch.zeros((len(item_features), 1), device=self.device)
        # for item_description_field in item_description_fields:
        #     item_descriptions = item_features[item_description_field]  # [0] is PAD
        #     for i in range(1, len(item_descriptions)):
        #         for termid in item_descriptions[i]:
        #             if termid > 0: # termid=0 is reserved for padding
        #                 term = dataset.id2token(item_description_field, termid)
        #                 term = str(term)
        #                 term = term.lower()
        #                 if model.vocab.__contains__(term):
        #                     wv_term_index = model.vocab.get(term).index
        #                 else:
        #                     wv_term_index = model.vocab.get("unk").index
        #
        #                 self.lm_gt[i][wv_term_index] += 1
        #                 self.lm_gt_cnt[i] += 1
        #                 if wv_term_index not in noise_dist:
        #                     noise_dist[wv_term_index] = 0
        #                 noise_dist[wv_term_index] += 1
        # self.logger.info(f"Done with lm_gt construction!")

        noise_dist = {}  # This is the noise distribution!
        self.lm_gt = torch.zeros((self.n_items, len(model.key_to_index)), device=self.device)
        item_LM_file = os.path.join(dataset.dataset.dataset_path, f"{dataset.dataset.dataset_name}.item")
        item_desc_fields = []
        if "item_description" in item_description_fields:
            item_desc_fields.append(3)
        if "item_genres" in item_description_fields:
            item_desc_fields.append(4)
        #TODO other fields? e.g. review? have to write another piece of code
        with open(item_LM_file, 'r') as infile:
            next(infile)
            for line in infile:
                split = line.split("\t")
                item_id = dataset.token2id("item_id", split[0])
                for fi in item_desc_fields:
                    desc = split[fi]
                    for term in desc.split():
                        if term in model.key_to_index:
                            wv_term_index = model.key_to_index[term]
                        else:
                            wv_term_index = model.key_to_index["unk"]
                        self.lm_gt[item_id][wv_term_index] += 1
                        if wv_term_index not in noise_dist:
                            noise_dist[wv_term_index] = 0
                        noise_dist[wv_term_index] += 1
        self.logger.info(f"Done with lm_gt construction!")

        self.sigmoid = nn.Sigmoid()
        self.loss_rec = nn.BCELoss()
        self.loss_lm = SoftCrossEntropyLossByNegSampling(LM_neg_samples, noise_dist, 0.75, self.device) # dist to the power of 3/4

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    @staticmethod
    def get_entries(array, keys, tensor=False, device=None):
        if not tensor:
            ret = []
            for k in keys:
                ret.append(array[k])
            return ret
        else:
            ret = torch.tensor(device=device)


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

        output_lm = self.forward_lm(item)# output should be unnormalized counts
        loss_lm = self.loss_lm(output_lm, self.lm_gt[item])
        print(loss_lm)

        return loss_rec, self.alpha * loss_lm

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        output = self.forward_rec(user, item)
        return output

