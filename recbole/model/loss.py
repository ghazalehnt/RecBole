# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
recbole.model.loss
#######################
Common Loss in recommender system
"""

import torch
import torch.nn as nn
import numpy as np

class BPRLoss(nn.Module):
    """ BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class RegLoss(nn.Module):
    """ RegLoss, L2 regularization on model parameters

    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss


class EmbMarginLoss(nn.Module):
    """ EmbMarginLoss, regularization on embeddings
    """

    def __init__(self, power=2):
        super(EmbMarginLoss, self).__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0).to(dev)
        cache_zero = torch.tensor(0.0).to(dev)
        emb_loss = torch.tensor(0.).to(dev)
        for embedding in embeddings:
            norm_e = torch.sum(embedding ** self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        logsoftmax = nn.LogSoftmax(dim=1)
        return torch.mean(torch.sum(- target * logsoftmax(output), 1))


class SoftCrossEntropyLossByNegSampling(nn.Module):
    def __init__(self, num_neg_samples, unig_dist, alpha): # TODO  neg_sampling_dis uniform, P_w, P_w to the power of sth
        self.num_neg_samples = num_neg_samples
        self.noise_dist = {key: val ** alpha for key, val in unig_dist.items()}
        self.noise_dist_Z = sum(self.noise_dist.values())
        super(SoftCrossEntropyLossByNegSampling, self).__init__()

    def forward(self, output, target_keys, target_values):
        batch_sum = 0
        for idx in range(0, len(output)):
            vals = torch.tensor(target_values[idx])
            s1 = torch.sum((vals / torch.sum(vals) * torch.log(torch.sigmoid(output[idx][target_keys[idx]]))))

            neg_samples = self.sample_negs(len(target_keys[idx]), target_keys[idx])
            s2 = torch.sum(torch.log(torch.sigmoid(-1 * output[idx][neg_samples])))

            batch_sum += s1 + s2
        return batch_sum/output.shape[0]

    def sample_negs(self, num_pos, item_lm_keys):
        num_samples = num_pos * self.num_neg_samples
        noise_dist = self.noise_dist.copy()
        Z = self.noise_dist_Z
        for ti in item_lm_keys:
            Z -= noise_dist[ti]
            noise_dist[ti] = 0
        samples = np.random.choice(list(noise_dist.keys()), num_samples, replace=False, p=np.divide(list(noise_dist.values()), Z))
        return samples

# # 2-layer HS https://github.com/leimao/Two-Layer-Hierarchical-Softmax-PyTorch/blob/1b65263308b556b5ae038f866cde925095bc0824/utils.py#L98
# class HierarchicalSoftmax(nn.Module):
#     def __init__(self, ntokens, nout, ntokens_per_class = None):
#         super(HierarchicalSoftmax, self).__init__()
#
#         # Parameters
#         self.ntokens = ntokens
#         self.nout = nout
#
#         if ntokens_per_class is None:
#             ntokens_per_class = int(np.ceil(np.sqrt(ntokens)))
#
#         self.ntokens_per_class = ntokens_per_class
#
#         self.nclasses = int(np.ceil(self.ntokens * 1. / self.ntokens_per_class))
#         self.ntokens_actual = self.nclasses * self.ntokens_per_class
#
#         print(self.ntokens)
#         print(self.ntokens_per_class)
#         print(self.nclasses)
#         print(self.ntokens_actual)
#
#
#         self.layer_top_W = nn.Parameter(torch.FloatTensor(self.nout, self.nclasses), requires_grad=True)
#         self.layer_top_b = nn.Parameter(torch.FloatTensor(self.nclasses), requires_grad=True)
#
#         self.layer_bottom_W = nn.Parameter(torch.FloatTensor(self.nclasses, self.nout, self.ntokens_per_class), requires_grad=True)
#         self.layer_bottom_b = nn.Parameter(torch.FloatTensor(self.nclasses, self.ntokens_per_class), requires_grad=True)
#
#         self.softmax = nn.Softmax(dim=1)
#
#         self.init_weights()
#
#     def init_weights(self):
#
#         initrange = 0.1
#         self.layer_top_W.data.uniform_(-initrange, initrange)
#         self.layer_top_b.data.fill_(0)
#         self.layer_bottom_W.data.uniform_(-initrange, initrange)
#         self.layer_bottom_b.data.fill_(0)
#
#     def forward(self, inputs, labels):
#         batch_size, d = inputs.size()
#
#         if labels is not None:
#             label_position_top = labels / self.ntokens_per_class
#             label_position_bottom = labels % self.ntokens_per_class
#
#             layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
#             layer_top_probs = self.softmax(layer_top_logits)
#
#             layer_bottom_logits = torch.squeeze(torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_W[label_position_top]), dim=1) + self.layer_bottom_b[label_position_top]
#             layer_bottom_probs = self.softmax(layer_bottom_logits)
#
#             target_probs = layer_top_probs[torch.arange(batch_size).long(), label_position_top] * layer_bottom_probs[torch.arange(batch_size).long(), label_position_bottom]
#             loss = -torch.mean(torch.log(target_probs))
#
#             return target_probs, loss
