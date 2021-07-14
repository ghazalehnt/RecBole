import torch

from recbole.data.dataloader.general_dataloader import GeneralNegSampleDataLoader
from recbole.data.interaction import Interaction, cat_interactions
from recbole.utils import InputType


class GeneralNegSampleBothWaysDataloader(GeneralNegSampleDataLoader):
    def __init__(
        self, config, dataset, sampler, neg_sample_args, batch_size=1, dl_format=InputType.POINTWISE, shuffle=False
    ):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.uid_list, self.uid2index, self.uid2items_num = None, None, None

        super().__init__(
            config, dataset, sampler, neg_sample_args, batch_size=batch_size, dl_format=dl_format, shuffle=shuffle
        )

    def _neg_sampling(self, inter_feat):
        uids = inter_feat[self.uid_field]
        labels = inter_feat[self.label_field]
        pos_uids_idx = (uids * labels).nonzero(as_tuple=True)
        neg_uids_idx = (uids * (1 - labels)).nonzero(as_tuple=True)
        neg_iids = self.sampler.sample_by_user_ids(uids[pos_uids_idx], self.neg_sample_by)
        pos_iids = self.sampler.sample_by_user_ids(uids[neg_uids_idx], round(self.neg_sample_by / 3))
        return self.sampling_func(inter_feat, pos_uids_idx, neg_iids, neg_uids_idx, pos_iids)

    def _neg_sample_by_point_wise_sampling(self, inter_feat, pos_idx, neg_iids, neg_idx, pos_iids):
        pos_inter_num = len(pos_idx[0])
        neg_inter_num = len(neg_idx[0])
        new_data_pos = inter_feat[pos_idx].repeat(self.neg_sample_by + 1)
        new_data_pos[self.iid_field][pos_inter_num:] = neg_iids
        new_data_pos = self.dataset.join(new_data_pos)
        labels_pos = torch.zeros(pos_inter_num * (self.neg_sample_by + 1))
        labels_pos[:pos_inter_num] = 1
        new_data_pos.update(Interaction({self.label_field: labels_pos}))
        new_data_neg = inter_feat[neg_idx].repeat(round(self.neg_sample_by / 3) + 1)
        new_data_neg[self.iid_field][neg_inter_num:] = pos_iids
        new_data_neg = self.dataset.join(new_data_neg)
        labels_neg = torch.ones(neg_inter_num * (round(self.neg_sample_by / 3) + 1))
        labels_neg[:neg_inter_num] = 0
        new_data_neg.update(Interaction({self.label_field: labels_neg}))
        new_data = cat_interactions([new_data_pos, new_data_neg])
        return new_data

    def _neg_sample_by_pair_wise_sampling(self, inter_feat, neg_iids):
        NotImplementedError("not implemented")
