# @Time   : 2020/10/19
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
recbole.data.customized_dataset
##################################

We only recommend building customized datasets by inheriting.

Customized datasets named ``[Model Name]Dataset`` can be automatically called.
"""
import numpy as np
import pandas as pd
import os

import gensim
import gensim.downloader as api

from recbole.data.dataset import Kg_Seq_Dataset, Dataset
from recbole.utils import FeatureType, FeatureSource
from recbole.utils.utils import set_color


class GRU4RecKGDataset(Kg_Seq_Dataset):

    def __init__(self, config):
        super().__init__(config)


class KSRDataset(Kg_Seq_Dataset):

    def __init__(self, config):
        super().__init__(config)


class JOINTSRMFFULLDataset(Dataset):

    def __init__(self, config):
        self.word2vec_model = None
        self.word_embedding_file = config["word_embedding_file"]
        self.item_description_fields = config['item_description_fields']
        self.LM_FIELD = config['item_lm_field']
        self.LM_LEN_FIELD = config['item_lm_len_field']
        super().__init__(config)

    def _load_feat(self, filepath, source):
        """Load features according to source into :class:`pandas.DataFrame`.

        Set features' properties, e.g. type, source and length.

        Args:
            filepath (str): path of input file.
            source (FeatureSource or str): source of input file.

        Returns:
            pandas.DataFrame: Loaded feature

        Note:
            For sequence features, ``seqlen`` will be loaded, but data in DataFrame will not be cut off.
            Their length is limited only after calling :meth:`~_dict_to_interaction` or
            :meth:`~_dataframe_to_interaction`
        """
        self.logger.debug(set_color(f'Loading feature from [{filepath}] (source: [{source}]).', 'green'))

        load_col, unload_col = self._get_load_and_unload_col(source)
        if load_col == set():
            return None

        field_separator = self.config['field_separator']
        columns = []
        usecols = []
        dtype = {}
        with open(filepath, 'r') as f:
            head = f.readline()[:-1]
        for field_type in head.split(field_separator):
            field, ftype = field_type.split(':')
            try:
                ftype = FeatureType(ftype)
            except ValueError:
                raise ValueError(f'Type {ftype} from field {field} is not supported.')
            if load_col is not None and field not in load_col:
                continue
            if unload_col is not None and field in unload_col:
                continue
            if isinstance(source, FeatureSource) or source != 'link':
                self.field2source[field] = source
                self.field2type[field] = ftype
                if not ftype.value.endswith('seq'):
                    self.field2seqlen[field] = 1
            columns.append(field)
            usecols.append(field_type)
            dtype[field_type] = np.float64 if ftype == FeatureType.FLOAT else str

        if len(columns) == 0:
            self.logger.warning(f'No columns has been loaded from [{source}]')
            return None

        df = pd.read_csv(filepath, delimiter=self.config['field_separator'], usecols=usecols, dtype=dtype)
        df.columns = columns

        seq_separator = self.config['seq_separator']
        for field in columns:
            ftype = self.field2type[field]
            if not ftype.value.endswith('seq'):
                continue
            df[field].fillna(value='', inplace=True)
            if ftype == FeatureType.TOKEN_SEQ:
                df[field] = [np.array(list(filter(None, _.split(seq_separator)))) for _ in df[field].values]
            elif ftype == FeatureType.FLOAT_SEQ:
                df[field] = [np.array(list(map(float, filter(None, _.split(seq_separator))))) for _ in df[field].values]
            elif ftype == FeatureType.LM_KEY_VAL_SEQ:
                df[field] = [np.array(list(map(str, filter(None, _.split(seq_separator))))) for _ in df[field].values]
            self.field2seqlen[field] = max(map(len, df[field].values))
        return df

    def _data_processing(self):
        super()._data_processing()
        self._create_item_language_model()

    def _create_item_language_model(self):
        # new_item_feat = pd.DataFrame
        if self.word2vec_model is None:
            gensim_cache = open('gensim_cache_path', 'r').read().strip()
            os.environ['GENSIM_DATA_DIR'] = str(gensim_cache)
            pretrained_embedding_name = self.word_embedding_file
            model_path = api.load(pretrained_embedding_name, return_path=True)
            self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(model_path)
        vocab_size = len(self.word2vec_model.key_to_index)
        new_field_lm = [np.zeros(vocab_size, dtype=np.uint8) for i in range(self.item_num)]
        new_field_lm_len = [0 for i in range(self.item_num)]
        for field in self.item_feat:
            ftype = self.field2type[field]
            if ftype == FeatureType.LM_KEY_VAL_SEQ: # required?? how to keep them beforehand?
                if field in self.item_description_fields:
                    for i in range(1, len(self.item_feat[field].values)):
                        key_vals = self.item_feat[field].values[i]
                        for s in key_vals:
                            k, v = s.split(",")
                            if k in self.word2vec_model.key_to_index:
                                wv_term_index = self.word2vec_model.key_to_index[k]
                                new_field_lm[i][wv_term_index] += np.uint8(v)
                                new_field_lm_len[i] += int(v)
                # TODO Here we should remove the columnsof there is no space
        self.item_feat[self.LM_FIELD] = new_field_lm
        self.item_feat[self.LM_LEN_FIELD] = new_field_lm_len
        self.field2type[self.LM_FIELD] = FeatureType.LM_KEY_VAL_SEQ
        self.field2type[self.LM_LEN_FIELD] = FeatureType.LM_KEY_VAL_SEQ
        self.field2seqlen[self.LM_FIELD] = max(map(len, self.item_feat[self.LM_FIELD].values))
        self.field2seqlen[self.LM_LEN_FIELD] = 1
        # this here or before other data processing???
