#Copyright 2022 Hamidreza Sadeghi. All rights reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import torch
from torch.utils.data import Dataset, DataLoader


class Kasreh_DataLoader(Dataset):
    r"""
    This is a class to define the Kasreh Ezafeh model. This model is created using BERT model and BiLSTM model.
    Args:
        all_sens (`list`):
            A list of all sentences. Each sentence is in the form of a list of words.
            e.g. 
            ```python
            >>>all_sens = [['ضحاک', 'بن', 'قیس', 'مردم', 'عراق', 'را', 'مذمت', 'کرد']]
            ```
        all_tags (`list`):
            A list of all tags. Each tag itself is a list that shows the tag of each corresponding word.
            e.g. 
            ```python
            >>>all_tags = [['O', 'O', 'O', 'e', 'O', 'O', 'O', 'O']]
            ```
        tokenizer (`transformers.models.bert.tokenization_bert_fast.BertTokenizerFast`):
            The tokenizer of pretrained BERT model.
        tag2idx (`dict`):
            A dictionary that uniquely maps each tag to an index.
            e.g. 
            ```python
            >>>tag2idx = {'O': 0, 'e': 1, 'ye': 2}
            ```
        mapping_dic (`dict`, *optional*, defaults to `None`):
            A dictionary to convert a tag to another tag.
            e.g. 
            ```python
            >>>mapping_dic = {'@e':'e', 'O': 'O', 'e':'e', 've': 'ye', 'y': 'ye', 'ye': 'ye'}
            ```
        device (`string`, *optional*, defaults to "cpu"):
            Specifies whether the code is executed on CPU or CUDA.
        
        batch_size (`int`, *optional*, defaults to 64):
            Batch size for each epoch of training, testing or evaluation.


    Examples:
    ```python
    >>>from transformers import AutoTokenizer

    >>>tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-zwnj-base")
    >>>train_sens = [['ضحاک', 'بن', 'قیس', 'مردم', 'عراق', 'را', 'مذمت', 'کرد']]
    >>>train_tags = [['O', 'O', 'O', 'e', 'O', 'O', 'O', 'O']]

    >>>train_dataLoader = Kasreh_DataLoader(train_sens, 
    ...                                  train_tags,
    ...                                  tokenizer)

    >>>train_dataLoader[0]
    ```"""

    def __init__(self,
                 all_sens, 
                 all_kasreh_tags,
                 all_comma_tags,
                 tokenizer, 
                 tag2idx,
                 mapping_dic = None,
                 device='cpu',
                 batch_size = 64
                 ):

        self.all_sens = all_sens
        self.all_kasreh_tags = all_kasreh_tags
        self.all_comma_tags = all_comma_tags
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx
        self.device = device
        self.mapping_dic = mapping_dic
        self.batch_size = batch_size    

    def __len__(self):
        r"""
        Denotes the number of batches per epoch.

        Returns:
            __len (`int`):
                Number of batches.
        """
        __len = int(len(self.all_sens) // self.batch_size)
        if __len*self.batch_size < len(self.all_sens):
            __len += 1
        return __len

    def __getitem__(self, index):
        r"""
        A function to take a batch of input and output data corresponding to the input and output of the neural network used
        Args:
            index (`int`):
                Index of a batch.

        Returns:
            encoded_sens (`dict`): 
                A dictionary of BERT model inputs. Each item is a pytorch tensor.
            encoded_labels (`torch.Tensor`): 
                A pytorch tensor obtained based on output tags.
        """

        # Specifying start and end index of a batch
        if (index+1)*self.batch_size > len(self.all_sens):
            start = index*self.batch_size
            end = len(self.all_sens)
            batch_len = len(self.all_sens) - index*self.batch_size
        else:
            start = index*self.batch_size
            end = (index+1)*self.batch_size
            batch_len = self.batch_size
        ###

        # Getting a batch of sentences and tags according to the start and end index
        sens_batch = self.all_sens[start:end]
        if self.all_kasreh_tags is not None:
            kasreh_tags_batch = self.all_kasreh_tags[start:end]
        if self.all_comma_tags is not None:
            comma_tags_batch = self.all_comma_tags[start:end]
        ###

        # Concatenating all words and creating sentence for each input sample
        _sens_batch = [' '.join(x) for x in sens_batch]
        
        # Tokenizing sentences using the pre-trained BERT tokenizer
        encoded_sens = self.tokenizer(_sens_batch, return_tensors='pt', padding=True)

        if self.all_kasreh_tags is not None:
            # Aligning kasreh tags with subwords after the tokenizer is applied to sentences
            _out_kasreh_labels = self._align_tokens_and_kasreh_labels(encoded_sens, kasreh_tags_batch)

            # Converting the kasreh tags of each sentence to indices
            encoded_kasreh_labels = self._encode_kasreh_labels(_out_kasreh_labels)

        if self.all_comma_tags is not None:
            # Aligning comma tags with subwords after the tokenizer is applied to sentences
            _out_comma_labels = self._align_tokens_and_comma_labels(encoded_sens, comma_tags_batch)

            # Converting the comma tags of each sentence to indices
            encoded_comma_labels = self._encode_comma_labels(_out_comma_labels)


        # Switching execution to CPU or CUDA
        encoded_sens = encoded_sens.to(self.device)
        if self.all_kasreh_tags is not None:
            encoded_kasreh_labels = encoded_kasreh_labels.to(self.device)
        if self.all_comma_tags is not None:
            encoded_comma_labels = encoded_comma_labels.to(self.device)
        ##

        if self.all_kasreh_tags is not None and self.all_comma_tags is not None:
            return encoded_sens, [encoded_kasreh_labels, encoded_comma_labels]
        elif self.all_kasreh_tags is not None:
            return encoded_sens, encoded_kasreh_labels
        elif self.all_comma_tags is not None:
            return encoded_sens, encoded_comma_labels
        else:
            return encoded_sens



    def _encode_kasreh_labels(self, out_labels):
        r"""
        Converting the tags of each sentence to indices.

        Args:
            out_labels (`list`):
                A list of lists of strings.

        Returns:
            _encoded_labels (`torch.Tensor`): 
                A pytorch tensor containing the indices of the tags.
   
        """
        _encoded_labels = []
        for x in out_labels:
            _curr = []
            for y in x:
                _curr += [self.tag2idx[y]]
            _encoded_labels.append(_curr)

        _encoded_labels = torch.tensor(_encoded_labels)
        return _encoded_labels


    def _encode_comma_labels(self, out_labels):
        _encoded_labels = []
        for x in out_labels:
            _curr = []
            for y in x:
                _curr += [0 if y == 'N' else 1]
            _encoded_labels.append(_curr)

        _encoded_labels = torch.tensor(_encoded_labels)
        return _encoded_labels


    def _align_tokens_and_kasreh_labels(self, encoded_sens, labels):
        r"""
        A function to align tags with subwords after the tokenizer is applied to sentences.

        Args:
            encoded_sens (`dict`):
                A dictionary of BERT model inputs. Each item is a pytorch tensor.
            labels (`list`):
                A list of lists of strings.

        Returns:
            _out_labels (`list`): 
                A list of lists of strings. 
   
        """

        _out_labels = []
        batch_len = len(encoded_sens['input_ids'])


        for i in range(batch_len):
            _curr_label = []

            num_prev_none = 0 # Number of special tokens before the beginning of the sentence
            num_next_none = 0 # Number of special tokens after the ending of the sentence

            # Specifying the number of special tokens before the beginning of the sentence
            for x in encoded_sens.word_ids(batch_index=i):
                if x is None:
                    num_prev_none += 1
                else:
                    break
            ##

            # Specifying the number of special tokens after the ending of the sentence
            for x in encoded_sens.word_ids(batch_index=i)[::-1]:
                if x is None:
                    num_next_none += 1
                else:
                    break
            ##

            # Number of real tokens
            num_tokens = len(set([x for x in encoded_sens.word_ids(batch_index=i) if x is not None]))

            # Adding 'O' tag to the number of tokens that come before the beginning of the sentence
            if self.mapping_dic is not None:
                _curr_label += [self.mapping_dic['O']]*num_prev_none
            else:
                _curr_label += ['O']*num_prev_none
            ##

            # Aligning the real words tag with the subwords generated by the tokenizer
            for j in range(num_tokens):
                start, end = encoded_sens.word_to_tokens(batch_or_word_index = i, word_index = j)
                _repetition = end - start - 1
                if self.mapping_dic is not None:
                    _curr_label += [self.mapping_dic['O']]*_repetition
                else:
                    _curr_label += ['O']*_repetition
                _curr_label += [labels[i][j]]
            ##

            # Adding 'O' tag to the number of tokens that come after the ending of the sentence
            if self.mapping_dic is not None:
                _curr_label += [self.mapping_dic['O']]*num_next_none
            else:
                _curr_label += ['O']*num_next_none
            ##

            _out_labels.append(_curr_label)
        return _out_labels



    
    def _align_tokens_and_comma_labels(self, encoded_sens, comma_labels):
        r"""
        A function to align comma_tags with subwords after the tokenizer is applied to sentences.

        Args:
            encoded_sens (`dict`):
                A dictionary of BERT model inputs. Each item is a pytorch tensor.
            labels (`list`):
                A list of lists of strings.

        Returns:
            _out_labels (`list`): 
                A list of lists of strings. 
   
        """

        _out_labels = []
        batch_len = len(encoded_sens['input_ids'])


        for i in range(batch_len):
            _curr_label = []

            num_prev_none = 0 # Number of special tokens before the beginning of the sentence
            num_next_none = 0 # Number of special tokens after the ending of the sentence

            # Specifying the number of special tokens before the beginning of the sentence
            for x in encoded_sens.word_ids(batch_index=i):
                if x is None:
                    num_prev_none += 1
                else:
                    break
            ##

            # Specifying the number of special tokens after the ending of the sentence
            for x in encoded_sens.word_ids(batch_index=i)[::-1]:
                if x is None:
                    num_next_none += 1
                else:
                    break
            ##

            # Number of real tokens
            num_tokens = len(set([x for x in encoded_sens.word_ids(batch_index=i) if x is not None]))

            # Adding 'N' tag to the number of tokens that come before the beginning of the sentence
            _curr_label += ['N']*num_prev_none
            ##

            # Aligning the real words tag with the subwords generated by the tokenizer
            for j in range(num_tokens):
                start, end = encoded_sens.word_to_tokens(batch_or_word_index = i, word_index = j)
                _repetition = end - start - 1
                _curr_label += ['N']*_repetition
                _curr_label += [comma_labels[i][j]]
            ##

            # Adding 'N' tag to the number of tokens that come after the ending of the sentence
            _curr_label += ['N']*num_next_none
            ##

            _out_labels.append(_curr_label)
        return _out_labels
