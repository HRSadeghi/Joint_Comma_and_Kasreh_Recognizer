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

from os.path import join
from pathlib import Path


class Config:
    def __init__(self, checkpoint_dir=None):
        self.base_ckpt_dir = join(str(Path.home()), 'joint_kasre_comma_resources')

        # model configs
        self.Pretrained_BERT_model_name = join(self.base_ckpt_dir, 'HooshvareLab/bert-fa-zwnj-base')
        self.no_of_bert_layer = 9


        # dataset configs
        self.train_file_path = "./dataset/train_data.txt"
        self.test_file_path = "./dataset/test_data.txt"
        self.valid_file_path = "./dataset/valid_data.txt"


        # checkpoint configs
        if checkpoint_dir is not None:
            self.checkpoint_dir = checkpoint_dir
        else:
            self.checkpoint_dir = join(self.base_ckpt_dir, "saved_checkpoints")
        self.n_saved_ckpts = 3


        # Traning, test and validation configs
        self.batch_size = 32


        # Traning configs
        self.epochs = 2
        self.valid_size = 0.1

        




