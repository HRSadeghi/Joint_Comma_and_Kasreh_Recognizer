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

import gdown
import os
from os.path import join
from pathlib import Path

base_ckpt_dir = join(str(Path.home()), 'joint_kasre_comma_resources')


bert_ckpt_dir = join(base_ckpt_dir, 'HooshvareLab/bert-fa-zwnj-base')

our_model_ckpt_dir = join(base_ckpt_dir, 'saved_checkpoints')

def downloader():
    if not os.path.exists(bert_ckpt_dir):
        os.makedirs(bert_ckpt_dir)
    if not os.path.exists(our_model_ckpt_dir):
        os.makedirs(our_model_ckpt_dir)
    files = [
        {"url" : "https://drive.google.com/file/d/1-U3DU6NB1jF4w-5EV2RV0UVrYVxUiB8U/view?usp=sharing",
        "output" : join(bert_ckpt_dir, "tokenizer_config.json")},

        {"url" : "https://drive.google.com/file/d/1-QQogUl_661IelbPBc830OqJga9boAHp/view?usp=sharing",
        "output" : join(bert_ckpt_dir, "tokenizer.json")},

        {"url" : "https://drive.google.com/file/d/1-ANAFR-VCFhWe4BE15iXJhidJ1LeFJq4/view?usp=sharing",
        "output" : join(bert_ckpt_dir, "vocab.txt")},

        {"url" : "https://drive.google.com/file/d/1-44_SHodc8s9OYGZcKeLXcb6t0OEK8Ns/view?usp=sharing",
        "output" : join(bert_ckpt_dir, "special_tokens_map.json")},

        {"url" : "https://drive.google.com/file/d/1-1qsC6-ueCDygGKeBxUgbnS2QQjZK8Kj/view?usp=sharing",
        "output" : join(bert_ckpt_dir, "pytorch_model.bin")},

        {"url" : "https://drive.google.com/file/d/1-RzQtWWGdix7h9y_jLk5xmVItTx6p6nj/view?usp=sharing",
        "output" : join(bert_ckpt_dir, "config.json")},

        {"url" : "https://drive.google.com/file/d/1-NikeDA8_ZkSThPANX9Y5tattCKd-xWt/view?usp=sharing",
        "output" : join(our_model_ckpt_dir, "best_global_time_2_val_accuracy=0.9718.pt")}
    ]

    for obj in files:
        if not os.path.exists(obj["output"]):
            print(f'downloading {obj["output"]}')
            gdown.download(obj['url'], obj["output"], quiet=False, fuzzy=True)
    print("done!")

if __name__ == "__main__":
    downloader()
