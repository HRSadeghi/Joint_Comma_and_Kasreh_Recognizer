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



from Kasre.utils.training_utils import load_pretrained_bert_model, get_device, train_step, evaluate
from Kasre.utils.tag_mapping import get_tag2idx_idx2tag_dics
from Kasre.utils.inference_utils import return_sen_to_real_form
from Kasre.models.model import KasreAdder
from Kasre.data_loader.loader import Kasreh_DataLoader
from Kasre.handlers.checkpoint_handler import load_checkpoint
from Kasre.configs import Config
import torch
import torch.optim as optim
import torch.nn as nn
import time
from tqdm import tqdm


class JointKasreCommaRecognizer:
    def __init__(self, checkpoint_dir = None, get_duration = False):
        self.config = Config(checkpoint_dir = checkpoint_dir)
        _, self.idx2tag = get_tag2idx_idx2tag_dics()
        self.device = get_device()
        self.tokenizer, bert_model = load_pretrained_bert_model(model_name = self.config.Pretrained_BERT_model_name)
        self.model = KasreAdder(bert_model = bert_model, no_of_bert_layer = self.config.no_of_bert_layer)
        self.model = self.model.to(self.device)

        to_load={
            'model_state_dict': self.model,
            }

        load_checkpoint(self.config.checkpoint_dir, to_load)

        self.get_duration = get_duration


    def inference(self, 
                  dataLoader,
                  model,
                  tokenizer,
                  idx2tag,
                  output_path = None
                ):
        start = time.time()
        for index, input in enumerate(dataLoader):
            __o = model(input)

            for i in range(len(__o)):
                input_sen = ' '.join(dataLoader.all_sens[i])

                out_ids = torch.argmax(__o[i], -1).detach().cpu().numpy()
                out_labels = [idx2tag[x] for x in out_ids]

                output = return_sen_to_real_form(tokenizer, input_sen, out_labels)

                output += '\n' 


                if output_path is not None:
                    with open(output_path, "a+") as out_file:
                        out_file.write(output)
                    output = ''

            output = output.strip()
        
        duration = time.time() - start

        if output_path is None:
            return output, duration
        else:
            return duration


    def __call__(self, sent):
        dataLoader = Kasreh_DataLoader(all_sens = [sent.split(' ')], 
                                       all_tags = None,
                                       tokenizer = self.tokenizer, 
                                       tag2idx = None,
                                       mapping_dic = None, 
                                       device=self.device,
                                       batch_size = 1)

        output, duration = self.inference(dataLoader,
                                          self.model,
                                          self.tokenizer,
                                          self.idx2tag
                                        )

        if self.get_duration:
            return output, duration
        else:
            return output
