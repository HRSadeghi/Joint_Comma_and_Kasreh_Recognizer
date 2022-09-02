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



from utils.training_utils import load_pretrained_bert_model, get_device, train_step, evaluate
from utils.tag_mapping import get_tag2idx_idx2tag_dics
from utils.inference_utils import return_sen_to_real_form
from models.Joint_BERT_BiLSTM import JointBERTBiLSTMTagger
from data_loader.loader import Kasreh_DataLoader
from handlers.checkpoint_handler import load_checkpoint
from configs import Config
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
        self.model = JointBERTBiLSTMTagger(bert_model = bert_model, no_of_bert_layer = self.config.no_of_bert_layer)
        self.model = self.model.to(self.device)

        to_load={
            'model_state_dict': self.model,
            }

        load_checkpoint(self.config.checkpoint_dir, to_load)

        self.get_duration = get_duration


    def inference(dataLoader,
                model,
                tokenizer,
                idx2tag,
                output_path = None
                ):
        comma_dict = {1:'C', 0:'N'}
        start = time.time()
        for index, input in enumerate(dataLoader):
            out = model(input)
            kasreh_tags = torch.argmax(out[0], -1).detach().cpu().numpy()
            comma_tags = torch.argmax(out[1], -1).detach().cpu().numpy()

            for i in range(len(kasreh_tags)):
                #input_ids = list(input['input_ids'][i].detach().cpu().numpy())
                #input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
                input_sen = ' '.join(dataLoader.all_sens[i])

                _kasreh_tags = kasreh_tags[i]
                _kasreh_tags_with_name = [idx2tag[x] for x in _kasreh_tags]
                _comma_tags = comma_tags[i]
                _comma_tags_with_name = [comma_dict[x] for x in _comma_tags]

                output = return_sen_to_real_form(tokenizer, input_sen, _kasreh_tags_with_name, _comma_tags_with_name)
                output += '\n' 

                if output_path is not None:
                    with open(output_path, "a+") as out_file:
                        out_file.write(output)
                    output = ''
        
        duration = time.time() - start

        if output_path is None:
            return output, duration
        else:
            return duration



    def __call__(self, sent):
        dataLoader = Kasreh_DataLoader(all_sens = [sent.split(' ')], 
                                       all_kasreh_tags = None,
                                       all_comma_tags = None,
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
