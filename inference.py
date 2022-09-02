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
from models.Joint_BERT_BiLSTM import JointBERTBiLSTMTagger
from data_loader.loader import Kasreh_DataLoader
from handlers.checkpoint_handler import load_checkpoint
import torch
import torch.optim as optim
import torch.nn as nn
import time
from tqdm import tqdm
import argparse



def inference(dataLoader,
              model,
              tokenizer,
              idx2tag,
              output_path = None
              ):
    comma_dict = {1:'C', 0:'N'}
    start = time.time()
    for input in tqdm(dataLoader):
        out = model(input)
        kasreh_tags = torch.argmax(out[0], -1).detach().cpu().numpy()
        comma_tags = torch.argmax(out[1], -1).detach().cpu().numpy()

        for i in range(len(kasreh_tags)):
            input_ids = list(input['input_ids'][i].detach().cpu().numpy())
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            _kasreh_tags = kasreh_tags[i]
            _kasreh_tags_with_name = [idx2tag[x] for x in _kasreh_tags]
            _comma_tags = comma_tags[i]
            _comma_tags_with_name = [comma_dict[x] for x in _comma_tags]

            output = ''
            for x,y,z in zip(input_tokens, _kasreh_tags_with_name, _comma_tags_with_name):
                if x not in set(tokenizer.special_tokens_map.values()):
                    output += x + '\t' + y + '\t' + z + '\n'
            
            output += '#######\n'

            if output_path is not None:
                with open(output_path, "a+") as out_file:
                    out_file.write(output)
                output = ''
    
    duration = time.time() - start

    if output_path is None:
        return output, duration
    else:
        return duration




def main():
    parser = argparse.ArgumentParser(description='Create a train command.')

    parser.add_argument('--input_sen', 
                        type=str,
                        default='',
                        help='A sentence in Persian language')
    parser.add_argument('--input_text_file', 
                        type=str,
                        default='',
                        help='Path of a .txt file where each of its lines is Persian sentence')
    parser.add_argument('--output_text_file', 
                        type=str,
                        default='',
                        help='A .txt file path to save tags generated by model')

    parser.add_argument('--checkpoint_dir', 
                        type=str,
                        default='saved_checkpoints',
                        help='path to the checkpoint directory')

    parser.add_argument('--batch_size', 
                        type=int,
                        default=64,
                        help='path to the valid_data.txt file')

    parser.add_argument('--Pretrained_BERT_model_name', 
                        type=str,
                        default='HooshvareLab/bert-fa-zwnj-base',
                        help='The name of pretrained BERT model or a path to pretrained BERT model')
    
    parser.add_argument('--no_of_bert_layer', 
                        type=int,
                        default=7,
                        help='Number of bert layers that is used in new model')

    args = parser.parse_args()



    device = get_device()
    _, idx2tag = get_tag2idx_idx2tag_dics()

    print('Loading tokenizer of pretrained BERT model ...')    
    tokenizer, bert_model = load_pretrained_bert_model(model_name = args.Pretrained_BERT_model_name)



    print('Loading model weights ...')   
    model = JointBERTBiLSTMTagger(bert_model = bert_model, no_of_bert_layer = args.no_of_bert_layer)
    model = model.to(device)

    to_load={
            'model_state_dict': model,
            }

    load_checkpoint(args.checkpoint_dir, to_load)

    output = ''
    duration = 0


    if args.input_sen != '':
        print(f'Finding Kasreh for {args.input_sen} ...')
        dataLoader = Kasreh_DataLoader(all_sens = [args.input_sen.split(' ')], 
                                    all_kasreh_tags = None,
                                    all_comma_tags = None,
                                    tokenizer = tokenizer, 
                                    tag2idx = None,
                                    mapping_dic = None, 
                                    device=device,
                                    batch_size = 1)


        output, duration = inference(dataLoader,
                                     model,
                                     tokenizer,
                                     idx2tag
                                     )

        print(output)
        print(f'Duration_time: {duration:4f}')


    if args.input_text_file != '':
        print(f'Finding Kasreh for {args.input_text_file} ...')
        with open(args.input_text_file) as f:
            all_sens = f.readlines()
        dataLoader = Kasreh_DataLoader(all_sens = [sen.split(' ') for sen in all_sens], 
                                       all_kasreh_tags = None,
                                       all_comma_tags = None,
                                       tokenizer = tokenizer, 
                                       tag2idx = None,
                                       mapping_dic = None, 
                                       device=device,
                                       batch_size = args.batch_size)

        duration = inference(dataLoader,
                             model,
                             tokenizer,
                             idx2tag,
                             args.output_text_file
                             )
        print(f'Duration_time: {duration:4f}')

        print(f'Kasreh_and_Comma for {args.input_text_file} was saved in {args.output_text_file}.')



if __name__ == '__main__':
    main()   