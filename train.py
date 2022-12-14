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


from utils.data_preprocessing import prepare_dataset_for_train
from utils.training_utils import load_pretrained_bert_model, get_device, train_step, evaluate
from utils.tag_mapping import get_tag2idx_idx2tag_dics, mapping_dic
from models.Joint_BERT_BiLSTM import JointBERTBiLSTMTagger
from data_loader.loader import Kasreh_DataLoader
from handlers.checkpoint_handler import save_checkpoint, load_checkpoint
from torchmetrics import MeanMetric
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
import time
import argparse



def train(model, 
          train_dataLoader, 
          val_dataLoader, 
          optimizer, 
          loss_object,
          epochs,
          checkpoint_dir,
          n_saved_ckpts
          ):
    for epoch in range(epochs):
        start = time.time()

        kasreh_train_loss = MeanMetric()
        comma_train_loss = MeanMetric()
        kasreh_train_accuracy = MeanMetric()
        comma_train_accuracy = MeanMetric()

        for batch, (input, (kasreh_tags, comma_tags)) in enumerate(train_dataLoader):
            train_step(model, 
                       input,
                       kasreh_tags,
                       comma_tags,
                       optimizer,
                       loss_object,
                       kasreh_train_loss = kasreh_train_loss,
                       comma_train_loss = comma_train_loss,
                       kasreh_train_accuracy = kasreh_train_accuracy,
                       comma_train_accuracy = comma_train_accuracy
                    )

            if batch % 100 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Kasreh_train_loss {kasreh_train_loss.compute().cpu().item():.4f} Comma_train_loss {comma_train_loss.compute().cpu().item():.4f} Kasreh_train_accuracy {kasreh_train_accuracy.compute().cpu().item():.4f} Comma_train_accuracy {comma_train_accuracy.compute().cpu().item():.4f}')

        kasreh_val_loss, comma_val_loss, kasreh_val_acc, comma_val_acc  = evaluate(val_dataLoader, model, loss_object)


        kasreh_t_loss = kasreh_train_loss.compute().cpu().item()
        comma_t_loss = comma_train_loss.compute().cpu().item()
        kasreh_t_acc = kasreh_train_accuracy.compute().cpu().item()
        comma_t_acc = comma_train_accuracy.compute().cpu().item()

        print(f'Epoch {epoch + 1} Batch {batch} Kasreh_train_loss {kasreh_t_loss:.4f} Comma_train_loss {comma_t_loss:.4f} Kasreh_train_accuracy {kasreh_t_acc:.4f} Comma_train_accuracy {comma_t_acc:.4f}   Kasreh_val_loss {kasreh_val_loss:.4f} Comma_val_loss {comma_val_loss:.4f} Kasreh_val_accuracy {kasreh_val_acc:.4f} Comma_val_accuracy {comma_val_acc:.4f}')
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
        if (epoch + 1) % 1 == 0:
            #ckpt_save_path = ckpt_manager.save()
            to_save = {'epoch': epoch + 1,
                       'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'kasreh_train_loss': round(kasreh_t_loss, 4),
                       'comma_train_loss': round(comma_t_loss, 4),
                       'kasreh_train_accuracy': round(kasreh_t_acc, 4),
                       'comma_train_accuracy': round(comma_t_acc, 4),
                       'kasreh_val_loss': round(kasreh_val_loss, 4),
                       'comma_val_loss': round(comma_val_loss, 4),
                       'val_loss':round((kasreh_val_loss + comma_val_loss)/2, 4),
                       'kasreh_val_accuracy': round(kasreh_val_acc, 4),
                       'comma_val_accuracy': round(comma_val_acc, 4),
                       'val_accuracy': round((kasreh_val_acc + comma_val_acc)/2, 4)
                }

            save_checkpoint(base_directory_path = checkpoint_dir, 
                            to_save = to_save,
                            score_name = 'val_accuracy',
                            n_saved = n_saved_ckpts,
                            filename_prefix = 'best',
                            ext = 'pt'
                        )





def main():
    parser = argparse.ArgumentParser(description='Create a train command.')
    parser.add_argument('--train_file_path',
                        type=str, 
                        default='dataset/train_data.txt',
                        help='path to the train_data.txt file')
    parser.add_argument('--test_file_path', 
                        type=str,
                        default='dataset/test_data.txt',
                        help='path to the test_data.txt file')
    parser.add_argument('--valid_file_path', 
                        type=str,
                        default='',
                        help='path to the valid_data.txt file')
    parser.add_argument('--checkpoint_dir', 
                        type=str,
                        default='saved_checkpoints',
                        help='path to the checkpoint directory. The checkpoints will be saved here')
    parser.add_argument('--load_checkpoint_dir', 
                        type=str,
                        default='',
                        help='path to the current checkpoint directory to load pretrained weights')
    parser.add_argument('--n_saved_ckpts', 
                        type=int,
                        default=3,
                        help='number of saved models in checkpoint directory')
    parser.add_argument('--batch_size', 
                        type=int,
                        default=64,
                        help='path to the valid_data.txt file')

    parser.add_argument('--epochs', 
                        type=int,
                        default=2,
                        help='path to the valid_data.txt file')


    parser.add_argument('--valid_size', 
                        type=float,
                        default=0.1,
                        help='A float number between 0 and 1 for splitting validation sample from training sample.')

    parser.add_argument('--Pretrained_BERT_model_name', 
                        type=str,
                        default='HooshvareLab/bert-fa-zwnj-base',
                        help='The name of pretrained BERT model or a path to pretrained BERT model')

    parser.add_argument('--no_of_bert_layer', 
                        type=int,
                        default=7,
                        help='Number of bert layers that is used in new model')
    args = parser.parse_args()



    print('Preparing training dataset ...')
    train_sens, kasreh_train_tags, comma_train_tags = prepare_dataset_for_train(args.train_file_path)
    print('Preparing test dataset ...')    
    test_sens, kasreh_test_tags, comma_test_tags = prepare_dataset_for_train(args.test_file_path)

    print('Preparing validation dataset ...')   
    if args.test_file_path != '':
        train_sens, val_sens, kasreh_train_tags, kasreh_val_tags, comma_train_tags, comma_val_tags = train_test_split(train_sens, kasreh_train_tags, comma_train_tags, test_size=args.valid_size, random_state=42)
    else:
        val_sens, kasreh_val_tags, comma_val_tags = prepare_dataset_for_train(args.valid_size)

    device = get_device()
    tag2idx, idx2tag = get_tag2idx_idx2tag_dics()

    print('Preparing dataloaders ...')   
    tokenizer, bert_model = load_pretrained_bert_model(model_name = args.Pretrained_BERT_model_name)



    train_dataLoader = Kasreh_DataLoader(train_sens, 
                           kasreh_train_tags,
                           comma_train_tags,
                           tokenizer = tokenizer, 
                           tag2idx = tag2idx,
                           mapping_dic = mapping_dic, 
                           device=device,
                           batch_size = args.batch_size)


    val_dataLoader = Kasreh_DataLoader(val_sens, 
                            kasreh_val_tags,
                            comma_val_tags,
                            tokenizer = tokenizer, 
                            tag2idx = tag2idx,
                            mapping_dic = mapping_dic, 
                            device=device,
                            batch_size = args.batch_size)


    test_dataLoader = Kasreh_DataLoader(test_sens, 
                            kasreh_test_tags,
                            comma_test_tags,
                            tokenizer = tokenizer, 
                            tag2idx = tag2idx,
                            mapping_dic = mapping_dic, 
                            device=device,
                            batch_size = args.batch_size)

    print('Creating BERT BiLSTM model ...')   
    model = JointBERTBiLSTMTagger(bert_model = bert_model, no_of_bert_layer = args.no_of_bert_layer)
    model = model.to(device)

    
    loss_object = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    if args.load_checkpoint_dir != '':
        print('Loading model weights ...')   
        to_load={
                'model_state_dict': model,
                'optimizer_state_dict': optimizer
                }

        load_checkpoint(args.load_checkpoint_dir, to_load)


    print('Starting to train model ...')  
    train(model, 
          train_dataLoader, 
          val_dataLoader, 
          optimizer, 
          loss_object,
          args.epochs,
          args.checkpoint_dir,
          args.n_saved_ckpts
          )






if __name__ == '__main__':
    main()   