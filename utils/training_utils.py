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
import torch.nn as nn
import torch.nn.functional as F
import time
from torchmetrics import MeanMetric
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertConfig






def accuracy_function(input, real_tar, pred):
    accuracies = torch.eq(real_tar, torch.argmax(pred, dim=2))

    mask = torch.logical_not(torch.eq(input['attention_mask'], 0))
    accuracies = torch.torch.logical_and(mask, accuracies)

    accuracies = accuracies.type(torch.FloatTensor)
    mask = mask.type(torch.FloatTensor)
    return torch.sum(accuracies)/torch.sum(mask)


def loss_function(input, 
                  real, 
                  pred, 
                  loss_object,
                  weights_for_labels = None
                  ):
  mask = torch.logical_not(torch.eq(input['attention_mask'], 0))
  loss_ = loss_object(pred, real)

  mask = mask.type(loss_.type())
  loss_ *= mask

  # Dedicating weights to classes
  if weights_for_labels is not None:
    weights_for_labels = torch.Tensor(weights_for_labels).to(mask.device.type)
    real_one_hot = torch.nn.functional.one_hot(real, len(weights_for_labels))

    weights_for_tokens = torch.sum(weights_for_labels * real_one_hot, -1)

    return torch.sum(loss_*weights_for_tokens)/torch.sum(mask*weights_for_tokens)
  ##

  return torch.sum(loss_)/torch.sum(mask)




def train_step(model,
               input,
               kasreh_tags, 
               comma_tags,
               optimizer, 
               loss_object, 
               kasreh_train_loss = None, 
               comma_train_loss = None, 
               kasreh_train_accuracy = None,
               comma_train_accuracy = None):
    # Step 1. Remember that Pytorch accumulates gradients.
    # We need to clear them out before each instance
    model.zero_grad()

    # Step 3. Run our forward pass.
    kasreh_tag_scores, comma_tag_scores = model(input)
    kasreh_tag_scores = torch.permute(kasreh_tag_scores, [0,2,1])
    comma_tag_scores = torch.permute(comma_tag_scores, [0,2,1])


    # Step 4. Compute the loss, gradients, and update the parameters by
    #  calling optimizer.step()
    kasreh_loss = loss_function(input, 
                         kasreh_tags, 
                         kasreh_tag_scores, 
                         loss_object = loss_object)

    comma_loss = loss_function(input, 
                         comma_tags,
                         comma_tag_scores,
                         loss_object = loss_object,
                         weights_for_labels = [0.1, 0.85])


    total_loss = kasreh_loss + comma_loss

    kasreh_acc = accuracy_function(input, kasreh_tags, torch.permute(kasreh_tag_scores, [0,2,1]))
    comma_acc = accuracy_function(input, comma_tags, torch.permute(comma_tag_scores, [0,2,1]))

    if kasreh_train_loss is not None:
        kasreh_train_loss.update(kasreh_loss.cpu().item())
    if comma_train_loss is not None:
        comma_train_loss.update(comma_loss.cpu().item())
    if kasreh_train_accuracy is not None:
        kasreh_train_accuracy.update(kasreh_acc.cpu().item())
    if comma_train_accuracy is not None:
        comma_train_accuracy.update(comma_acc.cpu().item())

    total_loss.backward()
    optimizer.step()



def evaluate(dataLoader, 
             model,
             loss_object
             ): 

    _kasreh_loss = MeanMetric()
    _comma_loss = MeanMetric()
    _kasreh_accuracy = MeanMetric()
    _comma_accuracy = MeanMetric()
    for batch, (input, (kasreh_tags, comma_tags)) in enumerate(dataLoader):
        with torch.no_grad():
            kasreh_tag_scores, comma_tag_scores = model(input)

            kasreh_tag_scores = torch.permute(kasreh_tag_scores, [0,2,1])
            kasreh_loss = loss_function(input, kasreh_tags, kasreh_tag_scores, loss_object)
            kasreh_acc = accuracy_function(input, kasreh_tags, torch.permute(kasreh_tag_scores, [0,2,1]))

            comma_tag_scores = torch.permute(comma_tag_scores, [0,2,1])
            comma_loss = loss_function(input, comma_tags, comma_tag_scores, loss_object)
            comma_acc = accuracy_function(input, comma_tags, torch.permute(comma_tag_scores, [0,2,1]))

            _kasreh_loss.update(kasreh_loss.cpu().item())
            _comma_loss.update(comma_loss.cpu().item())
            _kasreh_accuracy.update(kasreh_acc.cpu().item())
            _comma_accuracy.update(comma_acc.cpu().item())

    mean_kasreh_loss = _kasreh_loss.compute().cpu().item()
    mean_comma_loss = _comma_loss.compute().cpu().item()
    mean_kasreh_acc = _kasreh_accuracy.compute().cpu().item()
    mean_comma_acc = _comma_accuracy.compute().cpu().item()

    return mean_kasreh_loss, mean_comma_loss, mean_kasreh_acc, mean_comma_acc



def load_pretrained_bert_model(model_name = 'HooshvareLab/bert-fa-zwnj-base', 
                               output_hidden_states = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name,
                                        output_hidden_states=output_hidden_states)
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)

    return tokenizer, model



def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
