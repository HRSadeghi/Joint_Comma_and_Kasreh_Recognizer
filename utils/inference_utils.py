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




def return_sen_to_real_form(tokenizer, input_sen, kasreh_tags, comma_tags):
    if input_sen == '' or input_sen == ' ':
        return input_sen

    punctuations = '''!(-[{;:،'"\,<./?@#$٫+=×%^&*_~…٬»؛؟ـ'''
    out = ""
    encoded = tokenizer(input_sen)
    word_ids_tags = [(x, kasreh_tags[i], comma_tags[i]) for i,x in enumerate(encoded.word_ids(batch_index=0)) if x is not None]
    
    word_ids = [x[0] for x in word_ids_tags]
    _kasreh_tags = [x[1] for x in word_ids_tags]
    _comma_tags = [x[2] for x in word_ids_tags]
    num_words = max(sorted(list(set(word_ids)))) + 1

    for i in range(num_words):
        index_in_tag_seq = len(word_ids) - word_ids[::-1].index(i) - 1
        word_i_span = encoded.word_to_chars(0,i)

        out += input_sen[word_i_span.start:word_i_span.end]
        
        if out[-1] in punctuations:
            out += ' '
            continue

        if _kasreh_tags[index_in_tag_seq] == 'e':
            out += 'ِ '
        elif _kasreh_tags[index_in_tag_seq] == 'ye':
            if (out[-1] not in ['ه', 'ا','ی']):
                out += ' یِ '
            elif out[-1] == 'ی':
                out += 'ِ '
            else:
                out += ' '
        else:
            if _comma_tags[index_in_tag_seq] == 'C':
                out += '، '
            else:
                out += ' '

        

    return out




        
