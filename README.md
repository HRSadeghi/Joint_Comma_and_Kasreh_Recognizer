# Joint_Comma_and_Kasreh_Recognizer
This package includes source code for recognizing and adding Persian Kasre and Persian Comma to text.
## Installation    
First, clone the project

```
git clone https://github.com/HRSadeghi/Joint_Comma_and_Kasreh_Recognizer.git
```
<br>
And then, install the dependencies

```
pip install -r requirements.txt
```


## Train model

To train the model, just use the following command

```
python train.py --checkpoint_dir /path/to/checkpoint_dir --n_saved_ckpts 2 --batch_size 64 --epochs 2 --no_of_bert_layer 7
```

## Inference

To use the model for one sentence, just run the following command

```
python inference.py --input_sen 'به هر حال آخرین عکس را داریم که البته دورترین است' --no_of_bert_layer 7 --checkpoint_dir /path/to/checkpoint_dir
```

But if you want to give the model a text file with one sentence per line, run the following command

```
python inference.py --input_text_file /path/to/input_file.txt --output_text_file path/to/result_file.txt --no_of_bert_layer 7 --checkpoint_dir /path/to/checkpoint_dir
```