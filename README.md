# NLP Assigment for RNN, LSTM and GRU
Author: vietph
This repo using pytorch using pretrained weights from Glove

## How to run:
1. Install dependences:

    ``` pip install -r requirements.txt```

2. Download the Glove pretrained weights from this drive:

    https://drive.google.com/drive/folders/1u_FuIAfD5Lh8XB337xX8Yy-WTMnhcYlK?usp=sharing

    and replace the `./glove` with it.
3. To run the training:

    ``` python train.py --train_data train/data/dir --model_to_train <RNN/LSTM/GRU> --n_epochs 20 --learning_rate 0.001 --hidden_size 50 --n_epochs 30 --pretrained_size 50 ```

4. To get the testing results:

    ``` python test.py --model_to_test RNN --hidden_size 50 --pretrained_size 50 ```

## Performances on testing dataset:
The performances are recorded on 3 models (RNN|GRU|LSTM) with 3 setting (50d - 50g|100d - 100g|150d - 50g)

    d: number of hidden size
    g: number of Glove size

|   	        |   RNN	    |   GRU	    |   LSTM	|
|--:	        |--:	    |--:	    |--:	    |
|   50d - 50g	|   0.5921	|   0.6279	|   0.6336	|
|   100d - 100g	|   0.6371	|   0.6386	|   0.7243	|
|   150d - 50g 	|   0.6543	|   0.6779	|   0.735	|

## Inference:
Run 

```python inference.py```

There are 3 setting:

1. 50 hidden_size and 50 Glove size
2. 100 hidden_size and 100 Glove size
3. 150 hidden_size and 50 Glove size

Example run:
```
>>> Choose your model <RNN|LSTM|GRU>: RNN 
>>> Choose the hidden_size <50|100|150>: 100
>>> Choose the glove_size <50|100>:100
Model RNN_model_100_100.pt loaded
Glove loaded
>>> Input sequence or ctrl+d to finish: 
a taut , intelligent psychological drama
Positive
```

