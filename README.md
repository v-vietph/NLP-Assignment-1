# NLP Assigment for RNN, LSTM and GRU
Author: vietph
This repo using pytorch using pretrained weights from Glove

## How to run:
1. Install dependences:

    ``` pip install -r requirements.txt```

2. Download the Glove pretrained weights from this drive:
    https://drive.google.com/file/d/1ZJMqHGB6wdHfdD59-zKs5ZQyg3OLZLxo/view?usp=sharing

    and put it in the `./glove` folder
3. To run the training:

    ``` python train.py --train_data train/data/dir --model_to_train <RNN/LSTM/GRU> --n_epochs 20 --learning_rate 0.001```

4. To get the testing results:

    ``` python test.py --test_data train/data/dir --model_to_test <RNN/LSTM/GRU> ```

