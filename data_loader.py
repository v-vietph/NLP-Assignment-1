from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd 
import gc
import os

class WordEmbeddingDataset(Dataset):

    def __init__(self, txt_file, dimension=50):
        self.file_path = txt_file 
        self.content = []
        print("Dataset loading")
        with open(self.file_path, 'r', encoding='ISO-8859-1') as reader:
            self.content = reader.read().split('\n')
        self.dimension = dimension
        self.vocab_dict = self.load_glove()

    def load_glove(self):
        glove_path = './glove/glove.6B.'+str(self.dimension)+'d.txt'
        if not os.path.exists(glove_path):
            raise ValueError ('Can not find the glove file for dimension of', str(self.dimension))  
        if self.dimension == 50:
            self.unknown_vector = np.array([-0.12920076,-0.28866628,-0.01224866,-0.05676644,-0.20210965,-0.08389011
                    ,0.33359843,0.16045167,0.03867431,0.17833012,0.04696583,-0.00285802
                    ,0.29099807,0.04613704,-0.20923874,-0.06613114,-0.06822549,0.07665912
                    ,0.3134014,0.17848536,-0.1225775,-0.09916984,-0.07495987,0.06413227
                    ,0.14441176,0.60894334,0.17463093,0.05335403,-0.01273871,0.03474107
                    ,-0.8123879,-0.04688699,0.20193407,0.2031118,-0.03935686,0.06967544
                    ,-0.01553638,-0.03405238,-0.06528071,0.12250231,0.13991883,-0.17446303
                    ,-0.08011883,0.0849521,-0.01041659,-0.13705009,0.20127155,0.10069408
                    ,0.00653003,0.01685157])
        else:
            self.unknown_vector = np.random.rand(self.dimension,)
        frame = pd.read_csv(glove_path, sep=" ", quoting=3, header=None, index_col=0)
        return {key: val.values for key, val in frame.T.items()}

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        line = self.content[idx]
        line = line.split()
        embedding_tensors = self.word2tensor(line[1:-2])
        # print("line", line)
        if len(line) == 0:
            return embedding_tensors, -1
        if(line[-1] == '#neg'):
            label = torch.tensor(0)
        else:
            label = torch.tensor(1)
        return embedding_tensors, label

    def word2tensor(self, word_list):
        embeddings = []
        for word in word_list:
            try:
                embeddings.append(self.vocab_dict[word].astype(np.float32))
            except KeyError:
                embeddings.append(self.unknown_vector.astype(np.float32))
        tensors = torch.tensor(embeddings)
        embeddings = None
        return tensors
