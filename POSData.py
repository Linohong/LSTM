import torch
from torch.autograd import Variable

def prepare_sequence(seq, to_ix) :
    idxs = [to_ix[w] for w in seq] # getting indices of the seq. 
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)

training_data = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_idx = {}
for sent, tags in training_data : 
    for word in sent : 
        if word not in word_to_idx : 
            word_to_idx[word] = len(word_to_idx)

tag_to_idx = {"DET":0, "NN":1, "V":2}

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

print('Data Preparation Over.')
