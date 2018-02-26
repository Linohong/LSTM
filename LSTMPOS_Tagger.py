import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LSTMTagger (nn.Module) :
    def __init__ (self, embedding_dim, hidden_dim, vocab_size, tagset_size ) :
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The Linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self) :
        # Before we've done anything, we don't have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # Why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # return tuple : h0, c0
        # h0 : initial hidden state for each element in the batch (as a tensor)
        # c0 : initial cell state for each element in the batch (as a tensor)
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence) :
        embeds = self.word_embeddings(sentence) # len(sentence) * embedding_dim (6)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden) # embeds.view = 5 (sentence length) of (1 * 6)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1)) # tag_space = I guess 5x3
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
