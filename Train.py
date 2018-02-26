import LSTMPOS_Tagger as Network
import POSData as D
import torch.nn as nn
import torch.optim as optim

model = Network.LSTMTagger(D.EMBEDDING_DIM, D.HIDDEN_DIM, len(D.word_to_idx), len(D.tag_to_idx))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
inputs = D.prepare_sequence(D.training_data[0][0], D.word_to_idx)
tag_scores = model(inputs)

for epoch in range(300) : # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in D.training_data :
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        sentence_in = D.prepare_sequence(sentence, D.word_to_idx)
        targets = D.prepare_sequence(tags, D.tag_to_idx)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
inputs = D.prepare_sequence(D.training_data[0][0], D.word_to_idx)
tag_scores = model(inputs)
# "The dog ate the apple."
# 0 : DET
# 1 : NOUN
# 2 : VERB
print(tag_scores)