import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
'''
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        pass
    
    def forward(self, features, captions):
        pass

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass
'''

#https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.2):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size) ## outputsize, hidden_size
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True) ## embedding_dim, hidden_dim 
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        """Decode image feature vectors and generates captions."""
        features = features.view(len(features), 1, -1)
        embeddings = self.embed(captions[:, :-1]) ## remove the last `word`

        inputs = torch.cat((features, embeddings), 1)
        ltsm_out, hidden = self.lstm(inputs)
        ltsm_out = self.dropout(ltsm_out)
        out = self.linear(ltsm_out)
        return out
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        result = []
        for _ in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            out = self.linear(lstm_out.view(len(lstm_out), -1))
            idx = out.max(1)[1]
            result.append(int(idx.item()))
            inputs = self.embed(idx)
            inputs = inputs.unsqueeze(1)
        return result