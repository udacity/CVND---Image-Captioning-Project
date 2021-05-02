import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np


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


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        self.num_layes = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        captions = self.embed(captions[:, :-1])
        features = features.unsqueeze(1)

        x = torch.cat((features, captions), 1)
        x, (h, c) = self.lstm(x)
        x = self.fc(x)
        return x

    def sample(self, inputs, states=None, max_len=20):

        if (states is None):
            states = self.init_states(1)

        sentence = []
        x = inputs
        cuda_flag = x.is_cuda

        for i in range(max_len):
            x, states = self.lstm(x, states)
            out = self.fc(x)
            p = F.softmax(out, dim=2).data  # probability distribution of words

            p = p.cpu()
            word_indices = np.arange(self.vocab_size)
            p = p.detach().numpy().squeeze()
            word = np.random.choice(word_indices, p=p / p.sum())

            # Create an embedding from the next word
            x = torch.from_numpy(np.array([word])).long()

            if (cuda_flag):
                x = x.cuda()

            x = self.embed(x).unsqueeze(1)

            sentence.append(int(word))

        return sentence

    def init_states(self, seq_len):
        weight = next(self.parameters()).data
        return (weight.new(self.num_layes, seq_len, self.hidden_size).zero_(),
                weight.new(self.num_layes, seq_len, self.hidden_size).zero_())
