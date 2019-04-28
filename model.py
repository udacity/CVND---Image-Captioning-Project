import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from vocabulary import Vocabulary

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
    #def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):

        #super(DecoderRNN, self).__init__()
        #self.hidden_dim = hidden_size
        #self.embed = nn.Embedding(vocab_size, embed_size)
        #self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        #self.fc = nn.Linear(hidden_size, vocab_size)
        #self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size))
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        ## TODO: define the LSTM
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        ## TODO: define the final, fully-connected output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # initialize the weights
        #self.init_weights()
        #self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size))
        self.softmax = nn.Softmax(dim=2)
        
    def gforward(self, features, captions):
        
        try:
            caption_emb = self.embed(captions[:,:-1])
            embeddings = torch.cat((features.unsqueeze(1), caption_emb), 1)
            print('hi')
        except:
            embeddings = features
            print('ho')
        print(embeddings.shape)
        lstm_out, self.hidden = self.lstm(embeddings)
        
        output = self.fc(lstm_out)

        return output

    def gsample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        #feature_vec = EncoderCNN.forward(inputs)
        out_list = []
        for i in range(max_len):
            print('iteration', i)
            output, states = self.lstm(inputs, states)
            #inputs_act = self.hidden
            outputs = self.fc(output.squeeze(1))
            target_index = outputs.max(1)[1]
            print(target_index) 
            out_list.append(target_index.item())
            inputs = self.embed(target_index).unsqueeze(1)
        return out_list
    
        
    def forward(self, features, captions):

        cap_embedding = self.embed(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), 1)              
        lstm_out, self.hidden = self.lstm(embeddings, None)
        outputs = self.fc(lstm_out)

        return outputs
   

    def sample(self, inputs, states=None, max_len=20):

        outlist = []

        for i in range(max_len):

            outputs, states = self.lstm(inputs, states)
            #print(outputs.squeeze(1).shape)
            outputs = self.fc(outputs.squeeze(1))
            #print(outputs.max(1)[1])
            target_index = outputs.max(1)[1]
            outlist.append(target_index.item())
            inputs = self.embed(target_index).unsqueeze(1)

        return outlist
    
    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1
        
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)