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
    

class DecoderRNN(nn.Module):
    def __init__(
        self, 
        embed_size, 
        hidden_size, 
        vocab_size, 
        num_layers=1, 
        max_caption_length=100,
        end_word_idx=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_caption_length = max_caption_length
        self.end_word_idx = end_word_idx
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(
            embed_size, 
            hidden_size, 
            num_layers, 
            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embedded_features_and_captions = torch.cat(
            (features.unsqueeze(1),
             self.embedding(captions[:,:-1])),
            dim=1)

        output, _ = self.lstm(embedded_features_and_captions, None)
        
        return self.fc(output)
            
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        hidden_state = None
    
        caption = []
        while len(caption) < self.max_caption_length:
            output, hidden_state = self.lstm(inputs, hidden_state)
            word_vector = self.fc(output).squeeze(1)
            _, word_idx = torch.max(word_vector, dim=1)
            
            caption.append(word_idx.cpu().numpy()[0].item())
            
            if (word_idx == self.end_word_idx):
                break
            
            inputs = self.embedding(word_idx).unsqueeze(1)
            
        return caption