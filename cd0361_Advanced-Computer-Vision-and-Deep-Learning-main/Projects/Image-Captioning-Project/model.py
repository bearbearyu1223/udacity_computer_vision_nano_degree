import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        super().__init__()
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, bias=True,
                            batch_first=True, dropout=0, bidirectional=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # Discard the <end> word
        captions = captions[:, :-1]

        # Initialize the hidden state
        batch_size = features.shape[0]
        self.hidden = (torch.zeros((1, batch_size, self.hidden_size), device=device),
                       torch.zeros((1, batch_size, self.hidden_size), device=device))
        embeddings = self.word_embeddings(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        outputs = self.linear(lstm_out)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        decoder_output = []
        batch_size = inputs.shape[0]
        if states is None:
            states = (torch.zeros((1, batch_size, self.hidden_size), device=device),
                      torch.zeros((1, batch_size, self.hidden_size), device=device))

        predicted_vocab_index = -1
        step = 0

        while predicted_vocab_index != 1 and step < max_len:
            lstm_cell_output, states = self.lstm(inputs, states)
            predicted_outputs = (self.linear(lstm_cell_output)).squeeze(1)
            _, predicted_vocab_index = torch.max(predicted_outputs, dim=1)
            decoder_output.append(predicted_vocab_index.item())
            step = step + 1
            inputs = (self.word_embeddings(predicted_vocab_index)).reshape(1, 1, -1)
        return decoder_output


