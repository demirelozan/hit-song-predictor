import torch
import torch.nn as nn
from transformers import BertModel
import numpy as np

# Define the device (use GPU if available, otherwise use CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BERTLyricalModel(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, output_dim, n_layers, bidirectional, dropout, linear_hidden):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name).to(device)
        embedding_dim = self.bert.config.hidden_size

        self.gru = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.linear = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, linear_hidden)
        self.leaky_relu = nn.LeakyReLU()
        self.out = nn.Linear(linear_hidden, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.initialize_weights()

    def forward(self, text, attention_masks):
        with torch.no_grad():
            embedded = self.bert(input_ids=text, attention_mask=attention_masks)[0]

        _, hidden = self.gru(embedded)

        if self.gru.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        linear_hidden = self.leaky_relu(self.linear(hidden))
        output = self.out(linear_hidden)

        return output

    def initialize_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name and param.requires_grad:
                nn.init.xavier_normal_(param)


# Custom Loss Function
class MARELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, predicted_logits, y):
        yhat = torch.sigmoid(predicted_logits)
        return torch.mean(torch.sqrt(torch.abs(yhat - y) + self.eps))


def batch_accuracy(predicted_logits, labels, threshold=0.5):
    preds = torch.sigmoid(predicted_logits)
    preds = preds.detach().cpu().numpy()
    labels = labels.to('cpu').numpy()
    correct = np.isclose(preds, labels, atol=threshold)
    return np.mean(correct)

    # Training function


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        # Assuming each batch is a tuple of (inputs, attention masks, labels)
        inputs, masks, labels = batch
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        predictions = model(inputs, masks).squeeze(1)
        loss = criterion(predictions, labels)

        acc = batch_accuracy(predictions, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

    # Evaluation function


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            inputs, masks, labels = batch
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

            predictions = model(inputs, masks).squeeze(1)
            loss = criterion(predictions, labels)

            acc = batch_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

    # Function to calculate time of each epoch (moved outside of ModelEvaluation)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
