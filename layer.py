import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, keys):
        # query: [seq_len, batch_size, hidden_size]
        # keys: [seq_len, batch_size, hidden_size]

        query = query.unsqueeze(0)  # [1, batch_size, hidden_size]

        energy = torch.tanh(self.W_q(query) + self.W_k(keys))  # [seq_len, batch_size, hidden_size]

        attention_scores = self.V(energy).squeeze(-1)  # [seq_len, batch_size]
        attention_weights = torch.softmax(attention_scores, dim=0)  # [seq_len, batch_size]

        # Expand dimensions of attention_weights for broadcasting
        attention_weights_expanded = attention_weights.unsqueeze(2)  # shape: [seq_length, batch_size, 1]

        # Multiply attention weights with keys to get weighted sum
        context_vector = torch.sum(keys * attention_weights_expanded, dim=0)  # shape: [batch_size, 256]

        return context_vector, attention_weights

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.W_query = nn.Linear(hidden_size, hidden_size)
        self.W_key = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, x_t, h_prev):
        """
        Input : x_t -> [B, F]
                h_prev -> [B, H]

        (B: batch size, F: number of features, H: hidden size) 
        """

        query = self.W_query(h_prev)
        keys = self.W_key(x_t)

        # Batch matrix multiplication
        energy = self.V(torch.tanh(query + keys))
        attention_weights = F.softmax(energy, dim=1)

        context = torch.sum(x_t * attention_weights, dim=1).unsqueeze(-1)
        
        return context

class AttentionLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionLSTMCell, self).__init__()

        self.lstm_cell = torch.nn.LSTMCell(input_size+1, hidden_size)
        self.attention = Attention(input_size, hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        """
        Input : x_t -> [B, F]
                h_prev -> [B, H]

        (B: batch size, F: number of features, H: hidden size) 
        """

        context = self.attention(x_t, h_prev)

        lstm_input = torch.cat((x_t, context), dim=1)
        h_t, c_t = self.lstm_cell(lstm_input, (h_prev, c_prev))

        return h_t, c_t


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = AttentionLSTMCell(self.input_size, self.hidden_size)

    def forward(self, x, init_states=None):
        """
        Input : x -> [L, B, F]
        
        (L: sequence lenght, B: batch size, F: number of features) 
        """

        L, B, _ = x.size()

        h_t, c_t = (torch.zeros(B, self.hidden_size).to(x.device),
                    torch.zeros(B, self.hidden_size).to(x.device)) if init_states is None else init_states

        for i in range(L):

            h_t, c_t = self.cell(x[i], h_t, c_t)

        return h_t, c_t

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)