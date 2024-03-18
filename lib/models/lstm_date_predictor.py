import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (batch_size, seq_len, hidden_dim)
        # decoder_hidden: (batch_size, hidden_dim)

        # Calculate the attention scores.
        scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)  # (batch_size, seq_len)

        attn_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)

        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_dim)

        return context_vector, attn_weights


class LSTMDate(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 output_size,
                 query_size):
        super(LSTMDate, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc_q = nn.Linear(query_size, hidden_size)

    def forward(self, x, q):
        out, _ = self.lstm(x)
        q_enc = self.fc_q(q)

        out = self.fc(out[:, -1, :])
        return out
