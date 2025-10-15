from Architecture.tokenizer import *

# unidirectional 4 layer decoder
class Decoder(nn.Module):

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers=4, dropout=0.3, pad_idx=0):
        super().__init__()
        
        # learned dense vector representation
        # parameters: [32, 256, forced padded tokens' embedding to remain 0]
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)

        # randomly drop 30% embedding features to zero
        self.emb_dropout = nn.Dropout(dropout)

        # 4 layers unidirection LSTM (4 LSTMs) with hidden dimension = 512
        # emd_dim format: [B, seq_len, emb] (since decoder process 1 token at a time, seq_len = 1)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout)
        
        # Linear Function to compress 512 dim into 32 vocab_size
        # final hidden state:  in_feature dim = 512, out_feature dim = 32
        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.n_layers = n_layers

    # function for forward pass of decoder
    # input_token is 1D tensor, for Batch=64 it is one list of 64 indices [B]
    # hidden & cell are encoder's output h0 and c0, [1, B, 512] repeated for 4 layers of decoder
    def forward_step(self, input_token, hidden, cell):
        
        # learned dense vectors of input
        # embedding dim would be added as [B, emb_dim]
        # unsequeeze(1) adds time dimension at index 1, [B, 1, 256] (1 bcz decoder processes 1 seq_len at a time)
        emb = self.emb_dropout(self.embedding(input_token)).unsqueeze(1)

        # emb: [B, seq_len, emb_dim] = [64, 1, 256]
        # hidden, cell: [[num_layers, batch, dim] = [4, 64, 512]
        # input: [B, 1, emb] ---> output [B, 1, hid_dim]
        output, (hidden, cell) = self.lstm(emb, (hidden, cell))

        # sequeeze(1): seq_len index is removed from output
        # [B, 1, hid_dim] --> [B, hid_dim]
        # fc_out compresses hid_dim to output_dim: [B, 512] --> [B, 32] (32 is target vocab size)
        pred = self.fc_out(output.squeeze(1))

        # retuen predictions [B, output_dim] and final hidden states [4, B, 512]
        return pred, hidden, cell

