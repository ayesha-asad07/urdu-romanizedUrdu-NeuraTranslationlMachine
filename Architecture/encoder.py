from Architecture.tokenizer import *

# ---- Model Architecture: Encoder / Decoder ----

class EncoderBiLSTM(nn.Module):
    
    # constructor initialization function
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers=2, dropout=0.3, pad_idx=0):

        super().__init__()

        # learned dense vector representations of our model
        # parameters: ( 51, 256, 0 [forced padded tokens' embeddings set to zero] )
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        
        # 0.3 : randomly make 30% embedding features zero in vectors
        self.emb_dropout = nn.Dropout(dropout)
        
        # 2 layers of bidirectional lstm (4 LSTMs)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        
        # input feature is concatenation of 2 directions (forward 512 + backward 512 = hid_dim*2)
        # nn.Linear(input_features , output_features)
        # 1024 dim is compressed to 512 out_features to make it compatible with decoder's input_features dim
        
        # fc_h: final hidden state (in_features = 1024, out_features = 512)
        self.fc_h = nn.Linear(hid_dim*2, hid_dim)

        # fc_c: final cell state 
        self.fc_c = nn.Linear(hid_dim*2, hid_dim)
 
    # forward pass function, src: padded_src_tensors, src_len: true_tensor_lengths
    def forward(self, src, src_len):
        
        # make 256 embed dense vectors of src tensors,and then apply dropout
        emb = self.emb_dropout(self.embedding(src))
        
        # packed: compact padded sequence/tensors into orignal lenghts to avoid learning pad(0) indices

        # parameters:
        # embedded vectors, actual lengths, out_shape[B,L,emb], shuffled seqs[sortion would be desc length based]
        packed = pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        
        # bidriectional LSTM outputs: packed_output, hidden state, cell state
        packed_out, (h, c) = self.lstm(packed)
        
        #h has four outputs ( 2 layers * 2 directions) [4, B, 512] --> last ones ar of top layers
        # unpack out would add paddings again to equalize all sequences in length
        enc_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # concatenating forward and backward output of top layer column-wise(dim=1) [B,512] + [B,512] = [B,1024]
        h_cat = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)

        c_cat = torch.cat((c[-2,:,:], c[-1,:,:]), dim=1)

        # self.fc_h compresses 1024 dim into 512
        # torch.tanh bounds the large/sparse values non-linearly around zero [-1,1]
        # unsequeze(0) will add new dimension for layer at index 0 : [B, 512] => [1, B, 512] (valid format for decder input)
        h0 = torch.tanh(self.fc_h(h_cat)).unsqueeze(0)

        c0 = torch.tanh(self.fc_c(c_cat)).unsqueeze(0)

        # enc_out padded ouput for evals like attention etc
        # h0, c0 compatible input format for decoder 
        return enc_out, (h0, c0)
