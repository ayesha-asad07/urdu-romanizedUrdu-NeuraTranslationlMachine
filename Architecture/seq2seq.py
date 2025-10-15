from Architecture.decoder import *
from Architecture.encoder import *

# Seq2Seq class ( connect encoder and decoder for a complete system output)
class Seq2Seq(nn.Module):
    
    # take initialized encoder and decoder
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    # forward pass function
    # encodes src sequence/tensor
    # initializes decoder's h & c with enc_out h0, c0
    # send input token to decoder and get next predicted token as output (runs in loop for one seq/trg_len)
    # if trg is none, max_len is used as trg_len for loop
    def forward(self, src, src_len, trg=None, teacher_forcing=0.5, max_len=50):

        # src = [B, src_lens] size(0) is batchsize
        bsz = src.size(0)
        
        # trg = [B, trg_lens]
        trg_len = trg.size(1) if trg is not None else max_len

        # decoder.fc_out(in, out) , out_dim = vocab size
        vocab_size = self.decoder.fc_out.out_features
        
        # output [B, trg_len, vocab_size] is initialized/filled with zeroes
        outputs = torch.zeros(bsz, trg_len, vocab_size, device=self.device)

        # src, src_lens are passed to encoder forward pass for dense context vector output
        _, (h_enc, c_enc) = self.encoder(src, src_len)
        
        # since h_enc is [1, B, hid_dim] , but decoder needs [4, B, hid_dim]
        # h_enc is duplicated acrosso 4 layer of decoder
        # repeat: h_enc becomes [4, B, hid_dim]
        if h_enc.size(0) != self.decoder.n_layers:
            h = h_enc.repeat(self.decoder.n_layers, 1, 1).contiguous()
            c = c_enc.repeat(self.decoder.n_layers, 1, 1).contiguous()
        else:
            h, c = h_enc, c_enc
        
        # input_token is a 1D tensor 
        # if trg, then first column from [B, trg_lens]
        # otherwise, [B] list filled with sos_idx
        input_token = trg[:, 0] if trg is not None else torch.full(
            (bsz,), SOS_IDX, dtype=torch.long, device=self.device)

        # loop for trg sequences
        for t in range(1, trg_len):
            
            # input_token is [B] , h & c are [4, B, 512]
            # pred would be [B, output_dim] (predictions), h & c of [4, B, 512]
            pred, h, c = self.decoder.forward_step(input_token, h, c)

            # pred is [B, vocab_size]   (output_dim = target vocab_size)
            # output is [B, trg_len, vocab_size]
            outputs[:, t, :] = pred
            
            # if trg, pick grind truth randonmly on torch.rand
            if trg is not None:
                if torch.rand(1).item() < teacher_forcing:
                    input_token = trg[:, t]
                else:
                    input_token = pred.argmax(1)
            else:
                input_token = pred.argmax(1)

            # Assume B is 3, vocab_size is 5, pred would be 3x5 matrix [bsz, vocba_size]:

            # [ [2.1, 0.3, 1.7, 0.9, 0.2],
            #   [0.1, 4.5, 0.2, 1.3, 0.7],
            #   [3.0, 2.8, 0.4, 0.5, 1.2] ]

            # argmax(1) will pick highest index along dimension 1 (vocab_size), (means highest index from each row) 
            # which is index 0 in 1st row, index 1 in 2nd row, index 0 in 3rd row
            # argmax returns [0, 1, 0]   1D tensor of [B] to fed into decoder as next input_token

        # outputs is [B, trg_len, vocab_size]
        return outputs


