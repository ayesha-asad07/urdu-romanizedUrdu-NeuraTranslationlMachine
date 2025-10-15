import json
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import sacrebleu
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


# Special token indices (consistent with vocab files)
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


# ------- Load vocabularies

with open("source_vocab.json", "r", encoding="utf-8") as f:
    src_vocab = json.load(f)["char_to_idx"]
with open("target_vocab.json", "r", encoding="utf-8") as f:
    trg_vocab = json.load(f)["char_to_idx"]


'''
text:     "پیارخوبصورت ہے"

IDs:      "<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3,
           "پ":11, "ی":10, "ا":19, "ر":13, "خ":12, "و":18,
           "ب":14, "س":9, "ت":16, " ":8, "ہ":15, "ے":17

output from text_to_ids function will be: 

[1, 11, 10, 19, 13, 12, 18, 14, 9, 18, 13, 16, 8, 15, 17, 2]   (one tensor)
           
'''

def text_to_ids(text, vocab):  
    return [SOS_IDX] + [vocab.get(ch, UNK_IDX) for ch in text] + [EOS_IDX]

class UrduRomanDataset(Dataset):
    def __init__(self, csv_path, src_vocab, trg_vocab):
        df = pd.read_csv(csv_path)
        self.urdu_texts = df['urdu'].astype(str).tolist()
        self.roman_texts = df['roman'].astype(str).tolist()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.urdu_texts)

    #return tensors for all sentences 
    def __getitem__(self, idx):
        src_text = self.urdu_texts[idx]
        trg_text = self.roman_texts[idx]
        src_ids = torch.tensor(text_to_ids(src_text, self.src_vocab), dtype=torch.long)
        trg_ids = torch.tensor(text_to_ids(trg_text, self.trg_vocab), dtype=torch.long)
        return src_ids, trg_ids

# ---------------------------
def collate_fn(batch):
    srcs, trgs = zip(*batch)

    #actual lengths of tensors in each batch  
    #say: batch size is 3, (3 sentences/tensors in one batch) 
    #length of each sentence: src_lens = [16,10,11]  (16 is longest tensor in this batch)
    src_lens = torch.tensor([len(s) for s in srcs])  
    trg_lens = torch.tensor([len(t) for t in trgs])
    
    #add zeros on right side in short tensors to equa length with longest tensor of that batch
    #output shape": [batch size, max_length] , PAD_IDX=0 (0 wil be added for padding)
    src_pad = pad_sequence(srcs, batch_first=True, padding_value=PAD_IDX)
    trg_pad = pad_sequence(trgs, batch_first=True, padding_value=PAD_IDX)


    return src_pad, src_lens, trg_pad, trg_lens

'''
example return value of collate_fn function for batch size 3: 

src_pad = [
 [1, 11, 10, 19, 13, 12, 18, 14, 9, 18, 13, 16, 8, 15, 17, 2],  this one is longest (16 tokens)
 [1, 11, 10, 19, 13,  2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],         shorter (6 tokens)
 [1, 12, 18, 14, 9,  2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],          shorter (6 tokens)
]
src_lens = [16, 6, 6]
output shape = [3, 16]  ; batchsize, max_length
'''

# ---------------------------
# Load Datasets
full_ds = UrduRomanDataset("roman_urdu.csv", src_vocab, trg_vocab)

# split dataset manually (80%, 10%, 10%)
train_size = int(0.8 * len(full_ds))
val_size = int(0.1 * len(full_ds))
test_size = len(full_ds) - train_size - val_size

train_ds, val_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, val_size, test_size])

# padded 64 batch size datasets
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

# ---------------------------
# Loss Function
# do not compute loss for pad tokens, as they are not part of actual data (ignore pad_idx)
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)


# ---------------------------

# unique chaarcters/tokens in input: 51 
# will be used for embedding to map tokens into vectors: 
# capture what context these 51 tokens have for each token of tensor
INPUT_DIM = len(src_vocab)    

# unique chaarcters/tokens in output: 32
OUTPUT_DIM = len(trg_vocab)  

print("SRC vocab size:", INPUT_DIM)
print("TRG vocab size:", OUTPUT_DIM)

# for decoding tensors' tokens back to characters
src_id2tok = {str(v): k for k, v in src_vocab.items()}
tgt_id2tok = {str(v): k for k, v in trg_vocab.items()}


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

#------------------------------
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


#------------------------------
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



# ------------------------
# helper functions

# function to convert ids back to characters 
def ids_to_string(ids, id2tok):
    toks = []
    for idx in ids:
        if idx == PAD_IDX: 
            continue
        if idx == SOS_IDX: 
            continue
        if idx == EOS_IDX: 
            break
        toks.append(id2tok.get(str(idx), "<UNK>"))
    s = "".join(toks).replace(" ", " ").strip()
    return s

# function to calculate levenshtein distance between original target & predicted target strings
# levenshtein distance measures no.of edits (insertions, deletions, substitutions) [used of CER metric]
def levenshtein(a,b):

    n,m = len(a), len(b)
    if n==0: return m
    dp = list(range(m+1))

    for i in range(1,n+1):
        prev, dp[0] = dp[0], i
        for j in range(1,m+1):
            cur = min(dp[j] + 1, prev + (a[i-1] != b[j-1]), dp[j-1] + 1)
            prev, dp[j] = dp[j], cur
    return dp[m]

# metrics to measure model's performance
def compute_metrics(preds, refs, val_loss):

    # bleu: n-gram overlap score
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score if len(preds)>0 else 0.0

    # cer: character error rate gives normalized no. of edits 
    cers = [levenshtein(p,r)/max(1,len(r)) for p,r in zip(preds, refs)] if len(preds)>0 else [1.0]
    cer = sum(cers)/len(cers)

    # perplexity monitors model's peformance by taking exponential of avg-loss(negative log likelihood)
    try:
        ppl = math.exp(val_loss)
    except OverflowError:
        ppl = float('inf')

    return bleu, cer, ppl

# ---- training function (AMP: automatic mixed precision (float16 & float32)) ----
# parameters: (seq2seq model obj, dataloader, AdamW optim, CE criterion, GradScaler, OneCycleLR scheduler) 
def train_epoch(model, loader, optimizer, criterion, scaler, tf_ratio, scheduler=None):
    
    # model's training mode (activates dropout, grads)
    model.train()
    total_loss = 0.0
    
    # loop for one batch training
    for src, src_len, trg, _ in tqdm(loader, desc="train", leave=False):
        
        # tensos are stored on device CPU/GPU
        src, src_len, trg = src.to(device), src_len.to(device), trg.to(device)
        
        # set gradients to zero (re-initialize to remove stale values)
        optimizer.zero_grad()

        # autocast: pytorch itself would handle AMP to speed up process
        with autocast():
            
            # ** Forward pass **
            # calls seq2seq forward pass function
            # outputs is [B, trg_len, vocab_size]
            outputs = model(src, src_len, trg, teacher_forcing=tf_ratio)

            # get last dim(vocab_size) as output dim
            out_dim = outputs.shape[-1]

            # store predicted out_len and trg_len from 2nd dim 
            out_len = outputs.size(1)

            # trg is [B, trg_len]
            trg_len = trg.size(1)

            min_len = min(out_len, trg_len)
            
            # Calculate cross entropy loss 
            # parameters: outputs[N, C], trg[N]
            # outputs [B, trg_len, out_dim] --> reshaped [B*trg_len, out_dim]
            # trg [B, trg_len] --> reshaped [B*trg_lens]
            loss = criterion(outputs[:, :min_len, :].reshape(-1, out_dim),trg[:, :min_len].reshape(-1))

        # ** Backpropagation **
        # scale(loss) scale tiny values duing FP16 precision
        # loss is propagated backward from final layer
        scaler.scale(loss).backward()

        # graidents are restord to orignal values in optimizer state
        scaler.unscale_(optimizer)

        # exploding gradients are clipped
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # optimizer updates weights with updated gradients
        scaler.step(optimizer)

        # scaler updates scale value accordingly for next iteration
        scaler.update()

        # scehduler updates learning rate based on schduler (OneCycleLR) 
        # adaptive learning rate
        if scheduler is not None:
            scheduler.step()
        
        # loss.item() is scalar loss value
        total_loss += loss.item()

    # returns mean batch loss
    return total_loss / len(loader)

# ------------- Evaluation/Validation function
# parameters: (seq2seq model obj, dataloader, CE criterion) 
def evaluate(model, loader, criterion, id2tok):
    
    # evaluation mode on (deactiavte dopouts and batch norms) for deterministic output
    model.eval()

    total_loss = 0.0
    preds_str = []
    refs_str = []
    
    # disable gradient tracking
    with torch.no_grad():
        
        for src, src_len, trg, _ in tqdm(loader, desc="eval", leave=False):

            src, src_len, trg = src.to(device), src_len.to(device), trg.to(device)
            
            # Forward pass: src sentto model for prediction 
            # no target is sent, model purely predict its next token on prev (tf=0)
            # outputs [B, trg_len(max), Vocab_size]
            outputs = model(src, src_len, trg=None, teacher_forcing=0.0, max_len=trg.size(1))
            
            # get out_dim and prediction lengths
            out_dim = outputs.shape[-1]
            out_len = outputs.size(1)
            trg_len = trg.size(1)

            min_len = min(out_len, trg_len)
            
            # compute cross entropy loss on (outputs[N,V] , trg[N])
            loss = criterion(outputs[:, :min_len, :].reshape(-1, out_dim), trg[:, :min_len].reshape(-1))

            total_loss += loss.item()
            
            # get highest lrob tokens along vocab dimension(-1) and stores them in list for decoding
            top = outputs.argmax(-1).cpu().tolist()
            
            # restoring character strings form token ids
            for i in range(len(top)):
                preds_str.append(ids_to_string(top[i], id2tok))
                refs_str.append(ids_to_string(trg[i].cpu().tolist(), id2tok))

    # retuen mean batch loss, predicted strings and target strings for CER/BLE computations
    return total_loss / len(loader), preds_str, refs_str


# --------******************--------------
# System's Orchestrator 
def run_training(INPUT_DIM, OUTPUT_DIM, emb_dim=256, hid_dim=512, enc_layers=2, dec_layers=4, dropout=0.3, lr=5e-4,
                 epochs=12, save_dir="/", resume_from=None):

    # define model architectures
    encoder = EncoderBiLSTM(INPUT_DIM, emb_dim, hid_dim, n_layers=enc_layers, dropout=dropout, pad_idx=PAD_IDX)
    decoder = Decoder(OUTPUT_DIM, emb_dim, hid_dim, n_layers=dec_layers, dropout=dropout, pad_idx=PAD_IDX)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # adaptive optimizer with explicit decoupled weight decay for optimizing weights
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    # adaptive scheduler for optimizng learning rate
    # warmup lr start, cooldown lr later
    scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs,
                           pct_start=0.05, div_factor=25.0, final_div_factor=1e4, anneal_strategy="cos")
    
    # loss function with label smoothing
    # label smoothing replaces one hot targets with softer probability distribution
    # target [0,1,0,0] - after label smoothing -> [0.167, 0.95, 0.167, 0.167]
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.05)
    
    # To scale graidents (loss) before sending them backward 
    # mantains numeric precisin during FP16 training
    scaler = GradScaler()   

    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = -1                 

    # if best checkpoint is given, resume from it
    if resume_from is not None:
        
        # loading model state from .pt file
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        
        # starting record
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_epoch = checkpoint.get("best_epoch", -1)
        print(f"Resumed training from epoch {start_epoch} using {resume_from}")

    # storing history
    history = {"train_loss": [], "val_loss": [], "bleu": [], "cer": [], "ppl": []}
    
    # loop to run for specified epochs
    for epoch in range(start_epoch, start_epoch + epochs):

        # gradually decreses tf 0 -> 1
        tf = max(0.0, 1 - epoch / max(1, start_epoch + epochs))  
        
        # feeding data to model
        # train_epoch for runs gradints updates
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, tf, scheduler)
        
        # runs inference without grads 
        val_loss, preds, refs = evaluate(model, val_loader, criterion, tgt_id2tok)

        # compute performance metrices
        bleu, cer, ppl = compute_metrics(preds, refs, val_loss)

        # store metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["bleu"].append(bleu)
        history["cer"].append(cer)
        history["ppl"].append(ppl)

        print(f"[E{epoch+1}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} BLEU={bleu:.2f} CER={cer:.4f} PPL={ppl:.2f}")

        # save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch
        }
        torch.save(checkpoint, f"{save_dir}/checkpoint_epoch{epoch+1}.pt")

        # check for best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1  
            torch.save(checkpoint, f"{save_dir}/best_checkpoint.pt")
            print(f"==> Saved best model at epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    
    # returns metrics and final updated model
    return history, model

