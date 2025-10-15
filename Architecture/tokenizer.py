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
