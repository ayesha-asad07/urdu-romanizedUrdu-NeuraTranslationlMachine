from Architecture.seq2seq import *

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

