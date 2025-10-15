from helpers import *

# --------******************--------------
# System's Orchestrator  ( Runs on GPUs (Kaggle/Colab is used))

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

