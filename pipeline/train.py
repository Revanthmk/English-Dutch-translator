import yaml
import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data import (
    load_parallel, filter_pairs, subsample_and_split,
    save_pairs, TranslationDataset, DecoderOnlyTranslationDataset
)
from models import load_model_and_tokenizer

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pairs = load_parallel(cfg["paths"]["europarl_en"], cfg["paths"]["europarl_nl"])
pairs = filter_pairs(pairs, cfg["data"]["max_len"], cfg["data"]["min_len"])
train_p, val_p = subsample_and_split(
    pairs, cfg["data"]["total_size"], cfg["data"]["val_size"], cfg["data"]["seed"]
)

save_pairs(train_p, cfg["paths"]["train_en"], cfg["paths"]["train_nl"])
save_pairs(val_p, cfg["paths"]["val_en"], cfg["paths"]["val_nl"])

model, tokenizer = load_model_and_tokenizer(cfg, device)

ds_cls = TranslationDataset if cfg["training"]["model_type"] == "encoder_decoder" \
    else DecoderOnlyTranslationDataset

train_ds = ds_cls(cfg["paths"]["train_en"], cfg["paths"]["train_nl"],
                  tokenizer, cfg["data"]["max_len"])
val_ds = ds_cls(cfg["paths"]["val_en"], cfg["paths"]["val_nl"],
                tokenizer, cfg["data"]["max_len"])

train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"])

opt = AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))

wandb.init(project="EN-NL-MT", config=cfg)

for e in range(cfg["training"]["epochs"]):
    model.train()
    tot = 0
    for b in train_loader:
        b = {k: v.to(device) for k, v in b.items()}
        loss = model(**b).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()

    wandb.log({"train_loss": tot / len(train_loader), "epoch": e})

model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")
wandb.finish()
