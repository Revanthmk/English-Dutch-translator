import yaml
import torch
import wandb
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW

from data import (
    load_parallel, filter_pairs, subsample_and_split,
    save_pairs, TranslationDataset
)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pairs = load_parallel(cfg["paths"]["europarl_en"], cfg["paths"]["europarl_nl"])
pairs = filter_pairs(pairs, cfg["data"]["max_len"], cfg["data"]["min_len"])
train_pairs, val_pairs = subsample_and_split(
    pairs,
    cfg["data"]["total_size"],
    cfg["data"]["val_size"],
    cfg["data"]["seed"],
)

save_pairs(train_pairs, cfg["paths"]["train_en"], cfg["paths"]["train_nl"])
save_pairs(val_pairs, cfg["paths"]["val_en"], cfg["paths"]["val_nl"])

tokenizer = MarianTokenizer.from_pretrained(cfg["training"]["model_name"])
model = MarianMTModel.from_pretrained(cfg["training"]["model_name"]).to(device)

train_ds = TranslationDataset(
    cfg["paths"]["train_en"], cfg["paths"]["train_nl"],
    tokenizer, cfg["data"]["max_len"]
)
val_ds = TranslationDataset(
    cfg["paths"]["val_en"], cfg["paths"]["val_nl"],
    tokenizer, cfg["data"]["max_len"]
)

train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"])

optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"])

wandb.init(project="EN-NL-MT", config=cfg)

for epoch in range(cfg["training"]["epochs"]):
    model.train()
    total = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()

    wandb.log({"train_loss": total / len(train_loader), "epoch": epoch})

model.save_pretrained("finetuned-marian-en-nl")
tokenizer.save_pretrained("finetuned-marian-en-nl")
wandb.finish()