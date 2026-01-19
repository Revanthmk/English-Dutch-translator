import yaml
import torch
import sacrebleu
from transformers import MarianMTModel, MarianTokenizer

from data import load_flores, load_challenge_excel
from masking import mask_text, unmask_text, add_mask_tokens

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = MarianTokenizer.from_pretrained("finetuned-marian-en-nl")
model = MarianMTModel.from_pretrained("finetuned-marian-en-nl").to(device)

if cfg["evaluation"]["use_masking"]:
    add_mask_tokens(tokenizer, model)

model.eval()


def translate(text):
    if cfg["evaluation"]["use_masking"]:
        text, mappings = mask_text(text)
    else:
        mappings = None

    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    out = model.generate(**inputs, num_beams=cfg["evaluation"]["num_beams"])
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    return unmask_text(decoded, mappings) if mappings else decoded


def evaluate(src, ref):
    hyp = [translate(s) for s in src]
    print("BLEU:", sacrebleu.corpus_bleu(hyp, [ref]).score)
    print("chrF:", sacrebleu.corpus_chrf(hyp, [ref]).score)

src, ref = load_flores(cfg["paths"]["flores_dir"])
evaluate(src, ref)

src, ref = load_challenge_excel(cfg["paths"]["challenge_excel"])
evaluate(src, ref)