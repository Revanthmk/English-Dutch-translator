import yaml
import torch
import sacrebleu

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from data import load_flores, load_challenge_excel
from masking import add_mask_tokens, mask_text, unmask_text

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tok = AutoTokenizer.from_pretrained("trained_model")
model_type = cfg["training"]["model_type"]

if model_type == "encoder_decoder":
    model = AutoModelForSeq2SeqLM.from_pretrained("trained_model").to(device)
else:
    model = AutoModelForCausalLM.from_pretrained("trained_model").to(device)

if cfg["evaluation"]["use_masking"]:
    add_mask_tokens(tok, model)

model.eval()


def translate(text):
    if cfg["evaluation"]["use_masking"]:
        text, m = mask_text(text)
    else:
        m = None

    if model_type == "encoder_decoder":
        inp = tok(text, return_tensors="pt").to(device)
        out = model.generate(**inp, num_beams=cfg["evaluation"]["num_beams"])
        res = tok.decode(out[0], skip_special_tokens=True)
    else:
        prompt = f"Translate English to Dutch:\n{text}\nDutch:"
        inp = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(**inp, max_new_tokens=64)
        res = tok.decode(out[0], skip_special_tokens=True).split("Dutch:")[-1]

    return unmask_text(res, m) if m else res


def evaluate(src, ref):
    hyp = [translate(s) for s in src]
    print("BLEU:", sacrebleu.corpus_bleu(hyp, [ref]).score)
    print("chrF:", sacrebleu.corpus_chrf(hyp, [ref]).score)


src, ref = load_flores(cfg["paths"]["flores_dir"])
evaluate(src, ref)

src, ref = load_challenge_excel(cfg["paths"]["challenge_excel"])
evaluate(src, ref)
