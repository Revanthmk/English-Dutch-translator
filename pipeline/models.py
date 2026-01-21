from transformers import (
    MarianMTModel, MarianTokenizer,
    AutoModelForCausalLM, AutoTokenizer
)
from peft import LoraConfig, get_peft_model, TaskType


def load_model_and_tokenizer(cfg, device):
    mtype = cfg["training"]["model_type"]
    name = cfg["training"]["model_name"]

    if mtype == "encoder_decoder":
        tok = MarianTokenizer.from_pretrained(name)
        model = MarianMTModel.from_pretrained(name)

    elif mtype == "decoder_only":
        tok = AutoTokenizer.from_pretrained(name)
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(name)

    else:
        raise ValueError("Unknown model_type")

    if cfg["peft"]["use_lora"]:
        task = TaskType.SEQ_2_SEQ_LM if mtype == "encoder_decoder" else TaskType.CAUSAL_LM
        lora_cfg = LoraConfig(
            r=cfg["peft"]["r"],
            lora_alpha=cfg["peft"]["alpha"],
            lora_dropout=cfg["peft"]["dropout"],
            task_type=task,
        )
        model = get_peft_model(model, lora_cfg)

    return model.to(device), tok
