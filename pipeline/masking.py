import re

PLACEHOLDER = re.compile(r"\{\d+\}")
SYMBOL = re.compile(r"[™®]")


def add_mask_tokens(tokenizer, model, n=100):
    tokens = [f"<MASK{i}>" for i in range(n)]
    tokenizer.add_tokens(tokens)
    model.resize_token_embeddings(len(tokenizer))


def mask_text(text):
    mapping = {}
    idx = 0

    def repl(m):
        nonlocal idx
        k = f"<MASK{idx}>"
        mapping[k] = m.group(0)
        idx += 1
        return k

    text = PLACEHOLDER.sub(repl, text)
    text = SYMBOL.sub(repl, text)
    return text, mapping


def unmask_text(text, mapping):
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text
