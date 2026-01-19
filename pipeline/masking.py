import re

PLACEHOLDER_PATTERN = re.compile(r"\{\d+\}")
SYMBOL_PATTERN = re.compile(r"[™®]")


def add_mask_tokens(tokenizer, model, n=100):
    tokens = [f"<MASK{i}>" for i in range(n)]
    tokenizer.add_tokens(tokens)
    model.resize_token_embeddings(len(tokenizer))


def mask_text(text):
    mappings = {}
    counter = 0

    def repl(match):
        nonlocal counter
        key = f"<MASK{counter}>"
        mappings[key] = match.group(0)
        counter += 1
        return key

    text = PLACEHOLDER_PATTERN.sub(repl, text)
    text = SYMBOL_PATTERN.sub(repl, text)
    return text, mappings


def unmask_text(text, mappings):
    for k, v in mappings.items():
        text = text.replace(k, v)
    return text