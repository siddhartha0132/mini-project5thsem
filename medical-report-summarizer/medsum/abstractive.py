"""Step 4b: abstractive summarization with a transformer encoder-decoder.

Uses a pre-trained BART/T5 style model from Hugging Face. By default this is
"sshleifer/distilbart-cnn-12-6" (a smaller, faster BART trained for
summarization). If a fine-tuned model exists in models/medsum-bart (created by
scripts/finetune.py), that model is used instead.

Long reports are split into chunks that fit the model's input limit; each chunk
is summarized and, if needed, the partial summaries are summarized again.
"""

import os
import re
import threading

DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"
FINETUNED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "medsum-bart")

_lock = threading.Lock()
_cache = {}


class ModelUnavailable(Exception):
    pass


def model_name():
    if os.path.isdir(FINETUNED_DIR) and os.path.exists(os.path.join(FINETUNED_DIR, "config.json")):
        return FINETUNED_DIR
    return os.environ.get("MEDSUM_MODEL", DEFAULT_MODEL)


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(name=None):
    """Load (once) and return (tokenizer, model, device)."""
    name = name or model_name()
    with _lock:
        if name not in _cache:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(name)
                model = AutoModelForSeq2SeqLM.from_pretrained(name)
            except Exception as exc:
                raise ModelUnavailable("Could not load model '%s': %s" % (name, exc))
            device = _device()
            model.to(device).eval()
            _cache[name] = (tokenizer, model, device)
        return _cache[name]


def _chunk_sentences(sentences, tokenizer, max_tokens):
    chunks, current, length = [], [], 0
    for sentence in sentences:
        n = len(tokenizer.tokenize(sentence))
        if current and length + n > max_tokens:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(sentence)
        length += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def _generate(text, tokenizer, model, device, max_len, min_len):
    import torch

    prefix = "summarize: " if "t5" in model.config.model_type else ""
    inputs = tokenizer(prefix + text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            num_beams=4,
            max_length=max_len,
            min_length=min_len,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def clean_output(text):
    """Fix spacing before punctuation, repeated words ("male male") and a cut-off last sentence."""
    text = re.sub(r"\s+([.,;:%)])", r"\1", text)
    text = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if text and text[-1] not in ".!?" and "." in text:
        text = text[:text.rfind(".") + 1]
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def summarize(sentences, max_len=180, min_len=60):
    """Summarize a list of sentences into a short paragraph."""
    if not sentences:
        return ""
    return clean_output(_summarize(sentences, max_len, min_len))


def _summarize(sentences, max_len, min_len):
    tokenizer, model, device = load()
    limit = min(1000, getattr(tokenizer, "model_max_length", 1024) or 1024) - 24
    chunks = _chunk_sentences(sentences, tokenizer, limit)
    if len(chunks) == 1:
        tokens = len(tokenizer.tokenize(chunks[0]))
        return _generate(chunks[0], tokenizer, model, device, max_len, min(min_len, tokens // 2))
    partial = [_generate(c, tokenizer, model, device, 120, 30) for c in chunks]
    joined = " ".join(partial)
    if len(tokenizer.tokenize(joined)) > limit or len(partial) > 2:
        return _generate(joined, tokenizer, model, device, max_len, min_len)
    return joined
