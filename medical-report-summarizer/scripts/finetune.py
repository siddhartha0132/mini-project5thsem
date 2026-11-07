"""Fine-tune the BART summarization model on medical report/summary pairs.

Input: a CSV file with two columns, "report" and "summary" (for example made from
MIMIC-III discharge summaries or MTSamples transcriptions). The fine-tuned model is
saved to models/medsum-bart and is picked up automatically by the web app.

Usage:
    python scripts/finetune.py --data data/train.csv --epochs 3
    python scripts/finetune.py --data data/train.csv --base facebook/bart-base --batch 4

A GPU (or Google Colab) is recommended; Apple Silicon (MPS) and CPU also work for small sets.
"""

import argparse
import csv
import os
import random
import sys

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from medsum.abstractive import DEFAULT_MODEL, FINETUNED_DIR, _device  # noqa: E402
from medsum.preprocess import preprocess  # noqa: E402


class ReportDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_source=1024, max_target=256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_source = max_source
        self.max_target = max_target

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        report, summary = self.pairs[i]
        source = self.tokenizer(report, max_length=self.max_source, truncation=True)
        target = self.tokenizer(text_target=summary, max_length=self.max_target, truncation=True)
        return {"input_ids": source["input_ids"], "labels": target["input_ids"]}


def collate(batch, pad_id):
    def pad(seqs, value):
        width = max(len(s) for s in seqs)
        return torch.tensor([s + [value] * (width - len(s)) for s in seqs])

    input_ids = pad([b["input_ids"] for b in batch], pad_id)
    labels = pad([b["labels"] for b in batch], -100)  # -100 is ignored by the loss
    return {"input_ids": input_ids, "attention_mask": (input_ids != pad_id).long(), "labels": labels}


def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pairs = []
    for row in rows:
        report = preprocess(row["report"])["expanded"].replace("\n", " ")
        if report.strip() and row["summary"].strip():
            pairs.append((report, row["summary"].strip()))
    return pairs


def main():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with 'report' and 'summary' columns")
    parser.add_argument("--base", default=DEFAULT_MODEL, help="pre-trained model to start from")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--out", default=FINETUNED_DIR)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    pairs = load_pairs(args.data)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_split)) if len(pairs) > 5 else 0
    train, val = pairs[n_val:], pairs[:n_val]
    print("Training on %d pairs, validating on %d" % (len(train), len(val)))

    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base).to(device)

    def loader(data, shuffle):
        return DataLoader(ReportDataset(data, tokenizer), batch_size=args.batch, shuffle=shuffle,
                          collate_fn=lambda b: collate(b, tokenizer.pad_token_id))

    train_loader = loader(train, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item()
            if step % 10 == 0:
                print("epoch %d step %d/%d loss %.4f" % (epoch, step, len(train_loader), total / step))
        train_loss = total / max(1, len(train_loader))

        val_loss = train_loss
        if val:
            model.eval()
            with torch.no_grad():
                losses = [model(**{k: v.to(device) for k, v in b.items()}).loss.item() for b in loader(val, False)]
            val_loss = sum(losses) / len(losses)
        print("epoch %d: train loss %.4f, validation loss %.4f" % (epoch, train_loss, val_loss))

        if val_loss < best:
            best = val_loss
            os.makedirs(args.out, exist_ok=True)
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
            print("saved best model to", args.out)


if __name__ == "__main__":
    main()
