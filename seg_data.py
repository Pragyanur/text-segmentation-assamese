"""Datasets and batching for AsParaSeg (both tasks, both training modes)."""
import json
import random

import torch
from torch.utils.data import Dataset


TASKS = {
    "fine": {"n_heads": 1, "max_label": 1},   # section-level, {0,1}
    "hier": {"n_heads": 2, "max_label": 2},   # article-level, {0,1,2}
}


class SegDataset(Dataset):
    """One JSONL split. Records keep sentences as text; tokenisation happens
    in the collate function so padding is per-batch, not global."""

    def __init__(self, path, task="fine", max_sents=None, vectors=None):
        assert task in TASKS
        self.task = task
        self.max_label = TASKS[task]["max_label"]
        self.items = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                e = json.loads(line)
                labels = e["labels"]
                if self.max_label == 1:
                    labels = [1 if l >= 1 else 0 for l in labels]
                sents = e["sentences"]
                if max_sents and len(sents) > max_sents:
                    sents, labels = sents[:max_sents], labels[:max_sents]
                    labels = labels[:-1] + [0]   # never end on a boundary
                self.items.append({
                    "id": e["id"], "sentences": sents, "labels": labels,
                    "n": len(sents),
                })
        # optional precomputed sentence vectors: {doc_id: FloatTensor(n, d)}
        self.vectors = vectors
        if vectors is not None:
            keep = [it for it in self.items if it["id"] in vectors]
            if len(keep) != len(self.items):
                print(f"  warning: {len(self.items)-len(keep)} docs have no "
                      f"cached vectors and were dropped")
            self.items = keep

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def label_counts(self):
        c = {0: 0, 1: 0, 2: 0}
        for it in self.items:
            for l in it["labels"]:
                c[l] += 1
        return c


# ------------------------------------------------------------------ collates
def make_text_collate(tokenizer, max_tokens=64):
    """For finetuning: returns token ids (B,S,T)."""
    def collate(batch):
        B = len(batch)
        S = max(b["n"] for b in batch)
        flat = []
        for b in batch:
            flat.extend(b["sentences"])
            flat.extend([""] * (S - b["n"]))     # padding sentences
        enc = tokenizer(flat, padding=True, truncation=True,
                        max_length=max_tokens, return_tensors="pt")
        T = enc["input_ids"].shape[1]
        labels = torch.zeros(B, S, dtype=torch.long)
        sent_mask = torch.zeros(B, S, dtype=torch.long)
        for i, b in enumerate(batch):
            labels[i, :b["n"]] = torch.tensor(b["labels"])
            sent_mask[i, :b["n"]] = 1
        return {
            "input_ids": enc["input_ids"].view(B, S, T),
            "attention_mask": enc["attention_mask"].view(B, S, T),
            "sent_mask": sent_mask, "labels": labels,
            "ids": [b["id"] for b in batch],
            "lengths": [b["n"] for b in batch],
        }
    return collate


def make_vector_collate(vectors):
    """For frozen-encoder training: returns cached sentence vectors (B,S,d)."""
    def collate(batch):
        B = len(batch)
        S = max(b["n"] for b in batch)
        d = vectors[batch[0]["id"]].shape[-1]
        vecs = torch.zeros(B, S, d)
        labels = torch.zeros(B, S, dtype=torch.long)
        sent_mask = torch.zeros(B, S, dtype=torch.long)
        for i, b in enumerate(batch):
            v = vectors[b["id"]][:b["n"]]
            vecs[i, :b["n"]] = v
            labels[i, :b["n"]] = torch.tensor(b["labels"])
            sent_mask[i, :b["n"]] = 1
        return {
            "sent_vecs": vecs, "sent_mask": sent_mask, "labels": labels,
            "ids": [b["id"] for b in batch],
            "lengths": [b["n"] for b in batch],
        }
    return collate


# ------------------------------------------------------------------ batching
class BucketBatchSampler(torch.utils.data.Sampler):
    """Group documents of similar length into a batch.

    Article-level documents range from 8 to 300 sentences. Batching those at
    random means most of every batch is padding, and with an end-to-end
    encoder the cost is B*S*T forward passes -- so this is not cosmetic, it is
    the difference between fitting in memory and not.

    max_batch_sents additionally caps the PADDED sentence count of a batch,
    which is batch_size x the longest document in it. Peak GPU memory is driven
    by that PRODUCT, not by batch_size: the encoder sees one sequence per
    sentence, so 16 documents of 150 sentences is 2400 sequences in a single
    forward pass. A fixed document count therefore means a batch of long
    articles can be 20x heavier than a batch of short ones, which is why a run
    can train happily for hours and then die. With the cap the batch shrinks
    automatically wherever the documents are long. A document that exceeds the
    cap alone is still yielded on its own rather than dropped.
    """

    def __init__(self, lengths, batch_size, shuffle=True, pool=50, seed=0,
                 max_batch_sents=0):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pool = pool * batch_size
        self.rng = random.Random(seed)
        self.max_batch_sents = max_batch_sents
        # batch count varies slightly with the shuffle; fix it once so the LR
        # schedule has a stable number of steps
        self._len = len(self._build(random.Random(seed)))

    def _build(self, rng):
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(idx)
        batches = []
        for i in range(0, len(idx), self.pool):
            chunk = sorted(idx[i:i + self.pool], key=lambda j: self.lengths[j])
            cur, cur_max = [], 0
            for j in chunk:
                n = self.lengths[j]
                new_max = max(cur_max, n)
                too_many = len(cur) + 1 > self.batch_size
                too_big = (self.max_batch_sents and cur and
                           (len(cur) + 1) * new_max > self.max_batch_sents)
                if cur and (too_many or too_big):
                    batches.append(cur)
                    cur, cur_max = [j], n
                else:
                    cur.append(j)
                    cur_max = new_max
            if cur:
                batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._build(self.rng))

    def __len__(self):
        return self._len


def unpad(tensor2d, lengths):
    """(B,S) tensor -> list of python lists, padding removed."""
    return [tensor2d[i, :n].tolist() for i, n in enumerate(lengths)]
