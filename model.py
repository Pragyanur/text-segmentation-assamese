"""
Models for AsParaSeg.

Two models, one shared backbone:

    FINE          section-level input, binary labels {0,1}
                  one output head:  P(paragraph boundary)

    HIERARCHICAL  article-level input, ordinal labels {0,1,2}
                  two output heads (CUMULATIVE / CORAL-style):
                      head_a -> P(label >= 1)  "is there a break here at all?"
                      head_b -> P(label >= 2)  "is that break section-level?"

"Head" means an OUTPUT LAYER -- literally nn.Linear(d, 1). Nothing to do with
multi-head attention. Going from the fine model to the hierarchical one is one
extra linear layer plus one extra BCE term.

Why cumulative rather than a 3-way softmax: the labels NEST. A section break is
also a paragraph break -- a section never starts mid-paragraph -- so
"label >= 1" is literally true at both 1 and 2, and head_a learns one coherent
concept. A 3-way softmax would treat confusing 1-vs-2 as exactly as costly as
0-vs-2, throwing the ordering away. The cumulative form also makes the model
emit a graded SHIFT STRENGTH rather than a class, which is the part that
transfers to text with no section markup at all.

THREE STAGES

  1. sentence encoder   MuRIL encodes each sentence independently -> one vector
                        per sentence. Knows nothing about context.
  2. context encoder    a small transformer over the SEQUENCE OF SENTENCE
                        VECTORS. This is where segmentation happens: a boundary
                        is a property of the JOIN between sentences, which
                        stage 1 structurally cannot see.
  3. output head(s)     one scalar per sentence position.

Stages 1 and 2 are separate classes so stage 1 can be run ONCE offline
(precompute.py) and stage 2 trained on the cached vectors. On article-level
data that is the difference between needing a large GPU and not.
"""
import torch
import torch.nn as nn


# --------------------------------------------------------------- stage 1
class SentenceEncoder(nn.Module):
    """Wraps MuRIL. Input (B, S, T) token ids -> (B, S, d) sentence vectors."""

    def __init__(self, encoder_name="google/muril-base-cased", freeze=False):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.frozen = freeze

    def _layer_list(self):
        """The encoder's list of transformer layers, or None if the
        architecture does not expose one as an ordered stack.

        BERT / RoBERTa / XLM-R / MuRIL:  encoder.encoder.layer
        ALBERT deliberately returns None -- its layers SHARE weights across
        depth, so "train only the top 2 layers" is not a meaningful request
        there: training any layer trains all of them.
        """
        enc = getattr(self.encoder, "encoder", None)
        if enc is None:
            return None
        layers = getattr(enc, "layer", None)
        if layers is not None and len(layers):
            return layers
        return None

    def set_trainable_top_layers(self, n):
        """Freeze the whole encoder, then unfreeze only its top n layers.

        Returns (n_unfrozen, n_total) or None if the architecture has no
        ordered layer stack.

        Two reasons this is usually the right choice over full finetuning:

        1. Capacity. Most of MuRIL is its embedding table (~62% of its 227M
           parameters, because the vocabulary is ~184k). Adapting that on a
           few tens of thousands of silver-labelled documents is overfitting
           surface, not useful capacity.
        2. Memory AND speed. With the embeddings and lower layers frozen, the
           input to the first trainable layer does not require grad, so
           autograd never builds a graph for the bottom of the stack: those
           layers run effectively under no_grad. Activations are stored for n
           layers instead of all of them, and the backward pass only traverses
           n layers. Unlike gradient checkpointing this costs no recompute --
           it is strictly cheaper in both memory and time.
        """
        for p in self.encoder.parameters():
            p.requires_grad = False
        layers = self._layer_list()
        if layers is None:
            return None
        total = len(layers)
        n = max(0, min(int(n), total))
        for layer in list(layers)[total - n:] if n else []:
            for p in layer.parameters():
                p.requires_grad = True
        self.frozen = (n == 0)
        return n, total

    def trainable_report(self):
        """(trainable, total) parameter counts for the encoder."""
        tr = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        al = sum(p.numel() for p in self.encoder.parameters())
        return tr, al

    def forward(self, input_ids, attention_mask):
        B, S, T = input_ids.shape
        out = self.encoder(input_ids=input_ids.view(B * S, T),
                           attention_mask=attention_mask.view(B * S, T)
                           ).last_hidden_state
        # mean-pool over real tokens: more stable than [CLS] when the encoder
        # is frozen, since [CLS] was never trained for this objective
        m = attention_mask.view(B * S, T).unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return pooled.view(B, S, -1)


# --------------------------------------------------------------- stages 2 + 3
class ContextSegmenter(nn.Module):
    """Sentence vectors -> per-position boundary logits.

    n_heads=1 -> fine model      (returns a 1-tuple)
    n_heads=2 -> hierarchical    (returns logit_a, logit_b)
    """

    def __init__(self, d_in, n_heads=1, d_model=None, n_layers=2,
                 n_att_heads=8, dropout=0.1):
        super().__init__()
        d = d_model or d_in
        self.proj = nn.Linear(d_in, d) if d != d_in else nn.Identity()
        self.pos = nn.Embedding(1024, d)          # sentence-position embedding
        self.in_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_att_heads, dim_feedforward=4 * d,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        # enable_nested_tensor=False: incompatible with norm_first=True and
        # only emits a warning otherwise
        self.ctx = nn.TransformerEncoder(layer, num_layers=n_layers,
                                         enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(n_heads)])
        self.n_heads = n_heads

    def forward(self, sent_vecs, sent_mask):
        """sent_vecs (B,S,d_in); sent_mask (B,S) 1 on real sentences."""
        B, S, _ = sent_vecs.shape
        h = self.in_norm(self.proj(sent_vecs))
        idx = torch.arange(S, device=sent_vecs.device).clamp(max=1023)
        h = self.drop(h + self.pos(idx).unsqueeze(0))
        # nn.TransformerEncoder wants True = "ignore this position"
        h = self.ctx(h, src_key_padding_mask=(sent_mask == 0))
        return tuple(head(h).squeeze(-1) for head in self.heads)


class EndToEndSegmenter(nn.Module):
    """SentenceEncoder + ContextSegmenter, for finetuning MuRIL as well."""

    def __init__(self, encoder_name="google/muril-base-cased", n_heads=1,
                 freeze_encoder=False, **ctx_kw):
        super().__init__()
        self.enc = SentenceEncoder(encoder_name, freeze=freeze_encoder)
        self.ctx = ContextSegmenter(self.enc.hidden_size, n_heads=n_heads,
                                    **ctx_kw)

    def forward(self, input_ids, attention_mask, sent_mask):
        return self.ctx(self.enc(input_ids, attention_mask), sent_mask)


# --------------------------------------------------------------- losses
def fine_loss(logit, labels, sent_mask, pos_weight=None):
    """Binary. labels in {0,1}."""
    bce = nn.functional.binary_cross_entropy_with_logits
    l = bce(logit, (labels > 0).float(), reduction="none", pos_weight=pos_weight)
    m = sent_mask.float()
    return (l * m).sum() / m.sum().clamp(min=1.0)


def hierarchical_loss(logit_a, logit_b, labels, sent_mask,
                      pos_weight_a=None, pos_weight_b=None, w_b=1.0):
    """Ordinal. labels in {0,1,2}.

    Targets are CUMULATIVE -- this is what encodes the nesting:
        y_a = (label >= 1)   1 at BOTH paragraph and section boundaries
        y_b = (label >= 2)   1 only at section boundaries

    Both heads are trained at every real position; padding is masked out.
    w_b scales head B's contribution if it needs to be down/up-weighted
    relative to head A.
    """
    bce = nn.functional.binary_cross_entropy_with_logits
    y_a = (labels >= 1).float()
    y_b = (labels >= 2).float()
    la = bce(logit_a, y_a, reduction="none", pos_weight=pos_weight_a)
    lb = bce(logit_b, y_b, reduction="none", pos_weight=pos_weight_b)
    m = sent_mask.float()
    n = m.sum().clamp(min=1.0)
    return (la * m).sum() / n + w_b * (lb * m).sum() / n


# --------------------------------------------------------------- decoding
@torch.no_grad()
def decode_fine(logit, thr=0.5):
    return (torch.sigmoid(logit) >= thr).long()


@torch.no_grad()
def decode_hier(logit_a, logit_b, thr_a=0.5, thr_b=0.5):
    """Cumulative logits -> {0,1,2}. Monotonicity is enforced at decode time:
    a position can only be a 2 if it is already a 1."""
    pa, pb = torch.sigmoid(logit_a), torch.sigmoid(logit_b)
    out = torch.zeros_like(pa, dtype=torch.long)
    out[pa >= thr_a] = 1
    out[(pa >= thr_a) & (pb >= thr_b)] = 2
    return out


def pos_weight_for(counts_pos, counts_total):
    """negatives / positives -- pass into the BCE so the rare class survives."""
    p = max(int(counts_pos), 1)
    return torch.tensor(float(counts_total - p) / p)
