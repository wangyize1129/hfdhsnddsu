"""Auxiliary text contrastive loss; inference uses only MOMENT and its head."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def load_text_cache(path, subject_ids):
    with np.load(path, allow_pickle=False) as cache:
        ids = cache["subject_ids"].tolist()
        embeddings = cache["embeddings"].astype(np.float32)
        patterns = cache["answer_patterns"].tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs in text cache")
    if embeddings.shape != (len(ids), 768) or len(patterns) != len(ids):
        raise ValueError("Invalid text cache dimensions")
    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite text features")
    lookup = {sid: i for i, sid in enumerate(ids)}
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("Duplicate requested subject IDs")
    missing = set(subject_ids) - set(lookup)
    if missing:
        raise ValueError(f"Missing text features for {sorted(missing)}")
    indices = [lookup[sid] for sid in subject_ids]
    selected_patterns = [patterns[i] for i in indices]
    pattern_lookup = {p: i for i, p in enumerate(dict.fromkeys(selected_patterns))}
    groups = torch.tensor([pattern_lookup[p] for p in selected_patterns], dtype=torch.long)
    return torch.from_numpy(embeddings[indices]), groups


def contrastive_loss(time_vectors, text_vectors, targets, groups, temperature=0.07):
    """All candidates are TRAIN subjects. Equal questionnaires are not negatives."""
    # Casting tensors alone is insufficient inside AMP: matmul would cast back
    # to BF16 and erase small differences between frozen questionnaire vectors.
    with torch.autocast(device_type=time_vectors.device.type, enabled=False):
        logits = time_vectors.float() @ text_vectors.float().T / temperature
        same_pattern = groups[targets, None] == groups[None, :]
        own_subject = targets[:, None] == torch.arange(len(groups), device=groups.device)[None, :]
        logits = logits.masked_fill(same_pattern & ~own_subject, float("-inf"))
        return F.cross_entropy(logits, targets)


class AlignmentModel(nn.Module):
    def __init__(self, moment, d_model=768, n_channels=132, projection_dim=256):
        super().__init__()
        self.moment = moment
        self.d_model = d_model
        self.n_channels = n_channels
        self.time_projector = nn.Sequential(nn.Linear(d_model, projection_dim), nn.GELU(),
                                            nn.Linear(projection_dim, projection_dim))
        self.text_projector = nn.Sequential(nn.Linear(768, projection_dim), nn.GELU(),
                                            nn.Linear(projection_dim, projection_dim))

    def forward_timeseries(self, x):
        output = self.moment(x_enc=x, reduction="concat")
        # MOMENT concat ordering is (hidden dimension, channel), not (channel, hidden).
        features = output.embeddings.mean(dim=1)
        return output.logits, features

    def project_timeseries(self, features):
        pooled = features.reshape(-1, self.d_model, self.n_channels).mean(dim=-1)
        return F.normalize(self.time_projector(pooled.float()), dim=-1)

    def project_text(self, cached_text):
        return F.normalize(self.text_projector(cached_text), dim=-1)

    def classify_features(self, features):
        return self.moment.head(features.unsqueeze(1))

    @torch.inference_mode()
    def predict_timeseries(self, x):
        """Caller sets eval(); no text argument, cache access, or BERT dependency."""
        return self.forward_timeseries(x)[0]
