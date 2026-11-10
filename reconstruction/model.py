"""MOMENT and questionnaire classification with conditional masked reconstruction."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def load_cache(path, subject_ids):
    with np.load(path, allow_pickle=False) as data:
        ids = data['subject_ids'].tolist()
        if len(set(ids)) != len(ids) or len(set(subject_ids)) != len(subject_ids):
            raise ValueError('Duplicate subject IDs')
        lookup = {sid: i for i, sid in enumerate(ids)}
        answers = data['answers'][[lookup[sid] for sid in subject_ids]]
        conditions, targets = data['conditions'], data['targets']
    if answers.shape != (len(subject_ids), 30) or not np.isin(answers, [0, 1]).all():
        raise ValueError('Invalid binary answers')
    if conditions.shape != (30, 768) or targets.shape != (30, 2, 768):
        raise ValueError('Invalid BioLORD cache dimensions')
    if not np.isfinite(conditions).all() or not np.isfinite(targets).all():
        raise ValueError('Nonfinite text features')
    return torch.from_numpy(answers).long(), torch.from_numpy(conditions).float(), torch.from_numpy(targets).float()


def reconstruction_loss(prediction, target):
    # Sum feature dimensions; average subjects and questions.
    return (prediction.float()-target.float()).square().sum(-1).mean()


def questionnaire_mask(batch_size, ratio, generator, device):
    """Hide an exact number of answers per subject using a separate CPU RNG."""
    count = max(1, min(29, round(30 * ratio)))
    order = torch.rand(batch_size, 30, generator=generator).argsort(dim=1)
    mask = torch.zeros(batch_size, 30, dtype=torch.bool)
    mask.scatter_(1, order[:, :count], True)
    return mask.to(device)


def masked_reconstruction_loss(prediction, target, mask):
    return (prediction.float() - target.float()).square().sum(-1)[mask].mean()


def balanced_class_weights(labels, n_classes=3):
    labels = torch.as_tensor(labels, dtype=torch.long)
    if labels.ndim != 1 or labels.numel() == 0 or (labels < 0).any() or (labels >= n_classes).any():
        raise ValueError('Expected nonempty training labels in the configured class range')
    counts = torch.bincount(labels, minlength=n_classes)
    if (counts == 0).any():
        raise ValueError('Each class must occur in the training split')
    return labels.numel() / (n_classes * counts.float())


def weighted_classification_loss(logits, labels, class_weights, label_smoothing=0.0, reduction='mean'):
    losses = F.cross_entropy(logits.float(), labels, weight=class_weights,
                             label_smoothing=label_smoothing, reduction='none')
    if reduction == 'mean':
        return losses.mean()
    if reduction == 'sum':
        return losses.sum()
    raise ValueError('reduction must be mean or sum')


def movement_amplitude(x):
    """Log three-axis fluctuation magnitude, ordered by task, wrist, sensor."""
    if x.ndim != 3 or x.shape[1] != 132 or x.shape[2] < 2:
        raise ValueError('Expected movement with shape (batch, 132, time), time >= 2')
    channels = x.float().reshape(x.shape[0], 11, 2, 2, 3, x.shape[-1])
    variance = channels.var(dim=-1, correction=0)
    return (variance.sum(dim=-1).sqrt() + 1e-6).log().flatten(1)


class QuestionnaireFusionHead(nn.Module):
    """Retain channel identities and combine movement with 30 real answers."""
    def __init__(self, d_model, n_channels, channel_dim=16, time_dim=128,
                 question_dim=32, hidden_dim=64, dropout=0.3,
                 amplitude_mean=None, amplitude_std=None):
        super().__init__()
        self.d_model, self.n_channels = d_model, n_channels
        self.uses_amplitude = amplitude_mean is not None
        if (amplitude_mean is None) != (amplitude_std is None):
            raise ValueError('Both amplitude mean and std are required')
        if self.uses_amplitude:
            mean = torch.as_tensor(amplitude_mean, dtype=torch.float32)
            std = torch.as_tensor(amplitude_std, dtype=torch.float32)
            if (mean.shape != (44,) or std.shape != (44,)
                    or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
                    or not (std > 0).all()):
                raise ValueError('Expected 44 finite amplitude means and positive standard deviations')
            # Checkpoints serialize these fixed training statistics in configuration.
            self.register_buffer('amplitude_mean', mean, persistent=False)
            self.register_buffer('amplitude_std', std, persistent=False)
        self.channel_projection = nn.Sequential(nn.Linear(d_model, channel_dim), nn.GELU())
        self.time_projection = nn.Sequential(
            nn.Flatten(), nn.Linear(n_channels * channel_dim, time_dim),
            nn.GELU(), nn.LayerNorm(time_dim), nn.Dropout(dropout))
        self.question_projection = nn.Sequential(
            nn.Linear(30, question_dim),
            nn.GELU(), nn.LayerNorm(question_dim), nn.Dropout(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(time_dim + question_dim + (44 if self.uses_amplitude else 0), hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 3))

    def forward(self, features, answers, amplitude=None):
        if answers.shape != (features.shape[0], 30):
            raise ValueError('Expected one row of 30 answers per movement sample')
        channels = features.reshape(-1, self.d_model, self.n_channels).transpose(1, 2)
        time = self.time_projection(self.channel_projection(channels))
        if not torch.isfinite(answers).all() or not ((answers == 0) | (answers == 1)).all():
            raise ValueError('Questionnaire answers must be finite binary values')
        question = self.question_projection(answers.to(device=features.device, dtype=torch.float32))
        if self.uses_amplitude:
            if amplitude is None or amplitude.shape != (features.shape[0], 44):
                raise ValueError('Expected 44 amplitude features per subject')
            scaled = (amplitude.float() - self.amplitude_mean) / self.amplitude_std
            return self.classifier(torch.cat((time, question, scaled), dim=-1))
        return self.classifier(torch.cat((time, question), dim=-1))


class ReconstructionModel(nn.Module):
    def __init__(self, moment, hidden_dim=512, d_model=768, n_channels=132, init_std=1e-3, norm_eps=1e-5):
        super().__init__()
        self.moment = moment
        self.d_model, self.n_channels = d_model, n_channels
        self.time_norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.question_norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.decoder = nn.Sequential(nn.Linear(2*d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model))
        # Keep initial squared-L2 loss near the unit-target baseline, without
        # zeroing the gradient into MOMENT.
        nn.init.normal_(self.decoder[-1].weight, std=init_std)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward_timeseries(self, x):
        output = self.moment(x_enc=x, reduction='concat')
        return output.logits, output.embeddings.mean(1)

    def reconstruct(self, features, conditions, answers, mask, use_motion=True):
        if answers.shape != mask.shape or mask.dtype != torch.bool:
            raise ValueError('Expected aligned answers and a boolean mask')
        visible = torch.where(mask, 0.0, answers.float())
        context = self.masked_context(torch.cat((visible, (~mask).float()), dim=-1))
        if use_motion:
            time = self.time_norm(features.float().reshape(-1, self.d_model, self.n_channels).mean(-1))
            context = context + time
        question = self.question_norm(conditions.float())
        batch, questions = context.shape[0], question.shape[0]
        decoder_input = torch.cat((context[:, None].expand(-1, questions, -1),
                                   question[None].expand(batch, -1, -1)), dim=-1)
        return self.decoder(decoder_input)

    def forward(self, x, answers):
        amplitude = movement_amplitude(x) if self.fusion_head.uses_amplitude else None
        _, features = self.forward_timeseries(x)
        return self.fusion_head(features, answers, amplitude), features

    @torch.inference_mode()
    def predict(self, x, answers):
        if self.training:
            raise RuntimeError('Call eval() before inference')
        return self(x, answers)[0]

