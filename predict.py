"""Predict HC/PD/DD from movement and the subject's complete questionnaire."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np
from train import make_model
from momentfm.data.bin_timeseries import load_movement_bin


def load_model(checkpoint, moment_dir=None, device='auto'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = saved['configuration']
    if moment_dir:
        config['local_model_dir'] = moment_dir
    model = make_model(SimpleNamespace(**config))
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(saved['trainable_state']) != expected:
        raise ValueError('Checkpoint trainable parameter names do not match model')
    result = model.load_state_dict(saved['trainable_state'], strict=False)
    if result.unexpected_keys:
        raise ValueError(result.unexpected_keys)
    return model.to(device).eval(), config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--movement', required=True)
    p.add_argument('--moment-dir')
    p.add_argument('--answers', required=True, help='JSON list of 30 binary answers in training question order')
    p.add_argument('--device', default='auto', help='auto / cpu / cuda / cuda:0')
    args = p.parse_args()
    model, config = load_model(args.checkpoint, args.moment_dir, args.device)
    device = next(model.parameters()).device
    x = torch.from_numpy(load_movement_bin(args.movement, config['seq_len'])).unsqueeze(0).to(device)
    values = np.asarray(json.loads(Path(args.answers).read_text(encoding='utf-8')))
    if values.shape != (30,) or not np.isin(values, [0, 1]).all():
        p.error('--answers must contain a JSON list of 30 binary values')
    answers = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)
    logits = model.predict(x, answers)
    print(json.dumps({'prediction': int(logits.argmax(-1)), 'probabilities': logits.softmax(-1).cpu().tolist()[0]}))

if __name__ == '__main__':
    main()
