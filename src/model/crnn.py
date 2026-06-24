"""
CRNN (Convolutional Recurrent Neural Network) for license plate OCR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CHARACTERS, NUM_CLASSES, BLANK_IDX


class CRNN(nn.Module):
    """
    CNN + BiLSTM + FC for CTC-based text recognition.

    Input:  [B, 3, 32, 80]  (HR license plate image)
    Output: [20, B, 37]     (log probabilities per timestep)
    """

    def __init__(self, num_classes=NUM_CLASSES, hidden=256):
        super().__init__()

        def conv_block(in_c, out_c, pool):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )

        self.cnn = nn.Sequential(
            conv_block(3, 32, (2, 2)),
            conv_block(32, 64, (2, 2)),
            conv_block(64, 128, (2, 1)),
        )
        self.rnn = nn.LSTM(
            128 * 4, hidden, num_layers=2,
            bidirectional=True, batch_first=False,
        )
        self.fc = nn.Linear(hidden * 2, num_classes + 1)

    def forward(self, x):
        feat = self.cnn(x)                                  # [B,128,4,20]
        B, C, H, W = feat.shape
        feat = feat.permute(3, 0, 1, 2).reshape(W, B, C * H)  # [20,B,512]
        out, _ = self.rnn(feat)
        return F.log_softmax(self.fc(out), dim=-1)           # [20,B,37]

    def decode_with_probs(self, log_probs):
        """
        CTC greedy decode with per-character probability distributions.

        Returns:
            list of (decoded_string, list_of_char_prob_dicts)
        """
        probs = log_probs.exp()
        results = []
        for b in range(log_probs.shape[1]):
            seq = probs[:, b, :]
            argmaxes = seq.argmax(dim=-1).tolist()
            chars, char_probs, prev = [], [], None
            for t, idx in enumerate(argmaxes):
                if idx == prev:
                    continue
                prev = idx
                if idx == BLANK_IDX:
                    continue
                sp = seq[t, :NUM_CLASSES].cpu()
                sp = sp / (sp.sum() + 1e-8)
                chars.append(CHARACTERS[idx])
                char_probs.append(
                    {CHARACTERS[i]: sp[i].item() for i in range(NUM_CLASSES)}
                )
            results.append(("".join(chars), char_probs))
        return results
