"""
Word recognizer: a small fully convolutional network trained with CTC.

Input: ink map of INPUT_HEIGHT x W (seedsigner.helpers.handwriting.preprocess).
Output: for every 4 columns, log-probabilities of 37 classes: 0 is the CTC
blank, 1-26 the letters a-z (case-insensitive), 27-36 the digits 0-9.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
NUM_CLASSES = 1 + len(ALPHABET)



def encode(label: str) -> list[int]:
    return [ALPHABET.index(c) + 1 for c in label]



class WordNet(nn.Module):
    """
    conv 3x3 1->c1, max 2x2 ; conv 3x3 c1->c2, max 2x2 ;
    conv 3x3 c2->c3, max (2,1) ; conv 3x3 c3->c3, max (2,1) ;
    columns -> vectors of 2*c3 (index c * 2 + h) ;
    conv1d k3 -> d ; conv1d k3 dilation 2 -> d ; conv1d k1 -> 37.
    Batch normalization is folded into the weights at export.
    """

    def __init__(self, channels=(16, 32, 64), hidden=128):
        super().__init__()
        c1, c2, c3 = channels

        def conv(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU())

        def conv1d(i, o, dilation):
            return nn.Sequential(nn.Conv1d(i, o, 3, padding=dilation, dilation=dilation, bias=False),
                                 nn.BatchNorm1d(o), nn.ReLU())

        self.b1, self.b2, self.b3, self.b4 = conv(1, c1), conv(c1, c2), conv(c2, c3), conv(c3, c3)
        self.t1, self.t2 = conv1d(2 * c3, hidden, 1), conv1d(hidden, hidden, 2)
        self.out = nn.Conv1d(hidden, NUM_CLASSES, 1)


    def forward(self, x):
        x = F.max_pool2d(self.b1(x), 2)
        x = F.max_pool2d(self.b2(x), 2)
        x = F.max_pool2d(self.b3(x), (2, 1))
        x = F.max_pool2d(self.b4(x), (2, 1))
        x = self.t2(self.t1(x.flatten(1, 2)))
        return F.log_softmax(self.out(x), 1)


    def export(self) -> dict:
        """Float weights with batch normalization folded in."""
        weights = {}

        def fold(conv, bn, name):
            s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            weights[name + ".weight"] = (conv.weight * s.view(-1, *[1] * (conv.weight.dim() - 1))).detach().cpu().numpy()
            weights[name + ".bias"] = (bn.bias - bn.running_mean * s).detach().cpu().numpy()

        for name in ("b1", "b2", "b3", "b4", "t1", "t2"):
            block = getattr(self, name)
            fold(block[0], block[1], name)
        weights["out.weight"] = self.out.weight.detach().cpu().numpy()
        weights["out.bias"] = self.out.bias.detach().cpu().numpy()
        return {k: v.astype(np.float32) for k, v in weights.items()}
