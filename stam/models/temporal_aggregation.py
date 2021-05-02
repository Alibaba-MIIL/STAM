from torch import nn
import torch
# from fastai2.layers import trunc_normal_
from ..utils.utils import trunc_normal_

class TAggregate(nn.Module):
  def __init__(self, clip_length=None, embed_dim=2048, n_layers=6):
    super(TAggregate, self).__init__()
    self.clip_length = clip_length
    # self.nvids = nvids
    # embed_dim = 2048
    drop_rate = 0.
    enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
    self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(
      embed_dim))

    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + 1, embed_dim))
    self.pos_drop = nn.Dropout(p=drop_rate)

    with torch.no_grad():
      trunc_normal_(self.pos_embed, std=.02)
      trunc_normal_(self.cls_token, std=.02)
    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      with torch.no_grad():
        trunc_normal_(m.weight, std=.02)
      if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)


  def forward(self, x):
    nvids = x.shape[0] // self.clip_length
    x = x.view((nvids, self.clip_length, -1))

    cls_tokens = self.cls_token.expand(nvids, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + self.pos_embed
    # x = self.pos_drop(x)

    x.transpose_(1,0)
    o = self.transformer_enc(x)

    return o[0]


class MeanAggregate(nn.Module):
  def __init__(self, sampled_frames=None, nvids=None):
    super(MeanAggregate, self).__init__()
    self.clip_length = sampled_frames
    self.nvids = nvids


  def forward(self, x):
    # nvids = x.shape[0] // self.clip_length
    x = x.view((-1, self.clip_length) + x.size()[1:])
    o = x.mean(dim=1)
    return o
