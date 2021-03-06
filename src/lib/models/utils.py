from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    '''
    3ddd feat:torch.Size([1, 30720, 1])
    3ddd ind :torch.Size([1, 50])
    '''
    dim  = feat.size(2)

    # eg:
    # x = torch.Tensor([[1], [2], [3]])
    # y = x.expand(3 , 4)
    # y = tensor([[1., 1., 1., 1.],
    #    [2., 2., 2., 2.],
    #    [3., 3., 3., 3.]])
    # ind shape: batchsize x K x dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)

    # gather usage:
    # eg:
    #   a = Torch.tensor([[1 , 2] , [3 , 4]])
    #   b = torch.gather(a , 1 , torch.LongTensor([[1,0],[1,0]]))
    #   b = Torch.tensor([[2 , 1] , [4 , 3]])
    # gather返回的tensor shape和ind相等
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    
    # 返回的是index: batch * k * 1
    return feat

# feat: batchsize x 2 x 128 x 128
# ind:  batchsize x k
def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    
    feat = feat.view(feat.size(0), -1, feat.size(3))
    # feat: batchsize x (-1) x 2
    # ind:  batchsize x k
    feat = _gather_feat(feat, ind)
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)