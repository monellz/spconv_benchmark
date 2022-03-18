import sys
import time
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn

import torchsparse as ts
import torchsparse.nn as tsnn

def check(net, coords, feats, result_fn):
  in_tensor = ts.SparseTensor(coords=coords, feats=feats)
  out_tensor = net(in_tensor)
  with open(Path(__file__).parent / "data" / result_fn, "rb") as f:
    (val_coords, val_feats) = pickle.load(f)
  val_coords = torch.from_numpy(val_coords).to(out_tensor.coords.dtype).cuda()
  val_feats = torch.from_numpy(val_feats).to(out_tensor.feats.dtype).cuda()

  c_err = (val_coords - out_tensor.coords).sum()
  f_err = (val_feats - out_tensor.feats).norm()

  if c_err.item() == 0:
    print("coords CHECK PASS")
  else:
    print("coords CHECK FAILED! err:", c_err.item())
  if f_err.item() < 1e-8:
    print("feats CHECK PASS")
  else:
    print("feats CHECK FAILED! err:", f_err.item())
    print(val_feats)
    print(out_tensor.feats)

def main():
  SEED = 0
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)
  cudnn.deterministic = True

  if sys.argv[1] == 'heavy':
    print("Use heavy net")
    net = nn.Sequential(
      tsnn.Conv3d(3, 256, 3, bias=False),
      tsnn.Conv3d(256, 256, 3, bias=False),
      tsnn.Conv3d(256, 256, 3, bias=False),
      tsnn.Conv3d(256, 256, 3, bias=False),
      tsnn.Conv3d(256, 256, 3, bias=False),
      tsnn.Conv3d(256, 256, 3, bias=False),
      tsnn.Conv3d(256, 256, 3, bias=False),
      tsnn.Conv3d(256, 256, 3, bias=False),
      tsnn.Conv3d(256, 256, 3, bias=False),
      tsnn.Conv3d(256, 256, 3, bias=False),
      tsnn.Conv3d(256, 256, 3, bias=False),
      tsnn.Conv3d(256, 256, 3, bias=False),
      tsnn.Conv3d(256, 256, 3, bias=False),
    )
    result_fn = "heavy_result.pkl"
  elif sys.argv[1] == 'light':
    print("Use light net")
    net = nn.Sequential(
      tsnn.Conv3d(3, 64, 3, bias=False),
      tsnn.Conv3d(64, 64, 3, bias=False),
      tsnn.Conv3d(64, 96, 3, bias=False),
      tsnn.Conv3d(96, 96, 3, bias=False),
      tsnn.Conv3d(96, 128, 3, bias=False),
      tsnn.Conv3d(128, 160, 3, bias=False),
      tsnn.Conv3d(160, 160, 3, bias=False),
      tsnn.Conv3d(160, 192, 3, bias=False),
      tsnn.Conv3d(192, 192, 3, bias=False),
      tsnn.Conv3d(192, 224, 3, bias=False),
      tsnn.Conv3d(224, 224, 3, bias=False),
      tsnn.Conv3d(224, 256, 3, bias=False),
      tsnn.Conv3d(256, 256, 3, bias=False),
    )
    result_fn = "light_result.pkl"
  else:
    print("No net! exit")
    exit(-1)

  with open(Path(__file__).parent / "data" / "test_spconv.pkl", "rb") as f:
    (coords, feats) = pickle.load(f)
  print("coords shape:", coords.shape)
  print("feats shape:", feats.shape)

  dtype = torch.float32
  coords = torch.from_numpy(coords).cuda().int()
  feats = torch.from_numpy(feats).cuda().to(dtype)
  net = net.cuda().to(dtype).eval()

  # validation
  check(net, coords, feats, result_fn)

  # benchmark
  times = []
  for i in range(20):
    in_tensor = ts.SparseTensor(coords=coords, feats=feats)
    torch.cuda.synchronize()
    t = time.time()
    out_tensor = net(in_tensor)
    torch.cuda.synchronize()
    times.append(time.time() - t)
  
  print("time:", np.mean(times[10:]) * 1000.0, "ms")

if __name__ == "__main__":
  main()