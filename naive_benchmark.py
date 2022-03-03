import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchsparse import nn as spnn
from torchsparse import SparseTensor
from cumm import tensorview as tv
from spconv.core import ConvAlgo

import spconv.pytorch as spconv
from spconv.utils import Point2VoxelCPU3d


def waymo_data(batch_size=1):
    gen = Point2VoxelCPU3d([0.1, 0.1, 0.1], [-80, -80, -2, 80, 80, 6], 3,
                           150000, 1)
    # gen = VoxelGeneratorV2([0.1, 0.1, 0.1], [-80, -80, -2, 80, 80, 6], 1,
    #                        150000)
    data = np.load(Path(__file__).parent / "data" / "benchmark-pc.npz")
    pc = np.ascontiguousarray(data["pc"])
    print(pc.shape)
    voxels_tv, indices_tv, _ = gen.point_to_voxel(tv.from_numpy(pc))
    voxels = voxels_tv.numpy().reshape(-1, 3)
    coors = indices_tv.numpy()
    N = coors.shape[0]
    coors = np.concatenate([np.full([N, 1], 0, coors.dtype), coors], axis=1)
    return voxels, coors, gen.grid_size

class TSNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(3, 64, 3, bias=False),
            spnn.Conv3d(64, 64, 3, bias=False),
            spnn.Conv3d(64, 96, 3, bias=False),
            spnn.Conv3d(96, 96, 3, bias=False),
            spnn.Conv3d(96, 128, 3, bias=False),
            spnn.Conv3d(128, 160, 3, bias=False),
            spnn.Conv3d(160, 160, 3, bias=False),
            spnn.Conv3d(160, 192, 3, bias=False),
            spnn.Conv3d(192, 192, 3, bias=False),
            spnn.Conv3d(192, 224, 3, bias=False),
            spnn.Conv3d(224, 224, 3, bias=False),
            spnn.Conv3d(224, 256, 3, bias=False),
            spnn.Conv3d(256, 256, 3, bias=False),
        )
        max_batch_size = 1
    def forward(self, features, coors, batch_size, enable_timer: bool = False):
        x = SparseTensor(coords=coors, feats=features)
        return self.net(x)


class Net(nn.Module):
    def __init__(self, shape, algo):
        super().__init__()
        pool_algo = algo
        # pool_algo = ConvAlgo.Native
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(3, 64, 3, bias=False, indice_key="c0",
                              algo=algo),
            spconv.SubMConv3d(64,
                              64,
                              3,
                              bias=False,
                              indice_key="c0",
                              algo=algo),
            #spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
            spconv.SubMConv3d(64,
                              96,
                              3,
                              bias=False,
                              indice_key="c1",
                              algo=algo),
            spconv.SubMConv3d(96,
                              96,
                              3,
                              bias=False,
                              indice_key="c1",
                              algo=algo),
            #spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
            spconv.SubMConv3d(96,
                              128,
                              3,
                              bias=False,
                              indice_key="c2",
                              algo=algo),
            spconv.SubMConv3d(128,
                              128,
                              3,
                              bias=False,
                              indice_key="c2",
                              algo=algo),
            #spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
            spconv.SubMConv3d(128,
                              160,
                              3,
                              bias=False,
                              indice_key="c3",
                              algo=algo),
            spconv.SubMConv3d(160,
                              160,
                              3,
                              bias=False,
                              indice_key="c3",
                              algo=algo),
            #spconv.SparseMaxPool3d(2, 2, algo=pool_algo),
            spconv.SubMConv3d(160,
                              192,
                              3,
                              bias=False,
                              indice_key="c4",
                              algo=algo),
            spconv.SubMConv3d(192,
                              192,
                              3,
                              bias=False,
                              indice_key="c4",
                              algo=algo),
            #spconv.SparseMaxPool3d(2, 2, indice_key="m4", algo=pool_algo),
            spconv.SubMConv3d(192,
                              224,
                              3,
                              bias=False,
                              indice_key="c5",
                              algo=algo),
            spconv.SubMConv3d(224,
                              224,
                              3,
                              bias=False,
                              indice_key="c5",
                              algo=algo),
            #spconv.SparseMaxPool3d(2, 2, indice_key="m5", algo=pool_algo),
            spconv.SubMConv3d(224,
                              256,
                              3,
                              bias=False,
                              indice_key="c6",
                              algo=algo),
            spconv.SubMConv3d(256,
                              256,
                              3,
                              bias=False,
                              indice_key="c6",
                              algo=algo),
        )
        max_batch_size = 1
        # grid (dense map) is used for indice generation. use pre-allocated grid can run faster.
        self.grid = torch.full([max_batch_size, *shape], -1,
                               dtype=torch.int32).cuda()
        # self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size, enable_timer: bool = False):
        x = spconv.SparseConvTensor(features,
                                    coors,
                                    self.shape,
                                    batch_size,
                                    self.grid,
                                    enable_timer=enable_timer)
        return self.net(x)


import numpy as np
from cumm import tensorview as tv
from spconv.core_cc.csrc.sparse.all import SpconvOps
import pickle
import torch

from spconv.pytorch.cppcore import torch_tensor_to_tv


def spconv_main():
    import pickle
    np.random.seed(50051)
    torch.manual_seed(50051)
    with open(Path(__file__).parent / "data" / "test_spconv.pkl", "rb") as f:
        (voxels, coors, spatial_shape) = pickle.load(f)
    print(spatial_shape)
    print(voxels.shape)
    #dtype = torch.float16
    dtype = torch.float32
    device = torch.device("cuda:0")
    voxels_th = torch.from_numpy(voxels).to(device).to(dtype)
    coors_th = torch.from_numpy(coors).to(device).int()
    voxels_th.requires_grad = True
    algo = spconv.ConvAlgo.MaskSplitImplicitGemm
    #algo = spconv.ConvAlgo.MaskImplicitGemm
    #algo = spconv.ConvAlgo.Native
    net = Net(spatial_shape, algo).to(device).eval().to(dtype).train()
    spconv.assign_name_for_sparse_modules(net)
    print(coors_th.shape)
    out = net(voxels_th, coors_th, 1)
    print(out.spatial_shape)
    print(voxels.mean(), voxels.max(), voxels.min())
    dout = np.random.uniform(-0.2, 0.2, out.features.shape).astype(np.float32)
    dout_t = torch.from_numpy(dout).to(device).to(dtype)

    print(out.spatial_shape, out.features.mean(), out.features.max(),
          out.features.min())
    times = []
    with torch.no_grad():
        for i in range(20):
            print("------------")
            torch.cuda.synchronize()
            t = time.time()
            out_nograd = net(voxels_th, coors_th, 1, True)
            #timer = out_nograd._timer
            #res = timer.collect_by_name("forward", timer.get_all_pair_time())
            #res2 = timer.collect_by_name("forward0", timer.get_all_pair_time())

            #print(sum(res.values()) + sum(res2.values()))
            torch.cuda.synchronize()
            times.append(time.time() - t)
    print("spconv time", np.mean(times[10:]))

def torchsparse_main():
    import pickle
    np.random.seed(50051)
    torch.manual_seed(50051)
    with open(Path(__file__).parent / "data" / "test_spconv.pkl", "rb") as f:
        (voxels, coors, spatial_shape) = pickle.load(f)
    print(spatial_shape)
    print(voxels.shape)
    #dtype = torch.float16
    dtype = torch.float32
    device = torch.device("cuda:0")
    voxels_th = torch.from_numpy(voxels).to(device).to(dtype)
    coors_th = torch.from_numpy(coors).to(device).int()
    voxels_th.requires_grad = True
    net = TSNet(spatial_shape).to(device).eval().to(dtype).train()
    print(coors_th.shape)
    out = net(voxels_th, coors_th, 1)
    print(out.feats.shape)
    print(out.coords.shape)
    print(voxels.mean(), voxels.max(), voxels.min())

    print(out.feats.mean(), out.feats.max(), out.feats.min())
    times = []
    with torch.no_grad():
        for i in range(20):
            print("------------")
            torch.cuda.synchronize()
            t = time.time()
            out_nograd = net(voxels_th, coors_th, 1, True)
            #timer = out_nograd._timer
            #res = timer.collect_by_name("forward", timer.get_all_pair_time())
            #res2 = timer.collect_by_name("forward0", timer.get_all_pair_time())

            #print(sum(res.values()) + sum(res2.values()))
            torch.cuda.synchronize()
            times.append(time.time() - t)
    print("torchsparse time", np.mean(times[10:]))



if __name__ == "__main__":
    spconv_main()
    torchsparse_main()
