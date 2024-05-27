import torch
import utils.general as utils
import abc
import scipy.spatial as spatial

import numpy as np


class Sampler(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_points(self, pc_input):
        pass

    @staticmethod
    def get_sampler(sampler_type):

        return utils.get_class("model.sample.{0}".format(sampler_type))


def sample_gaussian_noise_around_shape(manifold_pnts):
    kd_tree = spatial.KDTree(manifold_pnts)
    # query each point for sigma
    dist, _ = kd_tree.query(manifold_pnts, k=50, workers=-1)
    sigmas = dist[:, -1:]
    return sigmas


def samper_near_points(manifold_pnts):
    pnts = manifold_pnts.detach().cpu().numpy()
    manifold_idxes_permutation = np.random.permutation(pnts.shape[0])
    mnfld_idx = manifold_idxes_permutation[:10000]
    sigmas = sample_gaussian_noise_around_shape(pnts)

    near_points = (pnts + sigmas[mnfld_idx] * np.random.randn(pnts.shape[0], pnts.shape[1])).astype(np.float32)
    return torch.from_numpy(near_points).cuda()


class NormalPerPoint(Sampler):

    def __init__(self, global_sigma, local_sigma=0.01):
        self.global_sigma = global_sigma
        self.local_sigma = local_sigma

    def get_points(self, pc_input, local_sigma=None):
        batch_size, sample_size, dim = pc_input.shape

        if local_sigma is not None:
            sample_local = pc_input + (torch.randn_like(pc_input) * local_sigma.unsqueeze(-1))
        else:
            sample_local = pc_input + (torch.randn_like(pc_input) * self.local_sigma)

        # sample_global = (torch.rand(batch_size, sample_size // 8, dim, device=pc_input.device) * (self.global_sigma * 2)) - self.global_sigma
        sample_global = (torch.rand(batch_size, sample_size, dim, device=pc_input.device) * (self.global_sigma * 2)) - self.global_sigma

        sample = torch.cat([sample_local, sample_global], dim=1)

        return sample
