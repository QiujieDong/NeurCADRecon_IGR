import os

import numpy as np
import torch
import trimesh

import random
import torch.backends.cudnn as cudnn

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                      for g in scene_or_mesh.geometry.values()))
    else:
        assert (isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def concat_home_dir(path):
    return os.path.join(os.environ['HOME'], 'data', path)


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def to_cuda(torch_obj):
    if torch.cuda.is_available():
        return torch_obj.cuda()
    else:
        return torch_obj

def same_seed(seed):
    """

    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def normalize_mesh_export(mesh, file_out=None):
    # unit to [-0.5, 0.5]
    bounds = mesh.extents
    if bounds.min() == 0.0:
        return

    # translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation)

    # scale to unit cube
    scale = 1.0 / bounds.max()
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_trafo)
    if file_out is not None:
        mesh.export(file_out)
    return mesh


def load_point_cloud_by_file_extension(file_name):
    ext = file_name.split('.')[-1]

    if ext == "npz" or ext == "npy":
        point_set = torch.tensor(np.load(file_name)).float()
    else:
        point_set = torch.tensor(normalize_mesh_export(trimesh.load(file_name, ext)).vertices).float()

    return point_set


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return np.maximum(self.initial * (self.factor ** (epoch // self.interval)), 5.0e-6)


def update_morse_weight(current_iteration, n_iterations, params_list=None, div_decay='linear'):
    # `params`` should be (start_weight, *optional middle, end_weight) where optional middle is of the form [percent, value]*
    # Thus (1e2, 0.5, 1e2 0.7 0.0, 0.0) means that the weight at [0, 0.5, 0.7, 1] of the training process, the weight should
    # be [1e2,1e2,0.0,0.0]. Between these points, the weights change as per the div_decay parameter, e.g. linearly, quintic, step etc.
    # Thus the weight stays at 1e2 from 0-0.5, decay from 1e2 to 0.0 from 0.5-0.75, and then stays at 0.0 from 0.75-1.

    weight = 0.0

    assert len(params_list) >= 2, params_list
    assert len(params_list[1:-1]) % 2 == 0
    decay_params_list = list(zip([params_list[0], *params_list[1:-1][1::2], params_list[-1]], [0, *params_list[1:-1][::2], 1]))

    curr = current_iteration / n_iterations
    we, e = min([tup for tup in decay_params_list if tup[1] >= curr], key=lambda tup: tup[1])
    w0, s = max([tup for tup in decay_params_list if tup[1] <= curr], key=lambda tup: tup[1])

    # Divergence term anealing functions
    if div_decay == 'linear':  # linearly decrease weight from iter s to iter e
        if current_iteration < s * n_iterations:
            weight = w0
        elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
            weight = w0 + (we - w0) * (current_iteration / n_iterations - s) / (e - s)
        else:
            weight = we
    elif div_decay == 'quintic':  # linearly decrease weight from iter s to iter e
        if current_iteration < s * n_iterations:
            weight = w0
        elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
            weight = w0 + (we - w0) * (1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5)
        else:
            weight = we
    elif div_decay == 'step':  # change weight at s
        if current_iteration < s * n_iterations:
            weight = w0
        else:
            weight = we
    elif div_decay == 'none':
        pass
    else:
        raise Warning("unsupported div decay value")

    return weight
