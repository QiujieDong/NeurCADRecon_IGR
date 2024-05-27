import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
from datetime import datetime
from pyhocon import ConfigFactory
import numpy as np
import argparse
import GPUtil
import torch
import utils.general as utils
from model.sample import Sampler
from model.network import gradient
from scipy.spatial import cKDTree
from utils.plots import plot_surface, plot_cuts_axis


class ReconstructionRunner:

    def run(self):

        print("running")

        self.data = self.data.cuda()
        self.data.requires_grad_()

        if self.eval:
            print("evaluating epoch: {0}".format(self.startepoch))
            my_path = os.path.join(self.cur_exp_dir, 'evaluation', str(self.startepoch))

            utils.mkdir_ifnotexists(os.path.join(self.cur_exp_dir, 'evaluation'))
            utils.mkdir_ifnotexists(my_path)
            self.plot_shapes(epoch=self.startepoch, path=my_path, with_cuts=True)
            return

        print("training")

        for epoch in range(self.startepoch, self.nepochs + 1):

            indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))

            cur_data = self.data[indices]

            mnfld_pnts = cur_data[:, :self.d_in]
            mnfld_sigma = self.local_sigma[indices]

            if epoch % self.conf.get_int('train.checkpoint_frequency') == 0:
            # if epoch % self.conf.get_int('train.checkpoint_frequency') == 0:
                print('saving checkpoint: ', epoch)
                self.save_checkpoints(epoch)
                print('plot validation epoch: ', epoch)
                self.plot_shapes(epoch, with_cuts=False)

            # change back to train mode
            self.network.train()
            self.adjust_learning_rate(epoch)

            nonmnfld_pnts = self.sampler.get_points(mnfld_pnts.unsqueeze(0), mnfld_sigma.unsqueeze(0)).squeeze()

            # forward pass

            mnfld_pred = self.network(mnfld_pnts)
            nonmnfld_pred = self.network(nonmnfld_pnts)

            # compute grad

            mnfld_grad = gradient(mnfld_pnts, mnfld_pred)
            nonmnfld_grad = gradient(nonmnfld_pnts, nonmnfld_pred)

            # manifold loss
            mnfld_loss = (mnfld_pred.abs()).mean()

            # eikonal loss
            grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()

            # hessian
            nonmnfld_dx = gradient(nonmnfld_pnts, nonmnfld_grad[:, 0])
            nonmnfld_dy = gradient(nonmnfld_pnts, nonmnfld_grad[:, 1])
            nonmnfld_dz = gradient(nonmnfld_pnts, nonmnfld_grad[:, 2])
            nonmnfld_hessian_term = torch.stack((nonmnfld_dx, nonmnfld_dy, nonmnfld_dz), dim=-1)

            nonmnfld_hessian_term = torch.unsqueeze(nonmnfld_hessian_term, 0)
            nonmnfld_grad = torch.unsqueeze(nonmnfld_grad, 0)

            nonmnfld_hessian_term = torch.cat((nonmnfld_hessian_term, nonmnfld_grad[:, :, :, None]), dim=-1)
            zero_grad = torch.zeros(
                (nonmnfld_grad.shape[0], nonmnfld_grad.shape[1], 1, 1),
                device=nonmnfld_grad.device)
            zero_grad = torch.cat((nonmnfld_grad[:, :, None, :], zero_grad), dim=-1)
            nonmnfld_hessian_term = torch.cat((nonmnfld_hessian_term, zero_grad), dim=-2)
            devlop_loss = (-1. / (nonmnfld_grad.norm(dim=-1) ** 2 + 1e-12)) * torch.det(
                nonmnfld_hessian_term)
            devlop_loss = devlop_loss.mean()


            if self.use_decay_devlop_lambda:
                self.devlop_lambda = utils.update_morse_weight(epoch, self.nepochs, self.decay_params)

            loss = mnfld_loss + self.grad_lambda * grad_loss + self.devlop_lambda * devlop_loss

            if self.with_normals:
                normals = cur_data[:, -self.d_in:]
                normals_loss = ((mnfld_grad - normals).abs()).norm(2, dim=1).mean()
                loss = loss + self.normals_lambda * normals_loss
            else:
                normals_loss = torch.zeros(1)

            # back propagation

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            if epoch % self.conf.get_int('train.status_frequency') == 0:
                print('Train Epoch: [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tManifold loss: {:.6f}'
                      '\tGrad loss: {:.6f}\tNormals Loss: {:.6f}\tDevlop Loss: {:.6f}\tDevlop Lambda: {:.6}'.format(
                    epoch, self.nepochs, 100. * epoch / self.nepochs,
                    loss.item(), mnfld_loss.item(), grad_loss.item(), normals_loss.item(), devlop_loss.item(),
                    self.devlop_lambda))

    def plot_shapes(self, epoch, path=None, with_cuts=False):
        # plot network validation shapes
        with torch.no_grad():

            self.network.eval()

            if not path:
                path = self.plots_dir

            indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))

            pnts = self.data[indices, :3]

            plot_surface(with_points=True,
                         points=pnts,
                         decoder=self.network,
                         path=path,
                         epoch=epoch,
                         shapename=self.expname,
                         **self.conf.get_config('plot'))

            if with_cuts:
                plot_cuts_axis(points=pnts,
                               decoder=self.network,
                               latent=None,
                               path=path,
                               epoch=epoch,
                               near_zero=False,
                               axis=2)

    def __init__(self, **kwargs):

        self.home_dir = os.path.abspath(os.pardir)

        # config setting

        if type(kwargs['conf']) == str:
            self.conf_filename = './reconstruction/' + kwargs['conf']
            self.conf = ConfigFactory.parse_file(self.conf_filename)
        else:
            self.conf = kwargs['conf']

        self.expname = kwargs['expname']

        self.decay_params = kwargs['decay_params']

        self.use_decay_devlop_lambda = kwargs['use_decay_devlop_lambda']

        self.num_of_gpus = torch.cuda.device_count()

        self.eval = kwargs['eval']

        # settings for loading an existing experiment

        if (kwargs['is_continue'] or self.eval) and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join(self.home_dir, 'exps', self.expname)):
                timestamps = os.listdir(os.path.join(self.home_dir, 'exps', self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue'] or self.eval

        self.exps_folder_name = 'exps'

        utils.mkdir_ifnotexists(utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name)))

        self.input_file = kwargs['input_path']
        self.data = utils.load_point_cloud_by_file_extension(self.input_file)

        sigma_set = []
        ptree = cKDTree(self.data)

        for p in np.array_split(self.data, 100, axis=0):
            d = ptree.query(p, 50 + 1)
            sigma_set.append(d[0][:, -1])

        sigmas = np.concatenate(sigma_set)
        self.local_sigma = torch.from_numpy(sigmas).float().cuda()

        self.expdir = utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name, self.expname))
        utils.mkdir_ifnotexists(self.expdir)

        if is_continue:
            self.timestamp = timestamp
        else:
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

        self.cur_exp_dir = os.path.join(self.expdir, self.timestamp)

        self.sub_exp_name = kwargs['sub_exp_name']
        self.exp_path = os.path.join(self.home_dir, self.exps_folder_name, self.expname)
        self.plots_dir = os.path.join(os.path.join(self.exp_path, self.sub_exp_name))
        os.makedirs(self.plots_dir, exist_ok=True)

        self.checkpoints_path = os.path.join(self.plots_dir, 'checkpoints')
        os.makedirs(self.checkpoints_path, exist_ok=True)

        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))

        self.nepochs = kwargs['nepochs']

        self.points_batch = kwargs['points_batch']

        self.global_sigma = self.conf.get_float('network.sampler.properties.global_sigma')
        self.sampler = Sampler.get_sampler(self.conf.get_string('network.sampler.sampler_type'))(self.global_sigma,
                                                                                                 self.local_sigma)
        self.grad_lambda = self.conf.get_float('network.loss.lambda')
        self.normals_lambda = self.conf.get_float('network.loss.normals_lambda')
        self.devlop_lambda = self.decay_params[0]

        # use normals if data has  normals and normals_lambda is positive
        self.with_normals = self.normals_lambda > 0 and self.data.shape[-1] >= 6

        self.d_in = self.conf.get_int('train.d_in')

        self.network = utils.get_class(self.conf.get_string('train.network_class'))(d_in=self.d_in,
                                                                                    **self.conf.get_config(
                                                                                        'network.inputs'))

        if torch.cuda.is_available():
            self.network.cuda()

        self.lr_schedules = self.get_learning_rate_schedules(self.conf.get_list('train.learning_rate_schedule'))
        self.weight_decay = self.conf.get_float('train.weight_decay')

        self.startepoch = 0

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.network.parameters(),
                    "lr": self.lr_schedules[0].get_learning_rate(0),
                    "weight_decay": self.weight_decay
                },
            ])

        # if continue load checkpoints

        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.network.load_state_dict(saved_model_state["model_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
            self.startepoch = saved_model_state['epoch']

    def get_learning_rate_schedules(self, schedule_specs):

        schedules = []

        for schedule_specs in schedule_specs:

            if schedule_specs["Type"] == "Step":
                schedules.append(
                    utils.StepLearningRateSchedule(
                        schedule_specs["Initial"],
                        schedule_specs["Interval"],
                        schedule_specs["Factor"],
                    )
                )

            else:
                raise Exception(
                    'no known learning rate schedule of type "{}"'.format(
                        schedule_specs["Type"]
                    )
                )

        return schedules

    def adjust_learning_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedules[i].get_learning_rate(epoch)

    def save_checkpoints(self, epoch):

        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,
                        default='../data/cube/input/cube.ply',
                        help='the path for input point cloud')
    parser.add_argument('--points_batch', type=int, default=10000, help='point batch size')
    parser.add_argument('--nepoch', type=int, default=20000, help='number of epochs to train for')
    parser.add_argument('--seed', type=int, default=2498504582, help='random seed')
    # parser.add_argument('--conf', type=str, default='setup_original_IGR.conf')
    parser.add_argument('--conf', type=str, default='setup.conf')
    parser.add_argument('--expname', type=str, default='Cube')
    parser.add_argument('--sub_exp_name', type=str, default='log')
    parser.add_argument('--is_continue', default=False, action="store_true", help='continue')
    parser.add_argument('--timestamp', default='latest', type=str)
    parser.add_argument('--checkpoint', default='latest', type=str)
    parser.add_argument('--eval', default=False, action="store_true")
    parser.add_argument('--decay_params', nargs='+', type=float, default=[1e-5, 0.3, 1e-7, 0.7, 1e-7, 0],
                        help='epoch number to evaluate')
    parser.add_argument('--use_decay_devlop_lambda', default=True,
                        help='if True, using decay for devlop_lambda')

    args = parser.parse_args()

    utils.same_seed(args.seed)

    trainrunner = ReconstructionRunner(
        conf=args.conf,
        points_batch=args.points_batch,
        nepochs=args.nepoch,
        expname=args.expname,
        is_continue=args.is_continue,
        timestamp=args.timestamp,
        checkpoint=args.checkpoint,
        eval=args.eval,
        input_path=args.input_path,
        sub_exp_name=args.sub_exp_name,
        decay_params=args.decay_params,
        use_decay_devlop_lambda=args.use_decay_devlop_lambda
    )

    trainrunner.run()
