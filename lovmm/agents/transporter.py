import os
import numpy as np

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from lovmm.tasks import cameras
from lovmm.utils import utils
from lovmm.models.core.attention import Attention
from lovmm.models.core.transport import Transport
from lovmm.models.streams.two_stream_attention import TwoStreamAttention
from lovmm.models.streams.two_stream_transport import TwoStreamTransport

from lovmm.models.streams.two_stream_attention import TwoStreamAttentionLat
from lovmm.models.streams.two_stream_transport import TwoStreamTransportLat

class TransporterAgent(LightningModule):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__()
        utils.set_seed(0)

        if cfg['train'].get('gpu_device') is not None:
            self.device_type = cfg['train'].get('gpu_device')
        else:
            self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # this is bad for PL :(

        self.name = name
        self.cfg = cfg
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.name = name
        self.task = cfg['train']['task']
        self.total_steps = 0
        self.crop_size = 64
        self.n_rotations = cfg['train']['n_rotations']
        # Add regress window size
        self.regress_window = (7,7,1)

        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        self.val_repeats = cfg['train']['val_repeats']
        self.save_steps = cfg['train']['save_steps']

        self._build_model()
        self._optimizers = {
            'attn': torch.optim.Adam(self.attention.parameters(), lr=self.cfg['train']['lr']),
            'trans': torch.optim.Adam(self.transport.parameters(), lr=self.cfg['train']['lr']),
        }
        print("Agent: {}, Logging: {}".format(name, cfg['train']['log']))

    def _build_model(self):
        self.attention = None
        self.transport = None
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def cross_entropy_with_logits(self, pred, labels, reduction='mean'):
        # Lucas found that both sum and mean work equally well
        x = (-labels * F.log_softmax(pred, -1))
        if reduction == 'sum':
            return x.sum()
        elif reduction == 'mean':
            return x.mean()
        else:
            raise NotImplementedError()

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']

        output = self.attention.forward(inp_img, softmax=softmax)
        return output

    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']

        inp = {'inp_img': inp_img}
        out = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta)

    def attn_criterion(self, backprop, compute_err, inp, out, p, theta):
        # Get label.
        theta_i = theta / (2 * np.pi / self.attention.n_rotations)
        theta_i = np.int32(np.round(theta_i)) % self.attention.n_rotations
        inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.attention.n_rotations,)
        label = np.zeros(label_size)
        label[p[0], p[1], theta_i] = 1
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=out.device)

        # Get loss.
        loss = self.cross_entropy_with_logits(out, label)

        # Backpropagate.
        if backprop:
            attn_optim = self._optimizers['attn']
            self.manual_backward(loss, attn_optim)
            attn_optim.step()
            attn_optim.zero_grad()

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            pick_conf = self.attn_forward(inp)
            pick_conf = pick_conf.detach().cpu().numpy()
            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            p0_pix = argmax[:2]
            p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

            err = {
                'dist': np.linalg.norm(np.array(p) - p0_pix, ord=1),
                'theta': abs((np.rad2deg(theta - p0_theta) + 180) % 360 - 180)
            }
        return loss, err

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']

        output = self.transport.forward(inp_img, p0, softmax=softmax)
        return output

    def transport_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']

        inp = {'inp_img': inp_img, 'p0': p0}
        output = self.trans_forward(inp, softmax=False)
        err, loss = self.transport_criterion(backprop, compute_err, inp, output, p0, p1, p1_theta)
        return loss, err

    def transport_criterion(self, backprop, compute_err, inp, output, p, q, theta):
        itheta = theta / (2 * np.pi / self.transport.n_rotations)
        itheta = np.int32(np.round(itheta)) % self.transport.n_rotations

        # Get one-hot pixel label map.
        if inp.get('inp_img_place') is not None:
            inp_img = inp['inp_img_place']
        else:
            inp_img = inp['inp_img']
        label_size = inp_img.shape[:2] + (self.transport.n_rotations,)
        label = np.zeros(label_size)
        label[q[0], q[1], itheta] = 1

        # Get loss.
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=output.device)
        output = output.reshape(1, np.prod(output.shape))
        loss = self.cross_entropy_with_logits(output, label)
        if backprop:
            transport_optim = self._optimizers['trans']
            self.manual_backward(loss, transport_optim)
            transport_optim.step()
            transport_optim.zero_grad()
 
        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            place_conf = self.trans_forward(inp)
            place_conf = place_conf.permute(1, 2, 0)
            place_conf = place_conf.detach().cpu().numpy()
            argmax = np.argmax(place_conf)
            argmax = np.unravel_index(argmax, shape=place_conf.shape)
            p1_pix = argmax[:2]
            p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

            err = {
                'dist': np.linalg.norm(np.array(q) - p1_pix, ord=1),
                'theta': abs((np.rad2deg(theta - p1_theta) + 180) % 360 - 180)
            }
        self.transport.iters += 1
        return err, loss

    def training_step(self, batch, batch_idx):
        self.attention.train()
        self.transport.train()

        frame, _ = batch

        # Get training losses.
        step = self.total_steps + 1
        loss0, err0 = self.attn_training_step(frame)
        if isinstance(self.transport, Attention):
            loss1, err1 = self.attn_training_step(frame)
        else:
            loss1, err1 = self.transport_training_step(frame)
        total_loss = loss0 + loss1
        self.log('tr/attn/loss', loss0)
        self.log('tr/trans/loss', loss1)
        self.log('tr/loss', total_loss)
        self.total_steps = step

        self.trainer.train_loop.running_loss.append(total_loss)

        self.check_save_iteration()

        return dict(
            loss=total_loss,
        )

    def check_save_iteration(self):
        global_step = self.trainer.global_step
        if (global_step + 1) in self.save_steps:
            self.trainer.run_evaluation()
            val_loss = self.trainer.callback_metrics['val_loss']
            steps = f'{global_step + 1:05d}'
            filename = f"steps={steps}-val_loss={val_loss:0.8f}.ckpt"
            checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
            ckpt_path = os.path.join(checkpoint_path, filename)
            self.trainer.save_checkpoint(ckpt_path)

        if (global_step + 1) % 1000 == 0:
            # save lastest checkpoint
            # print(f"Saving last.ckpt Epoch: {self.trainer.current_epoch} | Global Step: {self.trainer.global_step}")
            self.save_last_checkpoint()

    def save_last_checkpoint(self):
        checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
        ckpt_path = os.path.join(checkpoint_path, 'last.ckpt')
        self.trainer.save_checkpoint(ckpt_path)

    def validation_step(self, batch, batch_idx):
        self.attention.eval()
        self.transport.eval()

        loss0, loss1 = 0, 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame, _ = batch
            l0, err0 = self.attn_training_step(frame, backprop=False, compute_err=True)
            loss0 += l0
            if isinstance(self.transport, Attention):
                l1, err1 = self.attn_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
            else:
                l1, err1 = self.transport_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
        loss0 /= self.val_repeats
        loss1 /= self.val_repeats
        val_total_loss = loss0 + loss1

        self.trainer.evaluation_loop.trainer.train_loop.running_loss.append(val_total_loss)

        return dict(
            val_loss=val_total_loss,
            val_loss0=loss0,
            val_loss1=loss1,
            val_attn_dist_err=err0['dist'],
            val_attn_theta_err=err0['theta'],
            val_trans_dist_err=err1['dist'],
            val_trans_theta_err=err1['theta'],
        )

    def training_epoch_end(self, all_outputs):
        super().training_epoch_end(all_outputs)
        utils.set_seed(self.trainer.current_epoch+1)

    def validation_epoch_end(self, all_outputs):
        mean_val_total_loss = np.mean([v['val_loss'].item() for v in all_outputs])
        mean_val_loss0 = np.mean([v['val_loss0'].item() for v in all_outputs])
        mean_val_loss1 = np.mean([v['val_loss1'].item() for v in all_outputs])
        
        mean_attn_dist_err = np.mean([v['val_attn_dist_err'] for v in all_outputs])
        mean_attn_theta_err = np.mean([v['val_attn_theta_err'] for v in all_outputs])
        mean_trans_dist_err = np.mean([v['val_trans_dist_err'] for v in all_outputs])
        mean_trans_theta_err = np.mean([v['val_trans_theta_err'] for v in all_outputs])

        total_attn_dist_err = np.sum([v['val_attn_dist_err'] for v in all_outputs])
        total_attn_theta_err = np.sum([v['val_attn_theta_err'] for v in all_outputs])
        total_trans_dist_err = np.sum([v['val_trans_dist_err'] for v in all_outputs])
        total_trans_theta_err = np.sum([v['val_trans_theta_err'] for v in all_outputs])

        self.log('vl/attn/loss', mean_val_loss0)
        self.log('vl/trans/loss', mean_val_loss1)
        self.log('vl/loss', mean_val_total_loss)
        self.log('vl/total_attn_dist_err', total_attn_dist_err)
        self.log('vl/total_attn_theta_err', total_attn_theta_err)
        self.log('vl/total_trans_dist_err', total_trans_dist_err)
        self.log('vl/total_trans_theta_err', total_trans_theta_err)
        self.log('vl/mean_attn_dist_err', mean_attn_dist_err)
        self.log('vl/mean_attn_theta_err', mean_attn_theta_err)
        self.log('vl/mean_trans_dist_err', mean_trans_dist_err)
        self.log('vl/mean_trans_theta_err', mean_trans_theta_err)

        print("\nAttn Mean Err - Dist: {:.2f}, Theta: {:.2f}".format(mean_attn_dist_err, mean_attn_theta_err))
        print("Transport Mean Err - Dist: {:.2f}, Theta: {:.2f}".format(mean_trans_dist_err, mean_trans_theta_err))
        # print("\nAttn Err - Dist: {:.2f}, Theta: {:.2f}".format(total_attn_dist_err, total_attn_theta_err))
        # print("Transport Err - Dist: {:.2f}, Theta: {:.2f}".format(total_trans_dist_err, total_trans_theta_err))


        return dict(
            val_loss=mean_val_total_loss,
            val_loss0=mean_val_loss0,
            mean_val_loss1=mean_val_loss1,
            total_attn_dist_err=total_attn_dist_err,
            total_attn_theta_err=total_attn_theta_err,
            total_trans_dist_err=total_trans_dist_err,
            total_trans_theta_err=total_trans_theta_err,
        )

    def act(self, obs, info=None, goal=None, bounds_habitat=None,pixel_size_habitat=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.

        # TODO: REVISE here to directly get ortho image
        # img = self.test_ds.get_image(obs)
        cmap = obs['color'][0]
        hmap = obs['depth'][0]

        img = np.concatenate((cmap,
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None]), axis=2)

        # Attention model forward pass.
        pick_inp = {'inp_img': img}
        pick_conf = self.attn_forward(pick_inp)
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_inp = {'inp_img': img, 'p0': p0_pix}
        place_conf = self.trans_forward(place_inp)
        place_conf = place_conf.permute(1, 2, 0)
        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        # TODO: REVISE HERE TO CONVERT HABITAT FRAME

        p0_xyz = utils.pix_to_xyz_habitat(p0_pix, hmap, bounds_habitat, pixel_size_habitat)
        p1_xyz = utils.pix_to_xyz_habitat(p1_pix, hmap, bounds_habitat, pixel_size_habitat)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW_habitat((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW_habitat((0, 0, -p1_theta))

        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            'pick': p0_pix,
            'place': p1_pix,
        }

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure, on_tpu, using_native_amp, using_lbfgs):
        pass

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        return self.train_ds

    def val_dataloader(self):
        return self.test_ds

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path)['state_dict'])
        self.to(device=self.device_type)


class Transporter6dAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)
        self._optimizers = {
            'attn': torch.optim.Adam(self.attention.parameters(), lr=self.cfg['train']['lr']),
            'trans': torch.optim.Adam(self.transport.parameters(), lr=self.cfg['train']['lr']),
            'trans6d': torch.optim.Adam(self.transport6d.parameters(), lr=self.cfg['train']['lr']),
        }

    def _build_model(self):
        self.attention = None
        self.transport = None
        self.transport6d = None
        self.z_regressor = None
        self.roll_regressor = None
        self.pitch_regressor = None
        raise NotImplementedError()

    def trans6d_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        p0 = inp['p0']

        output = self.transport.forward(inp_img, p0, softmax=softmax)
        return output

    def transport6d_criterion(self, backprop, compute_err, inp, output, p, q, theta, z_predict,
                              roll_predict, pitch_predict):
        label_z = inp['p1_z']
        label_roll = inp['p1_roll']
        label_pitch = inp['p1_pitch']
        # TODO try MSE?
        lossfunction = torch.nn.SmoothL1Loss()

        loss_z = lossfunction(z_predict.squeeze(), torch.tensor(label_z).to(dtype=torch.float, device=self.device))
        loss_roll = lossfunction(roll_predict.squeeze(),
                                 torch.tensor(label_roll).to(dtype=torch.float, device=self.device))
        loss_pitch = lossfunction(pitch_predict.squeeze(),
                                  torch.tensor(label_pitch).to(dtype=torch.float, device=self.device))

        z_weight = 10.0
        roll_weight = 10.0
        pitch_weight = 10.0

        loss = z_weight * loss_z + roll_weight * loss_roll + pitch_weight * loss_pitch

        if backprop:
            transport6d_optim = self._optimizers['trans6d']
            self.manual_backward(loss, transport6d_optim)
            transport6d_optim.step()
            transport6d_optim.zero_grad()

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            err = {
                'z': np.linalg.norm(label_z - z_predict.item()),
                'roll': abs((np.rad2deg(label_roll - roll_predict.item()) + 180) % 360 - 180),
                'pitch': abs((np.rad2deg(label_pitch - pitch_predict.item()) + 180) % 360 - 180),
            }

        self.transport.iters += 1
        return err, loss

    def transport6d_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']

        inp = {'inp_img': inp_img, 'p0': p0}
        output = self.trans6d_forward(inp, softmax=False)
        err, loss = self.transport6d_criterion(backprop, compute_err, inp, output, p0, p1, p1_theta)
        return loss, err

    def training_step(self, batch, batch_idx):
        self.attention.train()
        self.transport.train()
        # self.transport6d.train()
        frame, _ = batch

        # Get training losses.
        step = self.total_steps + 1
        loss0, err0 = self.attn_training_step(frame)
        if isinstance(self.transport, Attention):
            loss1, err1 = self.attn_training_step(frame)
        else:
            loss1, err1 = self.transport_training_step(frame)
            pass
        # TODO: First stage training, transport6d does not participate
        loss2 = loss1
        err2 = {
             'z': -1.0,
             'theta': -1.0,
             'roll': -1.0,
             'pitch': -1.0,
         }
        # loss2, err2 = self.transport6d_training_step(frame)
        # loss1 = loss2
        # err1 = err2
        total_loss = loss0 + loss1 + loss2
        self.log('tr/attn/loss', loss0)
        self.log('tr/trans/loss', loss1)
        self.log('tr/trans6d/loss', loss2)
        self.log('tr/loss', total_loss)
        self.total_steps = step

        self.trainer.train_loop.running_loss.append(total_loss)

        self.check_save_iteration()

        return dict(
            loss=total_loss,
        )

    def validation_step(self, batch, batch_idx):
        self.attention.eval()
        self.transport.eval()
        # self.transport6d.eval()

        loss0, loss1, loss2 = 0, 0, 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame, _ = batch
            l0, err0 = self.attn_training_step(frame, backprop=False, compute_err=True)
            loss0 += l0
            if isinstance(self.transport, Attention):
                l1, err1 = self.attn_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
            else:
                l1, err1 = self.transport_training_step(frame, backprop=False, compute_err=True)
                loss1 += l1
            # TODO: First stage training, transport6d does not participate
            l2 = loss1
            err2 = {
                'z': -1.0,
                'theta': -1.0,
                'roll': -1.0,
                'pitch': -1.0,
            }
            # l2, err2 = self.transport6d_training_step(frame, backprop=False, compute_err=True)
            loss2 += l2
        loss0 /= self.val_repeats
        loss1 /= self.val_repeats
        loss2 /= self.val_repeats

        val_total_loss = loss0 + loss1 + loss2

        self.trainer.evaluation_loop.trainer.train_loop.running_loss.append(val_total_loss)

        return dict(
            val_loss=val_total_loss,
            val_loss0=loss0,
            val_loss1=loss1,
            val_loss2=loss2,
            val_attn_dist_err=err0['dist'],
            val_attn_theta_err=err0['theta'],
            val_trans_dist_err=err1['dist'],
            val_trans_theta_err=err1['theta'],
            val_trans6d_z_err=err2['z'],
            val_trans6d_roll_err=err2['roll'],
            val_trans6d_pitch_err=err2['pitch'],
        )

    def validation_epoch_end(self, all_outputs):
        mean_val_total_loss = np.mean([v['val_loss'].item() for v in all_outputs])
        mean_val_loss0 = np.mean([v['val_loss0'].item() for v in all_outputs])
        mean_val_loss1 = np.mean([v['val_loss1'].item() for v in all_outputs])
        mean_val_loss2 = np.mean([v['val_loss2'].item() for v in all_outputs])
        total_attn_dist_err = np.sum([v['val_attn_dist_err'] for v in all_outputs])
        total_attn_theta_err = np.sum([v['val_attn_theta_err'] for v in all_outputs])
        total_trans_dist_err = np.sum([v['val_trans_dist_err'] for v in all_outputs])
        total_trans_theta_err = np.sum([v['val_trans_theta_err'] for v in all_outputs])
        total_trans6d_z_err = np.sum([v['val_trans6d_z_err'] for v in all_outputs])
        total_trans6d_roll_err = np.sum([v['val_trans6d_roll_err'] for v in all_outputs])
        total_trans6d_pitch_err = np.sum([v['val_trans6d_pitch_err'] for v in all_outputs])
        mean_attn_dist_err = np.mean([v['val_attn_dist_err'] for v in all_outputs])
        mean_attn_theta_err = np.mean([v['val_attn_theta_err'] for v in all_outputs])
        mean_trans_dist_err = np.mean([v['val_trans_dist_err'] for v in all_outputs])
        mean_trans_theta_err = np.mean([v['val_trans_theta_err'] for v in all_outputs])
        mean_trans6d_z_err = np.mean([v['val_trans6d_z_err'] for v in all_outputs])
        mean_trans6d_roll_err = np.mean([v['val_trans6d_roll_err'] for v in all_outputs])
        mean_trans6d_pitch_err = np.mean([v['val_trans6d_pitch_err'] for v in all_outputs])

        self.log('vl/attn/loss', mean_val_loss0)
        self.log('vl/trans/loss', mean_val_loss1)
        self.log('vl/trans6d/loss', mean_val_loss2)
        self.log('vl/loss', mean_val_total_loss)
        self.log('vl/total_attn_dist_err', total_attn_dist_err)
        self.log('vl/total_attn_theta_err', total_attn_theta_err)
        self.log('vl/total_trans_dist_err', total_trans_dist_err)
        self.log('vl/total_trans_theta_err', total_trans_theta_err)
        self.log('vl/total_trans6d_z_err', total_trans6d_z_err)
        self.log('vl/total_trans6d_roll_err', total_trans6d_roll_err)
        self.log('vl/total_trans6d_pitch_err', total_trans6d_pitch_err)
        self.log('vl/total_attn_dist_err', mean_attn_dist_err)
        self.log('vl/total_attn_theta_err', mean_attn_theta_err)
        self.log('vl/total_trans_dist_err', mean_trans_dist_err)
        self.log('vl/total_trans_theta_err', mean_trans_theta_err)
        self.log('vl/total_trans6d_z_err', mean_trans6d_z_err)
        self.log('vl/total_trans6d_roll_err', mean_trans6d_roll_err)
        self.log('vl/total_trans6d_pitch_err', mean_trans6d_pitch_err)

        print("\nAttn Err - Dist: {:.2f}, Theta: {:.2f}".format(mean_attn_dist_err, mean_attn_theta_err))
        print("Transport Err - Dist: {:.2f}, Theta: {:.2f}".format(mean_trans_dist_err, mean_trans_theta_err))
        print("Transport6d Err - Z: {:.2f}, Roll: {:.2f}, Pitch: {:.2f}".format(mean_trans6d_z_err,
                                                                                               mean_trans6d_roll_err,
                                                                                               mean_trans6d_pitch_err))
        print("Attn Mean Loss: {:.2f}*10e-4, Transport Mean Loss: {:.2f}*10e-4, Transport6d Mean Loss: {:.2f}".format(
            mean_val_loss0 * 10000.0, mean_val_loss1 * 10000.0, mean_val_loss2))

        return dict(
            val_loss=mean_val_total_loss,
            val_loss0=mean_val_loss0,
            mean_val_loss1=mean_val_loss1,
            mean_val_loss2=mean_val_loss2,
            total_attn_dist_err=total_attn_dist_err,
            total_attn_theta_err=total_attn_theta_err,
            total_trans_dist_err=total_trans_dist_err,
            total_trans_theta_err=total_trans_theta_err,
            total_trans6d_z_err=total_trans6d_z_err,
            total_trans6d_roll_err=total_trans6d_roll_err,
            total_trans6d_pitch_err=total_trans6d_pitch_err,
        )


class OriginalTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'plain_resnet'
        self.attention = Attention(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = Transport(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class ClipUNetTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'clip_unet'
        self.attention = Attention(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = Transport(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamClipUNetTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'clip_unet'
        self.attention = TwoStreamAttention(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransport(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamClipUNetLatTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'clip_unet_lat'
        self.attention = TwoStreamAttentionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamClipWithoutSkipsTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        # TODO: lateral version
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'clip_woskip'
        self.attention = TwoStreamAttention(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransport(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamRN50BertUNetTransporterAgent(TransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        # TODO: lateral version
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'rn50_bert_unet'
        self.attention = TwoStreamAttention(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransport(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
