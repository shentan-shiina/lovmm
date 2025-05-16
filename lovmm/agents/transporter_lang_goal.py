import numpy as np
import torch

from lovmm.utils import utils
from lovmm.agents.transporter import TransporterAgent
from lovmm.agents.transporter import Transporter6dAgent
from lovmm.models.regressor import Regressor

from lovmm.models.streams.one_stream_attention_lang_fusion import OneStreamAttentionLangFusion
from lovmm.models.streams.one_stream_transport_lang_fusion import OneStreamTransportLangFusion
from lovmm.models.streams.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusion
from lovmm.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusion
from lovmm.models.streams.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusionLat
from lovmm.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusionLat
from lovmm.models.streams.two_stream_transport_lang_fusion import TwoStreamTransport6dLangFusionLat
from lovmm.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusionLatOriginal

class TwoStreamClipLingUNetTransporterAgent(TransporterAgent):
    # CLIPORT parent class
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'clip_lingunet'
        self.attention = TwoStreamAttentionLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        print(self.device_type)
        self.transport = TwoStreamTransportLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        print(self.device_type)
    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        lang_goal = inp['lang_goal']

        out = self.attention.forward(inp_img, lang_goal, softmax=softmax)
        return out

    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']
        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}
        out = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta)

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        inp_img_place = inp['inp_img_place']

        p0 = inp['p0']
        lang_goal = inp['lang_goal']

        out = self.transport.forward(inp_img, p0, lang_goal, softmax=softmax, inp_img_place=inp_img_place)
        return out

    def transport_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']
        lang_goal = frame['lang_goal']

        # Cross-workspace target place image as key image
        if frame.get('img_place') is not None:
            inp_img_place = frame['img_place']
        else:
            inp_img_place = inp_img

        inp = {'inp_img': inp_img, 'inp_img_place': inp_img_place, 'p0': p0, 'lang_goal': lang_goal}
        out = self.trans_forward(inp, softmax=False)
        err, loss = self.transport_criterion(backprop, compute_err, inp, out, p0, p1, p1_theta)
        return loss, err

    def act(self, obs, info, goal=None, bounds_habitat=None,pixel_size_habitat=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        # img = self.test_ds.get_image(obs)
        cmap = obs['color'][0]
        hmap = obs['depth'][0]

        img = np.concatenate((cmap,
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None]), axis=2)

        lang_goal = info['lang_goal']

        # Attention model forward pass.
        pick_inp = {'inp_img': img, 'lang_goal': lang_goal}
        pick_conf = self.attn_forward(pick_inp)
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_inp = {'inp_img': img, 'p0': p0_pix, 'lang_goal': lang_goal}
        place_conf = self.trans_forward(place_inp)
        place_conf = place_conf.permute(1, 2, 0)
        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        # Pixels to end effector poses.
        # TODO: change to bullet frame from habitat pixel location
        hmap = img[:, :, 3]

        p0_xyz = utils.pix_to_xyz_habitat(p0_pix, hmap, bounds_habitat, pixel_size_habitat)
        p1_xyz = utils.pix_to_xyz_habitat(p1_pix, hmap, bounds_habitat, pixel_size_habitat)
        p0_xyzw = utils.eulerXYZ_to_quatXYZW_habitat((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW_habitat((0, 0, -p1_theta))

        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            'pick': [p0_pix[0], p0_pix[1], p0_theta],
            'place': [p1_pix[0], p1_pix[1], p1_theta],
        }


class LOVMM(Transporter6dAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn_attn = 'plain_resnet_lat_add'
        stream_two_fcn_attn = 'clip_lingunet_lat'
        stream_one_fcn = 'plain_resnet_lat_add'
        stream_two_fcn = 'clip_lingunet_lat'
        self.attention = TwoStreamAttentionLangFusionLat(
            stream_fcn=(stream_one_fcn_attn, stream_two_fcn_attn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        print(self.device_type)
        self.transport6d = TwoStreamTransport6dLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        # print(self.device_type)
        # self.transport6d.load_state_dict(self.transport.state_dict())

        self.z_regressor = Regressor(
            (self.regress_window[0] * 2 + 1) * (self.regress_window[1] * 2 + 1) * (self.regress_window[2] * 2 + 1))
        self.roll_regressor = Regressor(
            (self.regress_window[0] * 2 + 1) * (self.regress_window[1] * 2 + 1) * (self.regress_window[2] * 2 + 1))
        self.pitch_regressor = Regressor(
            (self.regress_window[0] * 2 + 1) * (self.regress_window[1] * 2 + 1) * (self.regress_window[2] * 2 + 1))

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        lang_goal = inp['lang_goal']

        out = self.attention.forward(inp_img, lang_goal, softmax=softmax)
        return out

    def attn_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0, p0_theta = frame['p0'], frame['p0_theta']
        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}
        out = self.attn_forward(inp, softmax=False)
        # backprop = False
        return self.attn_criterion(backprop, compute_err, inp, out, p0, p0_theta)

    def trans_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        inp_img_place = inp['inp_img_place']

        p0 = inp['p0']
        lang_goal = inp['lang_goal']

        out = self.transport.forward(inp_img, p0, lang_goal, softmax=softmax, inp_img_place=inp_img_place)
        return out

    def transport_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']
        lang_goal = frame['lang_goal']

        # Cross-workspace target place image as key image
        if frame.get('img_place') is not None:
            inp_img_place = frame['img_place']
        else:
            inp_img_place = inp_img

        inp = {'inp_img': inp_img, 'inp_img_place': inp_img_place, 'p0': p0, 'lang_goal': lang_goal}
        out = self.trans_forward(inp, softmax=False)
        # backprop = False
        err, loss = self.transport_criterion(backprop, compute_err, inp, out, p0, p1, p1_theta)
        return loss, err

    def trans6d_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        inp_img_place = inp['inp_img_place']

        p0 = inp['p0']
        lang_goal = inp['lang_goal']

        out = self.transport6d.forward6d(inp_img, p0, lang_goal, softmax=softmax, inp_img_place=inp_img_place)
        return out

    def transport6d_training_step(self, frame, backprop=True, compute_err=False):
        inp_img = frame['img']
        p0 = frame['p0']
        p1, p1_theta = frame['p1'], frame['p1_theta']
        p1_z, p1_roll, p1_pitch = frame['p1_z'], frame['p1_roll'], frame['p1_pitch'],
        lang_goal = frame['lang_goal']

        # Cross-workspace target place image as key image
        if frame.get('img_place') is not None:
            inp_img_place = frame['img_place']
        else:
            inp_img_place = inp_img

        inp = {'inp_img': inp_img, 'inp_img_place': inp_img_place, 'p0': p0, 'lang_goal': lang_goal,
               'p1_z': p1_z, 'p1_theta':p1_theta,'p1_roll': p1_roll, 'p1_pitch': p1_pitch}

        out = self.trans6d_forward(inp, softmax=False)

        z_tensor, roll_tensor, pitch_tensor = out
        z_tensor = z_tensor.permute(0, 2, 3, 1)
        roll_tensor = roll_tensor.permute(0, 2, 3, 1)
        pitch_tensor = pitch_tensor.permute(0, 2, 3, 1)

        # Get one-hot pixel label map and 6DoF labels.
        itheta = p1_theta / (2 * np.pi / self.n_rotations)
        itheta = np.int32(np.round(itheta)) % self.n_rotations

        # Use a window for regression rather than only exact.
        u_window = self.regress_window[0]
        v_window = self.regress_window[1]
        theta_window = self.regress_window[2]

        # Before padding, use top and down window limitation
        # u_min = max(p1[0] - u_window, 0)
        # u_max = min(p1[0] + u_window + 1, z_tensor.shape[1])
        # v_min = max(p1[1] - v_window, 0)
        # v_max = min(p1[1] + v_window + 1, z_tensor.shape[2])
        # theta_min = max(itheta - theta_window, 0)
        # theta_max = min(itheta + theta_window + 1, z_tensor.shape[3])

        # (theta, v, u)
        tensor_padding = (0, 0, v_window, v_window, u_window, u_window)

        z_tensor_pad = torch.nn.functional.pad(z_tensor, tensor_padding)
        roll_tensor_pad = torch.nn.functional.pad(roll_tensor, tensor_padding)
        pitch_tensor_pad = torch.nn.functional.pad(pitch_tensor, tensor_padding)

        u_min_bound = p1[0] - v_window
        u_max_bound = p1[0] + u_window + 1
        v_min_bound = p1[1] - v_window
        v_max_bound = p1[1] + v_window + 1

        # Padding
        u_min = u_min_bound + u_window
        u_max = u_max_bound + u_window
        v_min = v_min_bound + v_window
        v_max = v_max_bound + v_window

        theta_min_bound = itheta - theta_window
        theta_max_bound = itheta + theta_window + 1

        if theta_min_bound < 0:
            theta_min = self.n_rotations + theta_min_bound
            theta_max = theta_max_bound
        elif theta_max_bound >= self.n_rotations:
            theta_max = theta_max_bound - self.n_rotations
            theta_min = theta_min_bound
        else:
            theta_max = theta_max_bound
            theta_min = theta_min_bound

        if theta_max < theta_min:
            z_est_at_xytheta_1 = z_tensor_pad[0, u_min:u_max, v_min:v_max,
                                 theta_min:]
            z_est_at_xytheta_2 = z_tensor_pad[0, u_min:u_max, v_min:v_max,
                                 :theta_max]
            roll_est_at_xytheta_1 = roll_tensor_pad[0, u_min:u_max, v_min:v_max,
                                    theta_min:]
            roll_est_at_xytheta_2 = roll_tensor_pad[0, u_min:u_max, v_min:v_max,
                                    :theta_max]
            pitch_est_at_xytheta_1 = pitch_tensor_pad[0, u_min:u_max, v_min:v_max,
                                     theta_min:]
            pitch_est_at_xytheta_2 = pitch_tensor_pad[0, u_min:u_max, v_min:v_max,
                                     :theta_max]

            z_est_at_xytheta = torch.cat((z_est_at_xytheta_1, z_est_at_xytheta_2), dim=-1)
            roll_est_at_xytheta = torch.cat((roll_est_at_xytheta_1, roll_est_at_xytheta_2), dim=-1)
            pitch_est_at_xytheta = torch.cat((pitch_est_at_xytheta_1, pitch_est_at_xytheta_2), dim=-1)
        else:
            z_est_at_xytheta = z_tensor_pad[0, u_min:u_max, v_min:v_max,
                               theta_min:theta_max]
            roll_est_at_xytheta = roll_tensor_pad[0, u_min:u_max, v_min:v_max,
                                  theta_min:theta_max]
            pitch_est_at_xytheta = pitch_tensor_pad[0, u_min:u_max, v_min:v_max,
                                   theta_min:theta_max]

        z_est_at_xytheta = z_est_at_xytheta.reshape((1, -1))
        roll_est_at_xytheta = roll_est_at_xytheta.reshape((1, -1))
        pitch_est_at_xytheta = pitch_est_at_xytheta.reshape((1, -1))

        z_est_at_xytheta = self.z_regressor(z_est_at_xytheta)
        roll_est_at_xytheta = self.roll_regressor(roll_est_at_xytheta)
        pitch_est_at_xytheta = self.pitch_regressor(pitch_est_at_xytheta)

        # backprop = False
        err, loss = self.transport6d_criterion(backprop, compute_err, inp, out, p0, p1, p1_theta,
                                               z_est_at_xytheta, roll_est_at_xytheta,
                                               pitch_est_at_xytheta)
        return loss, err

    def act(self, obs, info, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        img = self.test_ds.get_image(obs)
        lang_goal = info['lang_goal']

        # Attention model forward pass.
        pick_inp = {'inp_img': img, 'lang_goal': lang_goal}
        pick_conf = self.attn_forward(pick_inp)
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_inp = {'inp_img': img, 'p0': p0_pix, 'lang_goal': lang_goal}
        place_conf = self.trans_forward(place_inp)
        place_conf = place_conf.permute(1, 2, 0)
        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        # Modification for 6 dof
        place_conf = self.trans6d_forward(place_inp)

        out = self.trans6d_forward(place_inp, softmax=False)

        z_tensor, roll_tensor, pitch_tensor = out
        z_tensor=z_tensor.permute(0, 2, 3, 1)
        roll_tensor=roll_tensor.permute(0, 2, 3, 1)
        pitch_tensor=pitch_tensor.permute(0, 2, 3, 1)

        # Get one-hot pixel label map and 6DoF labels.
        itheta = p1_theta / (2 * np.pi / self.n_rotations)
        itheta = np.int32(np.round(itheta)) % self.n_rotations

        # Use a window for regression rather than only exact.
        u_window = self.regress_window[0]
        v_window = self.regress_window[1]
        theta_window = self.regress_window[2]

        # Before padding, use top and down window limitation
        # u_min = max(p1[0] - u_window, 0)
        # u_max = min(p1[0] + u_window + 1, z_tensor.shape[1])
        # v_min = max(p1[1] - v_window, 0)
        # v_max = min(p1[1] + v_window + 1, z_tensor.shape[2])
        # theta_min = max(itheta - theta_window, 0)
        # theta_max = min(itheta + theta_window + 1, z_tensor.shape[3])

        # (theta, v, u)
        tensor_padding = (0, 0, v_window, v_window, u_window, u_window)

        z_tensor_pad = torch.nn.functional.pad(z_tensor, tensor_padding)
        roll_tensor_pad = torch.nn.functional.pad(roll_tensor, tensor_padding)
        pitch_tensor_pad = torch.nn.functional.pad(pitch_tensor, tensor_padding)

        u_min_bound = p1_pix[0] - v_window
        u_max_bound = p1_pix[0] + u_window + 1
        v_min_bound = p1_pix[1] - v_window
        v_max_bound = p1_pix[1] + v_window + 1

        # Padding
        u_min = u_min_bound + u_window
        u_max = u_max_bound + u_window
        v_min = v_min_bound + v_window
        v_max = v_max_bound + v_window

        theta_min_bound = itheta - theta_window
        theta_max_bound = itheta + theta_window + 1

        if theta_min_bound < 0:
            theta_min = self.n_rotations + theta_min_bound
            theta_max = theta_max_bound
        elif theta_max_bound >= self.n_rotations:
            theta_max = theta_max_bound - self.n_rotations
            theta_min = theta_min_bound
        else:
            theta_max = theta_max_bound
            theta_min = theta_min_bound

        if theta_max < theta_min:
            z_est_at_xytheta_1 = z_tensor_pad[0, u_min:u_max, v_min:v_max,
                                        theta_min:]
            z_est_at_xytheta_2 = z_tensor_pad[0, u_min:u_max, v_min:v_max,
                                        :theta_max]
            roll_est_at_xytheta_1 = roll_tensor_pad[0, u_min:u_max, v_min:v_max,
                                        theta_min:]
            roll_est_at_xytheta_2 = roll_tensor_pad[0, u_min:u_max, v_min:v_max,
                                        :theta_max]
            pitch_est_at_xytheta_1 = pitch_tensor_pad[0, u_min:u_max, v_min:v_max,
                                        theta_min:]
            pitch_est_at_xytheta_2 = pitch_tensor_pad[0, u_min:u_max, v_min:v_max,
                                        :theta_max]
            z_est_at_xytheta=torch.cat((z_est_at_xytheta_1, z_est_at_xytheta_2), dim=-1)
            roll_est_at_xytheta=torch.cat((roll_est_at_xytheta_1, roll_est_at_xytheta_2), dim=-1)
            pitch_est_at_xytheta=torch.cat((pitch_est_at_xytheta_1, pitch_est_at_xytheta_2), dim=-1)
        else:
            z_est_at_xytheta = z_tensor_pad[0, u_min:u_max, v_min:v_max,
                                        theta_min:theta_max]
            roll_est_at_xytheta = roll_tensor_pad[0, u_min:u_max, v_min:v_max,
                                                theta_min:theta_max]
            pitch_est_at_xytheta = pitch_tensor_pad[0, u_min:u_max, v_min:v_max,
                                                theta_min:theta_max]

        z_est_at_xytheta = z_est_at_xytheta.reshape((1, -1))
        roll_est_at_xytheta = roll_est_at_xytheta.reshape((1, -1))
        pitch_est_at_xytheta = pitch_est_at_xytheta.reshape((1, -1))

        z_predict = self.z_regressor(z_est_at_xytheta)
        roll_predict = self.roll_regressor(roll_est_at_xytheta)
        pitch_predict = self.pitch_regressor(pitch_est_at_xytheta)

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = (p1_xyz[0],p1_xyz[1],z_predict.cpu().detach().numpy().item())
        p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = utils.eulerXYZ_to_quatXYZW((roll_predict, pitch_predict, -p1_theta))

        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            'pick': [p0_pix[0], p0_pix[1], p0_theta],
            'place': [p1_pix[0], p1_pix[1], p1_theta],
        }


class TwoStreamClipFilmLingUNetLatTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'clip_film_lingunet_lat'
        self.attention = TwoStreamAttentionLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamClipLingUNetLatTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'clip_lingunet_lat'
        self.attention = TwoStreamAttentionLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        print(self.device_type)
        
        self.transport = TwoStreamTransportLangFusionLatOriginal(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        print(self.device_type)


class TwoStreamRN50BertLingUNetTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'rn50_bert_lingunet'
        self.attention = TwoStreamAttentionLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamUntrainedRN50BertLingUNetTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet'
        stream_two_fcn = 'untrained_rn50_bert_lingunet'
        self.attention = TwoStreamAttentionLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusion(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class TwoStreamRN50BertLingUNetLatTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_one_fcn = 'plain_resnet_lat'
        stream_two_fcn = 'rn50_bert_lingunet_lat'
        self.attention = TwoStreamAttentionLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusionLat(
            stream_fcn=(stream_one_fcn, stream_two_fcn),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )


class OriginalTransporterLangFusionAgent(TwoStreamClipLingUNetTransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'plain_resnet_lang'
        self.attention = OneStreamAttentionLangFusion(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportLangFusion(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )



class ClipLingUNetTransporterAgent(TwoStreamClipLingUNetTransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'clip_lingunet'
        self.attention = OneStreamAttentionLangFusion(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportLangFusion(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
