"""Image dataset."""

import os
import pickle
import warnings

import numpy as np
from torch.utils.data import Dataset

from lovmm import tasks
from lovmm.tasks import cameras
from lovmm.utils import utils
from transforms3d import quaternions

# See transporter.py, regression.py, dummy.py, task.py, etc.
PIXEL_SIZE = 0.003125
CAMERA_CONFIG = cameras.RealSenseD415.CONFIG
BOUNDS = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

# Names as strings, REVERSE-sorted so longer (more specific) names are first.
TASK_NAMES = (tasks.names).keys()
TASK_NAMES = sorted(TASK_NAMES)[::-1]


class RavensDataset(Dataset):
    """A simple image dataset class."""

    def __init__(self, path, cfg, n_demos=0, augment=False):
        """A simple RGB-D image dataset."""
        self._path = path

        self.cfg = cfg
        self.sample_set = []
        self.max_seed = -1
        self.n_episodes = 0
        self.images = self.cfg['dataset']['images']
        self.cache = self.cfg['dataset']['cache']
        self.n_demos = n_demos
        self.augment = augment

        self.aug_theta_sigma = self.cfg['dataset']['augment']['theta_sigma'] if 'augment' in self.cfg['dataset'] else 60  # legacy code issue: theta_sigma was newly added
        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        # Track existing dataset if it exists.
        color_path = os.path.join(self._path, 'action')
        if os.path.exists(color_path):
            for fname in sorted(os.listdir(color_path)):
                if '.pkl' in fname:
                    seed = int(fname[(fname.find('-') + 1):-4])
                    self.n_episodes += 1
                    self.max_seed = max(self.max_seed, seed)

        self._cache = {}

        if self.n_demos > 0:
            self.images = self.cfg['dataset']['images']
            self.cache = self.cfg['dataset']['cache']

            # Check if there sufficient demos in the dataset
            if self.n_demos > self.n_episodes:
                raise Exception(f"Requested training on {self.n_demos} demos, but only {self.n_episodes} demos exist in the dataset path: {self._path}.")

            episodes = np.random.choice(range(self.n_episodes), self.n_demos, False)
            self.set(episodes)


    def add(self, seed, episode):
        """Add an episode to the dataset.

        Args:
          seed: random seed used to initialize the episode.
          episode: list of (obs, act, reward, info) tuples.
        """
        color, depth, action, reward, info = [], [], [], [], []
        for obs, act, r, i in episode:
            color.append(obs['color'])
            depth.append(obs['depth'])
            action.append(act)
            reward.append(r)
            info.append(i)

        color = np.uint8(color)
        depth = np.float32(depth)

        def dump(data, field):
            field_path = os.path.join(self._path, field)
            if not os.path.exists(field_path):
                os.makedirs(field_path)
            fname = f'{self.n_episodes:06d}-{seed}.pkl'  # -{len(episode):06d}
            with open(os.path.join(field_path, fname), 'wb') as f:
                pickle.dump(data, f)

        dump(color, 'color')
        dump(depth, 'depth')
        dump(action, 'action')
        dump(reward, 'reward')
        dump(info, 'info')

        self.n_episodes += 1
        self.max_seed = max(self.max_seed, seed)

    def set(self, episodes):
        """Limit random samples to specific fixed set."""
        self.sample_set = episodes

    def load(self, episode_id, images=True, cache=False):
        def load_field(episode_id, field, fname):

            # Check if sample is in cache.
            if cache:
                if episode_id in self._cache:
                    if field in self._cache[episode_id]:
                        return self._cache[episode_id][field]
                else:
                    self._cache[episode_id] = {}

            # Load sample from files.
            path = os.path.join(self._path, field)

            file_path = os.path.join(path, fname)
            if not os.path.exists(file_path):
                return None

            data = pickle.load(open(os.path.join(path, fname), 'rb'))
            if cache:
                self._cache[episode_id][field] = data
            return data

        # Get filename and random seed used to initialize episode.
        seed = None
        path = os.path.join(self._path, 'action')
        for fname in sorted(os.listdir(path)):
            if f'{episode_id:06d}' in fname:
                seed = int(fname[(fname.find('-') + 1):-4])

                # Load data.
                color = load_field(episode_id, 'color', fname)
                depth = load_field(episode_id, 'depth', fname)
                color_place = load_field(episode_id, 'color_place', fname)
                depth_place = load_field(episode_id, 'depth_place', fname)
                action = load_field(episode_id, 'action', fname)
                reward = load_field(episode_id, 'reward', fname)
                info = load_field(episode_id, 'info', fname)

                # Reconstruct episode.
                episode = []
                for i in range(len(action)):
                    obs = {'color': color[i], 'depth': depth[i],
                           'color_place': color_place[i] if color_place is not None else None,
                           'depth_place': depth_place[i] if depth_place is not None else None
                           } if images else {}
                    episode.append((obs, action[i], reward[i], info[i]))
                return episode, seed

    def get_image(self, obs, cam_config=None):
        """Stack color and height images image."""

        # if self.use_goal_image:
        #   colormap_g, heightmap_g = utils.get_fused_heightmap(goal, configs)
        #   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
        #   input_image = np.concatenate((input_image, goal_image), axis=2)
        #   assert input_image.shape[2] == 12, input_image.shape

        if cam_config is None:
            cam_config = self.cam_config

        # Get color and height maps from RGB-D images.
        cmap, hmap = utils.get_fused_heightmap(
            obs, cam_config, self.bounds, self.pix_size)
        img = np.concatenate((cmap,
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None]), axis=2)
        assert img.shape == self.in_shape, img.shape
        return img

    def get_six_dof(self,
                    transform_params,
                    heightmap,
                    pose0,
                    pose1,
                    augment=True):
        """Adjust SE(3) poses via the in-plane SE(2) augmentation transform."""

        p1_position, p1_rotation = pose1[0], pose1[1]
        p0_position, p0_rotation = pose0[0], pose0[1]

        if augment:
            t_world_center, t_world_centernew = utils.get_se3_from_image_transform(
                *transform_params, heightmap, self.bounds, self.pix_size)
            t_worldnew_world = t_world_centernew @ np.linalg.inv(t_world_center)
        else:
            t_worldnew_world = np.eye(4)

        p1_quat_wxyz = (p1_rotation[3], p1_rotation[0], p1_rotation[1],
                        p1_rotation[2])
        t_world_p1_mat = quaternions.quat2mat(p1_quat_wxyz)
        t_world_p1 = np.eye(4)
        t_world_p1[0:3, 0:3] = t_world_p1_mat
        t_world_p1[0:3, 3] = np.array(p1_position)

        t_worldnew_p1 = t_worldnew_world @ t_world_p1

        p0_quat_wxyz = (p0_rotation[3], p0_rotation[0], p0_rotation[1],
                        p0_rotation[2])
        t_world_p0_mat = quaternions.quat2mat(p0_quat_wxyz)

        t_world_p0 = np.eye(4)
        t_world_p0[0:3, 0:3] = t_world_p0_mat
        t_world_p0[0:3, 3] = np.array(p0_position)

        t_worldnew_p0 = t_worldnew_world @ t_world_p0

        # PICK FRAME, using 0 rotation due to suction rotational symmetry
        t_worldnew_p0theta0 = t_worldnew_p0 * 1.0
        t_worldnew_p0theta0[0:3, 0:3] = np.eye(3)

        # PLACE FRAME, adjusted for this 0 rotation on pick
        t_p0_p0theta0 = np.linalg.inv(t_worldnew_p0) @ t_worldnew_p0theta0
        t_worldnew_p1theta0 = t_worldnew_p1 @ t_p0_p0theta0

        # convert the above rotation to euler
        quatwxyz_worldnew_p1theta0 = quaternions.mat2quat(t_worldnew_p1theta0[0:3, 0:3])
        q = quatwxyz_worldnew_p1theta0
        quatxyzw_worldnew_p1theta0 = (q[1], q[2], q[3], q[0])
        p1_rotation = quatxyzw_worldnew_p1theta0
        p1_euler = utils.quatXYZW_to_eulerXYZ(p1_rotation)
        roll = p1_euler[0]
        pitch = p1_euler[1]
        p1_theta = -p1_euler[2]

        p0_theta = 0
        z = p1_position[2]
        return p0_theta, p1_theta, z, roll, pitch

    def process_sample(self, datum, augment=True):
        # Get training labels from data sample.
        (obs, act, _, info) = datum
        # TODO: Revise here to directly get ortho image from habitat instead of reconstructing bullet images
        # img = self.get_image(obs)
        # resize to (320,160)
        cmap = obs['color'][0]
        hmap = obs['depth'][0]

        img = np.concatenate((cmap,
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None]), axis=2)
        if obs.get('color_place') is not None:
            cmap_place = obs['color_place'][0]
            hmap_place = obs['depth_place'][0]

            img_place = np.concatenate((cmap_place,
                                  hmap_place[Ellipsis, None],
                                  hmap_place[Ellipsis, None],
                                  hmap_place[Ellipsis, None]), axis=2)
        else:
            img_place = None

        p0, p1 = None, None
        p0_theta, p1_theta = None, None
        perturb_params =  None

        if act:
            p0_xyz, p0_xyzw = act['pose0']
            p1_xyz, p1_xyzw = act['pose1']
            p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
            p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
            p1_theta = p1_theta - p0_theta
            p0_theta = 0
            # p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
            # p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
            # HABITAT -----------------------------------------------
            if act.get('habitat_pix_pose0') is not None:
                p0 = (act['habitat_pix_pose0'][0], act['habitat_pix_pose0'][1])
                p1 = (act['habitat_pix_pose1'][0], act['habitat_pix_pose1'][1])
            #---------------------------------------------------------
            p1_euler = utils.quatXYZW_to_eulerXYZ(p1_xyzw)
            z = p1_xyz[2]
            roll = p1_euler[0]
            pitch = p1_euler[1]

        # Data augmentation.
        if augment:
            (img, img_place), _, (p0, p1), perturb_params = utils.perturb((img, img_place), [p0, p1], theta_sigma=self.aug_theta_sigma, add_noise=True)
            _, p1_theta, z, roll, pitch = self.get_six_dof(
                perturb_params, img[:, :, 3], (p0_xyz, p0_xyzw), (p1_xyz, p1_xyzw))

        sample = {
            'img': img,
            'img_place': img_place,
            'p0': p0, 'p0_theta': p0_theta,
            'p1': p1, 'p1_theta': -p1_theta, # -p1_theta as habitat use z axis in object coordinate frame
            'p1_z': z, 'p1_roll': roll, 'p1_pitch': pitch,
            'perturb_params': perturb_params
        }

        # Add language goal if available.
        if 'lang_goal' not in info:
            warnings.warn("No language goal. Defaulting to 'task completed.'")

        if info and 'lang_goal' in info:
            sample['lang_goal'] = info['lang_goal']
        else:
            sample['lang_goal'] = "task completed."

        return sample

    def process_goal(self, goal, perturb_params):
        # Get goal sample.
        (obs, act, _, info) = goal
        # img = self.get_image(obs)

        cmap = obs['color'][0]
        hmap = obs['depth'][0]

        img = np.concatenate((cmap,
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None]), axis=2)

        p0, p1 = None, None
        p0_theta, p1_theta = None, None

        # Data augmentation with specific params.
        if perturb_params:
            img = utils.apply_perturbation(img, perturb_params)

        sample = {
            'img': img,
            'p0': p0, 'p0_theta': p0_theta,
            'p1': p1, 'p1_theta': p1_theta,

            'perturb_params': perturb_params
        }

        # Add language goal if available.
        if 'lang_goal' not in info:
            warnings.warn("No language goal. Defaulting to 'task completed.'")

        if info and 'lang_goal' in info:
            sample['lang_goal'] = info['lang_goal']
        else:
            sample['lang_goal'] = "task completed."

        return sample

    def __len__(self):
        return len(self.sample_set)

    def __getitem__(self, idx):
        # Choose random episode.
        if len(self.sample_set) > 0:
            episode_id = np.random.choice(self.sample_set)
        else:
            episode_id = np.random.choice(range(self.n_episodes))
        episode, _ = self.load(episode_id, self.images, self.cache)

        # Is the task sequential like stack-block-pyramid-seq?
        is_sequential_task = '-seq' in self._path.split("/")[-1]

        # Return random observation action pair (and goal) from episode.
        i = np.random.choice(range(len(episode)-1))
        g = i+1 if is_sequential_task else -1
        sample, goal = episode[i], episode[g]

        # Process sample.
        sample = self.process_sample(sample, augment=self.augment)
        goal = self.process_goal(goal, perturb_params=sample['perturb_params'])

        return sample, goal


class RavensMultiTaskDataset(RavensDataset):

    MULTI_TASKS = {
        'multi-custom-seen-lovmm':{
            'train':[
            'laundry-bathroom-pack-shoes',
            'livingroom-stack-cups',
            'guestroom-organize-bottles',
            'officeroom-tidy-food',
            'bedroom-sort-toys',
            'kitchen-livingroom-put-towels',],

            'val': [
                'laundry-bathroom-pack-shoes',
                'livingroom-stack-cups',
                'guestroom-organize-bottles',
                'officeroom-tidy-food',
                'bedroom-sort-toys',
                'kitchen-livingroom-put-towels', ],

            'test': [
                'laundry-bathroom-pack-shoes',
                'livingroom-stack-cups',
                'guestroom-organize-bottles',
                'officeroom-tidy-food',
                'bedroom-sort-toys',
                'kitchen-livingroom-put-towels', ],

        },

        'multi-custom-seen': {
            'train': [
                'packing-boxes-pairs-seen-colors',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'assembling-kits-seq-seen-colors',
                'put-block-in-bowl-seen-colors',
                'packing-boxes-pairs-unseen-colors',
                'packing-unseen-google-objects-seq',
                'packing-unseen-google-objects-group',
                'assembling-kits-seq-unseen-colors',
                'put-block-in-bowl-unseen-colors',
                'packing-shapes',
                'put-block-in-bowl-seen-colors-6dof',
                'put-block-in-bowl-unseen-colors-6dof',
                'packing-shapes-6dof',
            ],
            'val': [
                'packing-boxes-pairs-seen-colors',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'assembling-kits-seq-seen-colors',
                'put-block-in-bowl-seen-colors',
                'packing-boxes-pairs-unseen-colors',
                'packing-unseen-google-objects-seq',
                'packing-unseen-google-objects-group',
                'assembling-kits-seq-unseen-colors',
                'put-block-in-bowl-unseen-colors',
                'packing-shapes',
                'put-block-in-bowl-seen-colors-6dof',
                'put-block-in-bowl-unseen-colors-6dof',
                'packing-shapes-6dof',
            ],
            'test': [
                'packing-boxes-pairs-seen-colors',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'assembling-kits-seq-seen-colors',
                'put-block-in-bowl-seen-colors',
                'packing-boxes-pairs-unseen-colors',
                'packing-unseen-google-objects-seq',
                'packing-unseen-google-objects-group',
                'assembling-kits-seq-unseen-colors',
                'put-block-in-bowl-unseen-colors',
                'packing-shapes',
                'put-block-in-bowl-seen-colors-6dof',
                'put-block-in-bowl-unseen-colors-6dof',
                'packing-shapes-6dof',
            ],
        },
        'multi-custom-unseen': {
            'train': [
                'packing-boxes-pairs-unseen-colors',
                'packing-unseen-google-objects-seq',
                'packing-unseen-google-objects-group',
                'assembling-kits-seq-unseen-colors',
                'put-block-in-bowl-unseen-colors',
                'packing-shapes',
                'put-block-in-bowl-unseen-colors-6dof',
                'packing-shapes-6dof',
            ],
            'val': [
                'packing-boxes-pairs-unseen-colors',
                'packing-unseen-google-objects-seq',
                'packing-unseen-google-objects-group',
                'assembling-kits-seq-unseen-colors',
                'put-block-in-bowl-unseen-colors',
                'packing-shapes',
                'put-block-in-bowl-unseen-colors-6dof',
                'packing-shapes-6dof',
            ],
            'test': [
                'packing-boxes-pairs-unseen-colors',
                'packing-unseen-google-objects-seq',
                'packing-unseen-google-objects-group',
                'assembling-kits-seq-unseen-colors',
                'put-block-in-bowl-unseen-colors',
                'packing-shapes',
                'put-block-in-bowl-unseen-colors-6dof',
                'packing-shapes-6dof',
            ],
        },
        # all tasks
        'multi-all': {
            'train': [
                'align-box-corner',
                'assembling-kits',
                'block-insertion',
                'manipulating-rope',
                'packing-boxes',
                'palletizing-boxes',
                'place-red-in-green',
                'stack-block-pyramid',
                'sweeping-piles',
                'towers-of-hanoi',
                'align-rope',
                'assembling-kits-seq-unseen-colors',
                'packing-boxes-pairs-unseen-colors',
                'packing-shapes',
                'packing-unseen-google-objects-seq',
                'packing-unseen-google-objects-group',
                'put-block-in-bowl-unseen-colors',
                'stack-block-pyramid-seq-unseen-colors',
                'separating-piles-unseen-colors',
                'towers-of-hanoi-seq-unseen-colors',
            ],
            'val': [
                'align-box-corner',
                'assembling-kits',
                'block-insertion',
                'manipulating-rope',
                'packing-boxes',
                'palletizing-boxes',
                'place-red-in-green',
                'stack-block-pyramid',
                'sweeping-piles',
                'towers-of-hanoi',
                'align-rope',
                'assembling-kits-seq-seen-colors',
                'assembling-kits-seq-unseen-colors',
                'packing-boxes-pairs-seen-colors',
                'packing-boxes-pairs-unseen-colors',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-unseen-google-objects-seq',
                'packing-seen-google-objects-group',
                'packing-unseen-google-objects-group',
                'put-block-in-bowl-seen-colors',
                'put-block-in-bowl-unseen-colors',
                'stack-block-pyramid-seq-seen-colors',
                'stack-block-pyramid-seq-unseen-colors',
                'separating-piles-seen-colors',
                'separating-piles-unseen-colors',
                'towers-of-hanoi-seq-seen-colors',
                'towers-of-hanoi-seq-unseen-colors',
            ],
            'test': [
                'align-box-corner',
                'assembling-kits',
                'block-insertion',
                'manipulating-rope',
                'packing-boxes',
                'palletizing-boxes',
                'place-red-in-green',
                'stack-block-pyramid',
                'sweeping-piles',
                'towers-of-hanoi',
                'align-rope',
                'assembling-kits-seq-seen-colors',
                'assembling-kits-seq-unseen-colors',
                'packing-boxes-pairs-seen-colors',
                'packing-boxes-pairs-unseen-colors',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-unseen-google-objects-seq',
                'packing-seen-google-objects-group',
                'packing-unseen-google-objects-group',
                'put-block-in-bowl-seen-colors',
                'put-block-in-bowl-unseen-colors',
                'stack-block-pyramid-seq-seen-colors',
                'stack-block-pyramid-seq-unseen-colors',
                'separating-piles-seen-colors',
                'separating-piles-unseen-colors',
                'towers-of-hanoi-seq-seen-colors',
                'towers-of-hanoi-seq-unseen-colors',
            ],
        },

        # demo-conditioned tasks
        'multi-demo-conditioned': {
            'train': [
                'align-box-corner',
                'assembling-kits',
                'block-insertion',
                'manipulating-rope',
                'packing-boxes',
                'palletizing-boxes',
                'place-red-in-green',
                'stack-block-pyramid',
                'sweeping-piles',
                'towers-of-hanoi',
            ],
            'val': [
                'align-box-corner',
                'assembling-kits',
                'block-insertion',
                'manipulating-rope',
                'packing-boxes',
                'palletizing-boxes',
                'place-red-in-green',
                'stack-block-pyramid',
                'sweeping-piles',
                'towers-of-hanoi',
            ],
            'test': [
                'align-box-corner',
                'assembling-kits',
                'block-insertion',
                'manipulating-rope',
                'packing-boxes',
                'palletizing-boxes',
                'place-red-in-green',
                'stack-block-pyramid',
                'sweeping-piles',
                'towers-of-hanoi',
            ],
        },

        # goal-conditioned tasks
        'multi-language-conditioned': {
            'train': [
                'align-rope',
                'assembling-kits-seq-unseen-colors', # unseen here refers to training only seen splits to be consitent with single-task setting
                'packing-boxes-pairs-unseen-colors',
                'packing-shapes',
                'packing-unseen-google-objects-seq',
                'packing-unseen-google-objects-group',
                'put-block-in-bowl-unseen-colors',
                'stack-block-pyramid-seq-unseen-colors',
                'separating-piles-unseen-colors',
                'towers-of-hanoi-seq-unseen-colors',
            ],
            'val': [
                'align-rope',
                'assembling-kits-seq-seen-colors',
                'assembling-kits-seq-unseen-colors',
                'packing-boxes-pairs-seen-colors',
                'packing-boxes-pairs-unseen-colors',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-unseen-google-objects-seq',
                'packing-seen-google-objects-group',
                'packing-unseen-google-objects-group',
                'put-block-in-bowl-seen-colors',
                'put-block-in-bowl-unseen-colors',
                'stack-block-pyramid-seq-seen-colors',
                'stack-block-pyramid-seq-unseen-colors',
                'separating-piles-seen-colors',
                'separating-piles-unseen-colors',
                'towers-of-hanoi-seq-seen-colors',
                'towers-of-hanoi-seq-unseen-colors',
            ],
            'test': [
                'align-rope',
                'assembling-kits-seq-seen-colors',
                'assembling-kits-seq-unseen-colors',
                'packing-boxes-pairs-seen-colors',
                'packing-boxes-pairs-unseen-colors',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-unseen-google-objects-seq',
                'packing-seen-google-objects-group',
                'packing-unseen-google-objects-group',
                'put-block-in-bowl-seen-colors',
                'put-block-in-bowl-unseen-colors',
                'stack-block-pyramid-seq-seen-colors',
                'stack-block-pyramid-seq-unseen-colors',
                'separating-piles-seen-colors',
                'separating-piles-unseen-colors',
                'towers-of-hanoi-seq-seen-colors',
                'towers-of-hanoi-seq-unseen-colors',
            ],
        },


        ##### multi-attr tasks
        'multi-attr-align-rope': {
            'train': [
                'assembling-kits-seq-full',
                'packing-boxes-pairs-full',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-full',
                'separating-piles-full',
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'align-rope',
            ],
            'test': [
                'align-rope',
            ],
            'attr_train_task': None,
        },

        'multi-attr-packing-shapes': {
            'train': [
                'align-rope',
                'assembling-kits-seq-full',
                'packing-boxes-pairs-full',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-full',
                'separating-piles-full',
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'packing-shapes',
            ],
            'test': [
                'packing-shapes',
            ],
            'attr_train_task': None,
        },

        'multi-attr-assembling-kits-seq-unseen-colors': {
            'train': [
                'align-rope',
                'assembling-kits-seq-seen-colors', # seen only
                'packing-boxes-pairs-full',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-full',
                'separating-piles-full',
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'assembling-kits-seq-unseen-colors',
            ],
            'test': [
                'assembling-kits-seq-unseen-colors',
            ],
            'attr_train_task': 'assembling-kits-seq-seen-colors',
        },

        'multi-attr-packing-boxes-pairs-unseen-colors': {
            'train': [
                'align-rope',
                'assembling-kits-seq-full',
                'packing-boxes-pairs-seen-colors', # seen only
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-full',
                'separating-piles-full',
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'packing-boxes-pairs-unseen-colors',
            ],
            'test': [
                'packing-boxes-pairs-unseen-colors',
            ],
            'attr_train_task': 'packing-boxes-pairs-seen-colors',
        },

        'multi-attr-packing-unseen-google-objects-seq': {
            'train': [
                'align-rope',
                'assembling-kits-seq-full',
                'packing-boxes-pairs-full',
                'packing-shapes',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-full',
                'separating-piles-full',
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'packing-unseen-google-objects-seq',
            ],
            'test': [
                'packing-unseen-google-objects-seq',
            ],
            'attr_train_task': 'packing-seen-google-objects-group',
        },

        'multi-attr-packing-unseen-google-objects-group': {
            'train': [
                'align-rope',
                'assembling-kits-seq-full',
                'packing-boxes-pairs-full',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-full',
                'separating-piles-full',
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'packing-unseen-google-objects-group',
            ],
            'test': [
                'packing-unseen-google-objects-group',
            ],
            'attr_train_task': 'packing-seen-google-objects-seq',
        },

        'multi-attr-put-block-in-bowl-unseen-colors': {
            'train': [
                'align-rope',
                'assembling-kits-seq-full',
                'packing-boxes-pairs-full',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-seen-colors', # seen only
                'stack-block-pyramid-seq-full',
                'separating-piles-full',
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'put-block-in-bowl-unseen-colors',
            ],
            'test': [
                'put-block-in-bowl-unseen-colors',
            ],
            'attr_train_task': 'put-block-in-bowl-seen-colors',
        },

        'multi-attr-stack-block-pyramid-seq-unseen-colors': {
            'train': [
                'align-rope',
                'assembling-kits-seq-full',
                'packing-boxes-pairs-full',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-seen-colors', # seen only
                'separating-piles-full',
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'stack-block-pyramid-seq-unseen-colors',
            ],
            'test': [
                'stack-block-pyramid-seq-unseen-colors',
            ],
            'attr_train_task': 'stack-block-pyramid-seq-seen-colors',
        },

        'multi-attr-separating-piles-unseen-colors': {
            'train': [
                'align-rope',
                'assembling-kits-seq-full',
                'packing-boxes-pairs-full',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-full',
                'separating-piles-seen-colors', # seen only
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'separating-piles-unseen-colors',
            ],
            'test': [
                'separating-piles-unseen-colors',
            ],
            'attr_train_task': 'separating-piles-seen-colors',
        },

        'multi-attr-towers-of-hanoi-seq-unseen-colors': {
            'train': [
                'align-rope',
                'assembling-kits-seq-full',
                'packing-boxes-pairs-full',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-full',
                'separating-piles-full',
                'towers-of-hanoi-seq-seen-colors', # seen only
            ],
            'val': [
                'towers-of-hanoi-seq-unseen-colors',
            ],
            'test': [
                'towers-of-hanoi-seq-unseen-colors',
            ],
            'attr_train_task': 'towers-of-hanoi-seq-seen-colors',
        },

    }
    def __init__(self, path, cfg, group='multi-all',
                 mode='train', n_demos=100, augment=False):
        """A multi-task dataset."""
        self.root_path = path
        self.mode = mode
        self.tasks = self.MULTI_TASKS[group][mode]
        self.attr_train_task = self.MULTI_TASKS[group]['attr_train_task'] if 'attr_train_task' in self.MULTI_TASKS[group] else None

        self.cfg = cfg
        self.sample_set = {}
        self.max_seed = -1
        self.n_episodes = 0
        self.images = self.cfg['dataset']['images']
        self.cache = self.cfg['dataset']['cache']
        self.n_demos = n_demos
        self.augment = augment

        self.aug_theta_sigma = self.cfg['dataset']['augment']['theta_sigma'] if 'augment' in self.cfg['dataset'] else 60  # legacy code issue: theta_sigma was newly added
        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        self.n_episodes = {}
        episodes = {}

        for task in self.tasks:
            task_path = os.path.join(self.root_path, f'{task}-{mode}')
            action_path = os.path.join(task_path, 'action')
            n_episodes = 0
            if os.path.exists(action_path):
                for fname in sorted(os.listdir(action_path)):
                    if '.pkl' in fname:
                        n_episodes += 1
            self.n_episodes[task] = n_episodes

            if n_episodes == 0:
                raise Exception(f"{task}-{mode} has 0 episodes. Remove it from the list in dataset.py")

            # Select random episode depending on the size of the dataset.
            episodes[task] = np.random.choice(range(n_episodes), min(self.n_demos, n_episodes), False)

        if self.n_demos > 0:
            self.images = self.cfg['dataset']['images']
            self.cache = False # TODO(mohit): fix caching for multi-task dataset
            self.set(episodes)

        self._path = None
        self._task = None

    def __len__(self):
        # Average number of episodes across all tasks
        total_episodes = 0
        for _, episode_ids in self.sample_set.items():
            total_episodes += len(episode_ids)
        avg_episodes = total_episodes // len(self.sample_set)
        return avg_episodes

    def __getitem__(self, idx):
        # Choose random task.
        self._task = np.random.choice(self.tasks)
        self._path = os.path.join(self.root_path, f'{self._task}')

        # Choose random episode.
        if len(self.sample_set[self._task]) > 0:
            episode_id = np.random.choice(self.sample_set[self._task])
        else:
            episode_id = np.random.choice(range(self.n_episodes[self._task]))
        episode, _ = self.load(episode_id, self.images, self.cache)

        # Is the task sequential like stack-block-pyramid-seq?
        is_sequential_task = '-seq' in self._path.split("/")[-1]

        # Return observation action pair (and goal) from episode.
        if len(episode) > 1:
            i = np.random.choice(range(len(episode)-1))
            g = i+1 if is_sequential_task else -1
            sample, goal = episode[i], episode[g]
        else:
            sample, goal = episode[0], episode[0]

        # Process sample
        sample = self.process_sample(sample, augment=self.augment)
        goal = self.process_goal(goal, perturb_params=sample['perturb_params'])

        return sample, goal

    def add(self, seed, episode):
        raise Exception("Adding tasks not supported with multi-task dataset")

    def load(self, episode_id, images=True, cache=False):
        if self.attr_train_task is None or self.mode in ['val', 'test']:
            self._task = np.random.choice(self.tasks)
        else:
            all_other_tasks = list(self.tasks)
            all_other_tasks.remove(self.attr_train_task)
            all_tasks = [self.attr_train_task] + all_other_tasks # add seen task in the front

            # 50% chance of sampling the main seen task and 50% chance of sampling any other seen-unseen task
            mult_attr_seen_sample_prob = 0.5
            sampling_probs = [(1-mult_attr_seen_sample_prob) / (len(all_tasks)-1)] * len(all_tasks)
            sampling_probs[0] = mult_attr_seen_sample_prob

            self._task = np.random.choice(all_tasks, p=sampling_probs)

        self._path = os.path.join(self.root_path, f'{self._task}-{self.mode}')
        return super().load(episode_id, images, cache)

    def get_curr_task(self):
        return self._task
