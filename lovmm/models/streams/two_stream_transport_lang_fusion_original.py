import torch
import numpy as np

import lovmm.models as models
import lovmm.models.core.fusion as fusion
from lovmm.models.core.transport import Transport


class TwoStreamTransportLangFusionOriginal(Transport):
    """Two Stream Transport (a.k.a Place) module"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.key_stream_two = stream_two_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_two = stream_two_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.fusion_key = fusion.names[self.fusion_type](input_dim=self.kernel_dim)
        self.fusion_query = fusion.names[self.fusion_type](input_dim=self.kernel_dim)

        print(f"Transport FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def transport(self, in_tensor, crop, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)

        return logits, kernel

    def forward(self, inp_img, p, lang_goal, softmax=True, inp_img_place = None):
        """Forward pass."""
        img_unprocessed = np.pad(inp_img, self.padding, mode='constant')
        input_data = img_unprocessed
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape)
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)
        if inp_img_place is not None:
            img_unprocessed_place = np.pad(inp_img_place, self.padding, mode='constant')
            input_data_place = img_unprocessed_place
            in_shape_place = (1,) + input_data_place.shape
            input_data_place = input_data_place.reshape(in_shape_place)
            in_tensor_place = torch.from_numpy(input_data_place).to(dtype=torch.float, device=self.device)
            in_tensor_place = in_tensor_place.permute(0, 3, 1, 2)
        else:
            in_tensor_place = in_tensor
        # Rotation pivot.
        pv = np.array([p[0], p[1]]) + self.pad_size

        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2)
        # TODO: REVISE HERE TO MAKE crop of PICK WORKSPACE IMAGE
        crop = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        crop = self.rotator(crop, pivot=pv)
        crop = torch.cat(crop, dim=0)
        crop = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]
        
        logits, kernel = self.transport(in_tensor_place, crop, lang_goal)

        # TODO(Mohit): Crop after network. Broken for now.
        # Crop after network (for receptive field, and more elegant).
        #in_tensor = in_tensor.permute(0, 3, 1, 2)

        #logits, crop = self.transport(in_tensor_place, in_tensor, lang_goal)
        #crop = crop.repeat(self.n_rotations, 1, 1, 1)
        #crop = self.rotator(crop, pivot=pv)
        #crop = torch.cat(crop, dim=0)
        #hcrop = self.pad_size
        #kernel = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        return self.correlate(logits, kernel, softmax)


    def forward6d(self, inp_img, p, lang_goal, softmax=True, inp_img_place = None):
        """Forward pass."""
        img_unprocessed = np.pad(inp_img, self.padding, mode='constant')
        input_data = img_unprocessed
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape)
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)
        if inp_img_place is not None:
            img_unprocessed_place = np.pad(inp_img_place, self.padding, mode='constant')
            input_data_place = img_unprocessed_place
            in_shape_place = (1,) + input_data_place.shape
            input_data_place = input_data_place.reshape(in_shape_place)
            in_tensor_place = torch.from_numpy(input_data_place).to(dtype=torch.float, device=self.device)
            in_tensor_place = in_tensor_place.permute(0, 3, 1, 2)
        else:
            in_tensor_place = in_tensor
        # Rotation pivot.
        pv = np.array([p[0], p[1]]) + self.pad_size

        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2)
        # # TODO: REVISE HERE TO MAKE crop of PICK WORKSPACE IMAGE
        crop = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        crop = self.rotator(crop, pivot=pv)
        crop = torch.cat(crop, dim=0)
        crop = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]
        #
        logits, kernel = self.transport(in_tensor_place, crop, lang_goal)

        # TODO(Mohit): Crop after network. Broken for now.
        # Crop after network (for receptive field, and more elegant).
        #in_tensor = in_tensor.permute(0, 3, 1, 2)

        #logits, crop = self.transport(in_tensor_place, in_tensor, lang_goal)
        #crop = crop.repeat(self.n_rotations, 1, 1, 1)
        #crop = self.rotator(crop, pivot=pv)
        #crop = torch.cat(crop, dim=0)
        #hcrop = self.pad_size
        #kernel = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        return self.correlate6d(logits, kernel, softmax)


class TwoStreamTransportLangFusionLatOriginal(TwoStreamTransportLangFusionOriginal):
    """Two Stream Transport (a.k.a Place) module with lateral connections"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.output_dim = 3
        self.kernel_dim = 3
        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def transport(self, in_tensor, crop, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)

        return logits, kernel

class TwoStreamTransport6dLangFusionLat(TwoStreamTransportLangFusionOriginal):
    """Two Stream Transport (a.k.a Place) module with lateral connections"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):

        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        self.output_dim = 36
        self.kernel_dim = 36
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def transport(self, in_tensor, crop, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)

        return logits, kernel

    def forward(self, inp_img, p, lang_goal, softmax=True, inp_img_place = None):
        """Forward pass."""
        img_unprocessed = np.pad(inp_img, self.padding, mode='constant')
        input_data = img_unprocessed
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape)
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)
        if inp_img_place is not None:
            img_unprocessed_place = np.pad(inp_img_place, self.padding, mode='constant')
            input_data_place = img_unprocessed_place
            in_shape_place = (1,) + input_data_place.shape
            input_data_place = input_data_place.reshape(in_shape_place)
            in_tensor_place = torch.from_numpy(input_data_place).to(dtype=torch.float, device=self.device)
            in_tensor_place = in_tensor_place.permute(0, 3, 1, 2)
        else:
            in_tensor_place = in_tensor
        # Rotation pivot.
        pv = np.array([p[0], p[1]]) + self.pad_size

        # Crop before network (default for Transporters CoRL 2020).
        # hcrop = self.pad_size
        # in_tensor = in_tensor.permute(0, 3, 1, 2)
        # # TODO: REVISE HERE TO MAKE crop of PICK WORKSPACE IMAGE
        # crop = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        # crop = self.rotator(crop, pivot=pv)
        # crop = torch.cat(crop, dim=0)
        # crop = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]
        #
        # logits, kernel = self.transport(in_tensor_place, crop, lang_goal)

        # TODO(Mohit): Crop after network. Broken for now.
        # Crop after network (for receptive field, and more elegant).
        in_tensor = in_tensor.permute(0, 3, 1, 2)

        logits, crop = self.transport(in_tensor_place, in_tensor, lang_goal)
        crop = crop.repeat(self.n_rotations, 1, 1, 1)
        crop = self.rotator(crop, pivot=pv)
        crop = torch.cat(crop, dim=0)
        hcrop = self.pad_size
        kernel = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        return self.correlate(logits, kernel, softmax)
