from lovmm.models.resnet import ResNet43_8s,ResNet43_8s_3
from lovmm.models.clip_wo_skip import CLIPWithoutSkipConnections

from lovmm.models.rn50_bert_unet import RN50BertUNet
from lovmm.models.rn50_bert_lingunet import RN50BertLingUNet
from lovmm.models.rn50_bert_lingunet_lat import RN50BertLingUNetLat
from lovmm.models.untrained_rn50_bert_lingunet import UntrainedRN50BertLingUNet

from lovmm.models.clip_unet import CLIPUNet
from lovmm.models.clip_lingunet import CLIPLingUNet

from lovmm.models.resnet_lang import ResNet43_8s_lang

from lovmm.models.resnet_lat import ResNet45_10s
from lovmm.models.resnet_lat_add import ResNet45_10s_add
from lovmm.models.clip_unet_lat import CLIPUNetLat
from lovmm.models.clip_lingunet_lat import CLIPLingUNetLat
from lovmm.models.clip_film_lingunet_lat import CLIPFilmLingUNet

from lovmm.models.regressor import Regressor

names = {
    # resnet
    'plain_resnet': ResNet43_8s_3,
    'plain_resnet_lang': ResNet43_8s_lang,

    # without skip-connections
    'clip_woskip': CLIPWithoutSkipConnections,

    # unet
    'clip_unet': CLIPUNet,
    'rn50_bert_unet': RN50BertUNet,

    # lingunet
    'clip_lingunet': CLIPLingUNet,
    'rn50_bert_lingunet': RN50BertLingUNet,
    'untrained_rn50_bert_lingunet': UntrainedRN50BertLingUNet,

    # lateral connections
    'plain_resnet_lat': ResNet45_10s,
    'plain_resnet_lat_add': ResNet45_10s_add,
    'clip_unet_lat': CLIPUNetLat,
    'clip_lingunet_lat': CLIPLingUNetLat,
    'clip_film_lingunet_lat': CLIPFilmLingUNet,
    'rn50_bert_lingunet_lat': RN50BertLingUNetLat,

    'regressor': Regressor,
}
