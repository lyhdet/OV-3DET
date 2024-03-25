# Copyright (c) Facebook, Inc. and its affiliates.
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from utils.pc_util import scale_points, shift_scale_points
from utils.sunrgbd_pc_util import write_oriented_bbox
import utils.pc_util as pc_util
from utils.sunrgbd_utils import SUNRGBD_Calibration
import cv2
from PIL import Image
import torchvision.transforms as T
from utils.box_util import extract_pc_in_box3d

try:
	from torchvision.transforms import InterpolationMode
	BICUBIC = InterpolationMode.BICUBIC
except ImportError:
	BICUBIC = Image.BICUBIC

from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine
from models.transformer import (MaskedTransformerEncoder, TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer)

from .DETR.backbone import build_backbone
from .DETR.transformer import build_transformer
from .DETR.detr import DETR, SetCriterion
from .DETR.matcher import build_matcher

# clip model
import sys
import clip
import traceback

from utils.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls

class BoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        if angle_logits.shape[-1] == 1:
            # special case for datasets with no rotation angle
            # we still use the predictions so that model outputs are used
            # in the backwards pass (DDP may complain otherwise)
            angle = angle_logits * 0 + angle_residual * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        # assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        return self.dataset_config.box_parametrization_to_corners(
            box_center_unnorm, box_size_unnorm, box_angle
        )


class Model3DETR(nn.Module):
    """
    Main 3DETR model. Consists of the following learnable sub-models
    - pre_encoder: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    - query computation: samples a set of B coordinates from the N'' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD box features
                Takes N''xD features from the encoder and BxD query features.
    - mlp_heads: Predicts bounding box parameters and classes from the BxD box features
    """

    def __init__(
        self,
        args,
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        mlp_dropout=0.3,
        num_queries=256,
    ):
        super().__init__()
        if args.clip_model in ["ViT-L/14@336px"]:
            clip_head_dim = 768
        elif args.clip_model in ["ViT-B/32"]:
            clip_head_dim = 512
        else:
            print("Error: No such clip model")
            traceback.print_stack()
            exit()

        self.pre_encoder = pre_encoder
        self.encoder = encoder
        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        self.decoder = decoder
        self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

        self.num_queries = num_queries
        self.box_processor = BoxProcessor(dataset_config)

        self.clip_header = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[512,512],
            output_dim=clip_head_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=False,
            output_use_norm=False,
            output_use_bias=True,
        )
        
        # header for transform features
        # self.clip_header = nn.Parameter(torch.empty(decoder_dim, 512))
        # nn.init.normal_(self.clip_header, std=decoder_dim ** -0.5)
        
        

    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # add 1 for background/not-an-object class
        semcls_head = mlp_func(output_dim=dataset_config.num_semcls + 1)

        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=dataset_config.num_angle_bin)
        angle_reg_head = mlp_func(output_dim=dataset_config.num_angle_bin)

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = furthest_point_sample(encoder_xyz, self.num_queries)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds):
        xyz, features = self._break_up_pc(point_clouds)
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )
        if enc_inds is None:
            # encoder does not perform any downsampling
            enc_inds = pre_enc_inds
        else:
            # use gather here to ensure that it works for both FPS and random sampling
            enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.long())
        return enc_xyz, enc_features, enc_inds

    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)
        center_offset = (
            self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
        )
        size_normalized = (
            self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
        )
        angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](
            box_features
        ).transpose(1, 2)

        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
        size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(
            num_layers, batch, num_queries, -1
        )
        angle_residual = angle_residual_normalized * (
            np.pi / angle_residual_normalized.shape[-1]
        )

        outputs = []
        for l in range(num_layers):
            # box processor converts outputs so we can get a 3D bounding box
            (
                center_normalized,
                center_unnormalized,
            ) = self.box_processor.compute_predicted_center(
                center_offset[l], query_xyz, point_cloud_dims
            )
            angle_continuous = self.box_processor.compute_predicted_angle(
                angle_logits[l], angle_residual[l]
            )
            size_unnormalized = self.box_processor.compute_predicted_size(
                size_normalized[l], point_cloud_dims
            )
            box_corners = self.box_processor.box_parametrization_to_corners(
                center_unnormalized, size_unnormalized, angle_continuous
            )

            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                (
                    semcls_prob,
                    objectness_prob,
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])

            box_prediction = {
                "sem_cls_logits": cls_logits[l],
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "size_normalized": size_normalized[l],
                "size_unnormalized": size_unnormalized,
                "angle_logits": angle_logits[l],
                "angle_residual": angle_residual[l],
                "angle_residual_normalized": angle_residual_normalized[l],
                "angle_continuous": angle_continuous,
                "objectness_prob": objectness_prob,
                "sem_cls_prob": semcls_prob,
                "box_corners": box_corners,
            }
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }

    def forward(self, inputs, encoder_only=False):
        point_clouds = inputs["point_clouds"]

        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3

        if encoder_only:
            # return: batch x npoints x channels
            return enc_xyz, enc_features.transpose(0, 1)

        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)
        # query_embed: batch x channel x npoint
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # decoder expects: npoints x batch x channel
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        box_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos
        )[0]
        
        # token for clip
        pc_query_feat = box_features[-1,:,:,:]

        box_predictions = self.get_box_predictions(
            query_xyz, point_cloud_dims, box_features
        )
        
        # token for clip
        pc_query_feat = pc_query_feat.permute(1, 2, 0)
        pc_query_feat = self.clip_header(pc_query_feat)
        pc_query_feat = pc_query_feat.permute(2, 0, 1)
        
        # pc_query_feat = pc_query_feat @ self.clip_header
        pc_query_feat = pc_query_feat / pc_query_feat.norm(dim=2, keepdim=True)

        box_predictions["pc_query_feat"] = pc_query_feat
        return box_predictions


def build_preencoder(args):
    mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=args.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder


def build_encoder(args):
    if args.enc_type == "vanilla":
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=args.enc_nlayers
        )
    elif args.enc_type in ["masked"]:
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        interim_downsampling = PointnetSAModuleVotes(
            radius=0.4,
            nsample=32,
            npoint=args.preenc_npoints // 2,
            mlp=[args.enc_dim, 256, 256, args.enc_dim],
            normalize_xyz=True,
        )
        
        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
        )
    else:
        raise ValueError(f"Unknown encoder type {args.enc_type}")
    return encoder


def build_decoder(args):
    decoder_layer = TransformerDecoderLayer(
        d_model=args.dec_dim,
        nhead=args.dec_nhead,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
    )
    return decoder


class DETR_3D_2D(nn.Module):
	def __init__(self, args, pc_model, img_model):
		super().__init__()

		if args.clip_model in ["ViT-L/14@336px"]:
			self.patch_size = 336
		elif args.clip_model in ["ViT-B/32"]:
			self.patch_size = 224
		else:
			print("Error: No such clip model")
			traceback.print_stack()
			exit()


		self.pc_model = pc_model
		self.img_model = img_model
		# self.img_clip_header = nn.Parameter(torch.empty(512, 512))
		# nn.init.normal_(self.img_clip_header, std=512 ** -0.5)
		# self.text_clip_header = nn.Parameter(torch.empty(512, 512))
		# nn.init.normal_(self.text_clip_header, std=512 ** -0.5)

		self.text = ["A photo of human",
                    "A photo of sneakers",
                    "A photo of chair",
                    "A photo of hat",
                    "A photo of lamp",
                    "A photo of bottle",
                    "A photo of cabinet/shelf",
                    "A photo of cup",
                    "A photo of car",
                    "A photo of glasses",
                    "A photo of picture/frame",
                    "A photo of desk",
                    "A photo of handbag",
                    "A photo of street lights",
                    "A photo of book",
                    "A photo of plate",
                    "A photo of helmet",
                    "A photo of leather shoes",
                    "A photo of pillow",
                    "A photo of glove",
                    "A photo of potted plant",
                    "A photo of bracelet",
                    "A photo of flower",
                    "A photo of monitor",
                    "A photo of storage box",
                    "A photo of plants pot/vase",
                    "A photo of bench",
                    "A photo of wine glass",
                    "A photo of boots",
                    "A photo of dining table",
                    "A photo of umbrella",
                    "A photo of boat",
                    "A photo of flag",
                    "A photo of speaker",
                    "A photo of trash bin/can",
                    "A photo of stool",
                    "A photo of backpack",
                    "A photo of sofa",
                    "A photo of belt",
                    "A photo of carpet",
                    "A photo of basket",
                    "A photo of towel/napkin",
                    "A photo of slippers",
                    "A photo of bowl",
                    "A photo of barrel/bucket",
                    "A photo of coffee table",
                    "A photo of suv",
                    "A photo of toy",
                    "A photo of tie",
                    "A photo of bed",
                    "A photo of traffic light",
                    "A photo of pen/pencil",
                    "A photo of microphone",
                    "A photo of sandals",
                    "A photo of canned",
                    "A photo of necklace",
                    "A photo of mirror",
                    "A photo of faucet",
                    "A photo of bicycle",
                    "A photo of bread",
                    "A photo of high heels",
                    "A photo of ring",
                    "A photo of van",
                    "A photo of watch",
                    "A photo of combine with bowl",
                    "A photo of sink",
                    "A photo of horse",
                    "A photo of fish",
                    "A photo of apple",
                    "A photo of traffic sign",
                    "A photo of camera",
                    "A photo of candle",
                    "A photo of stuffed animal",
                    "A photo of cake",
                    "A photo of motorbike/motorcycle",
                    "A photo of wild bird",
                    "A photo of laptop",
                    "A photo of knife",
                    "A photo of cellphone",
                    "A photo of paddle",
                    "A photo of truck",
                    "A photo of cow",
                    "A photo of power outlet",
                    "A photo of clock",
                    "A photo of drum",
                    "A photo of fork",
                    "A photo of bus",
                    "A photo of hanger",
                    "A photo of nightstand",
                    "A photo of pot/pan",
                    "A photo of sheep",
                    "A photo of guitar",
                    "A photo of traffic cone",
                    "A photo of tea pot",
                    "A photo of keyboard",
                    "A photo of tripod",
                    "A photo of hockey stick",
                    "A photo of fan",
                    "A photo of dog",
                    "A photo of spoon",
                    "A photo of blackboard/whiteboard",
                    "A photo of balloon",
                    "A photo of air conditioner",
                    "A photo of cymbal",
                    "A photo of mouse",
                    "A photo of telephone",
                    "A photo of pickup truck",
                    "A photo of orange",
                    "A photo of banana",
                    "A photo of airplane",
                    "A photo of luggage",
                    "A photo of skis",
                    "A photo of soccer",
                    "A photo of trolley",
                    "A photo of oven",
                    "A photo of remote",
                    "A photo of combine with glove",
                    "A photo of paper towel",
                    "A photo of refrigerator",
                    "A photo of train",
                    "A photo of tomato",
                    "A photo of machinery vehicle",
                    "A photo of tent",
                    "A photo of shampoo/shower gel",
                    "A photo of head phone",
                    "A photo of lantern",
                    "A photo of donut",
                    "A photo of cleaning products",
                    "A photo of sailboat",
                    "A photo of tangerine",
                    "A photo of pizza",
                    "A photo of kite",
                    "A photo of computer box",
                    "A photo of elephant",
                    "A photo of toiletries",
                    "A photo of gas stove",
                    "A photo of broccoli",
                    "A photo of toilet",
                    "A photo of stroller",
                    "A photo of shovel",
                    "A photo of baseball bat",
                    "A photo of microwave",
                    "A photo of skateboard",
                    "A photo of surfboard",
                    "A photo of surveillance camera",
                    "A photo of gun",
                    "A photo of Life saver",
                    "A photo of cat",
                    "A photo of lemon",
                    "A photo of liquid soap",
                    "A photo of zebra",
                    "A photo of duck",
                    "A photo of sports car",
                    "A photo of giraffe",
                    "A photo of pumpkin",
                    "A photo of Accordion/keyboard/piano",
                    "A photo of radiator",
                    "A photo of converter",
                    "A photo of tissue ",
                    "A photo of carrot",
                    "A photo of washing machine",
                    "A photo of vent",
                    "A photo of cookies",
                    "A photo of cutting/chopping board",
                    "A photo of tennis racket",
                    "A photo of candy",
                    "A photo of skating and skiing shoes",
                    "A photo of scissors",
                    "A photo of folder",
                    "A photo of baseball",
                    "A photo of strawberry",
                    "A photo of bow tie",
                    "A photo of pigeon",
                    "A photo of pepper",
                    "A photo of coffee machine",
                    "A photo of bathtub",
                    "A photo of snowboard",
                    "A photo of suitcase",
                    "A photo of grapes",
                    "A photo of ladder",
                    "A photo of pear",
                    "A photo of american football",
                    "A photo of basketball",
                    "A photo of potato",
                    "A photo of paint brush",
                    "A photo of printer",
                    "A photo of billiards",
                    "A photo of fire hydrant",
                    "A photo of goose",
                    "A photo of projector",
                    "A photo of sausage",
                    "A photo of fire extinguisher",
                    "A photo of extension cord",
                    "A photo of facial mask",
                    "A photo of tennis ball",
                    "A photo of chopsticks",
                    "A photo of Electronic stove and gas stove",
                    "A photo of pie",
                    "A photo of frisbee",
                    "A photo of kettle",
                    "A photo of hamburger",
                    "A photo of golf club",
                    "A photo of cucumber",
                    "A photo of clutch",
                    "A photo of blender",
                    "A photo of tong",
                    "A photo of slide",
                    "A photo of hot dog",
                    "A photo of toothbrush",
                    "A photo of facial cleanser",
                    "A photo of mango",
                    "A photo of deer",
                    "A photo of egg",
                    "A photo of violin",
                    "A photo of marker",
                    "A photo of ship",
                    "A photo of chicken",
                    "A photo of onion",
                    "A photo of ice cream",
                    "A photo of tape",
                    "A photo of wheelchair",
                    "A photo of plum",
                    "A photo of bar soap",
                    "A photo of scale",
                    "A photo of watermelon",
                    "A photo of cabbage",
                    "A photo of router/modem",
                    "A photo of golf ball",
                    "A photo of pine apple",
                    "A photo of crane",
                    "A photo of fire truck",
                    "A photo of peach",
                    "A photo of cello",
                    "A photo of notepaper",
                    "A photo of tricycle",
                    "A photo of toaster",
                    "A photo of helicopter",
                    "A photo of green beans",
                    "A photo of brush",
                    "A photo of carriage",
                    "A photo of cigar",
                    "A photo of earphone",
                    "A photo of penguin",
                    "A photo of hurdle",
                    "A photo of swing",
                    "A photo of radio",
                    "A photo of CD",
                    "A photo of parking meter",
                    "A photo of swan",
                    "A photo of garlic",
                    "A photo of french fries",
                    "A photo of horn",
                    "A photo of avocado",
                    "A photo of saxophone",
                    "A photo of trumpet",
                    "A photo of sandwich",
                    "A photo of cue",
                    "A photo of kiwi fruit",
                    "A photo of bear",
                    "A photo of fishing rod",
                    "A photo of cherry",
                    "A photo of tablet",
                    "A photo of green vegetables",
                    "A photo of nuts",
                    "A photo of corn",
                    "A photo of key",
                    "A photo of screwdriver",
                    "A photo of globe",
                    "A photo of broom",
                    "A photo of pliers",
                    "A photo of hammer",
                    "A photo of volleyball",
                    "A photo of eggplant",
                    "A photo of trophy",
                    "A photo of board eraser",
                    "A photo of dates",
                    "A photo of rice",
                    "A photo of tape measure/ruler",
                    "A photo of dumbbell",
                    "A photo of hamimelon",
                    "A photo of stapler",
                    "A photo of camel",
                    "A photo of lettuce",
                    "A photo of goldfish",
                    "A photo of meat balls",
                    "A photo of medal",
                    "A photo of toothpaste",
                    "A photo of antelope",
                    "A photo of shrimp",
                    "A photo of rickshaw",
                    "A photo of trombone",
                    "A photo of pomegranate",
                    "A photo of coconut",
                    "A photo of jellyfish",
                    "A photo of mushroom",
                    "A photo of calculator",
                    "A photo of treadmill",
                    "A photo of butterfly",
                    "A photo of egg tart",
                    "A photo of cheese",
                    "A photo of pomelo",
                    "A photo of pig",
                    "A photo of race car",
                    "A photo of rice cooker",
                    "A photo of tuba",
                    "A photo of crosswalk sign",
                    "A photo of papaya",
                    "A photo of hair dryer",
                    "A photo of green onion",
                    "A photo of chips",
                    "A photo of dolphin",
                    "A photo of sushi",
                    "A photo of urinal",
                    "A photo of donkey",
                    "A photo of electric drill",
                    "A photo of spring rolls",
                    "A photo of tortoise/turtle",
                    "A photo of parrot",
                    "A photo of flute",
                    "A photo of measuring cup",
                    "A photo of shark",
                    "A photo of steak",
                    "A photo of poker card",
                    "A photo of binoculars",
                    "A photo of llama",
                    "A photo of radish",
                    "A photo of noodles",
                    "A photo of mop",
                    "A photo of yak",
                    "A photo of crab",
                    "A photo of microscope",
                    "A photo of barbell",
                    "A photo of Bread/bun",
                    "A photo of baozi",
                    "A photo of lion",
                    "A photo of red cabbage",
                    "A photo of polar bear",
                    "A photo of lighter",
                    "A photo of mangosteen",
                    "A photo of seal",
                    "A photo of comb",
                    "A photo of eraser",
                    "A photo of pitaya",
                    "A photo of scallop",
                    "A photo of pencil case",
                    "A photo of saw",
                    "A photo of table tennis  paddle",
                    "A photo of okra",
                    "A photo of starfish",
                    "A photo of monkey",
                    "A photo of eagle",
                    "A photo of durian",
                    "A photo of rabbit",
                    "A photo of game board",
                    "A photo of french horn",
                    "A photo of ambulance",
                    "A photo of asparagus",
                    "A photo of hoverboard",
                    "A photo of pasta",
                    "A photo of target",
                    "A photo of hotair balloon",
                    "A photo of chainsaw",
                    "A photo of lobster",
                    "A photo of iron",
                    "A photo of flashlight",
                    "A photo of unclear image"]

		device = "cuda" if torch.cuda.is_available() else "cpu"
		text = clip.tokenize(self.text).to(device)
		self.text_feats = self.batch_encode_text(text)
		# self.text_feats = self.img_model.encode_text(text).detach()

		self.text_num = self.text_feats.shape[0] -1 
		self.text_label = torch.arange(self.text_num, dtype=torch.int).to(device)

		if args.dataset_name in ["sunrgbd"]:
			self.eval_text = ["A photo of chair",
							"A photo of table",
							"A photo of pillow",
							"A photo of desk",
							"A photo of bed",
							"A photo of sofa",
							"A photo of lamp",
							"A photo of garbage_bin",
							"A photo of cabinet",
							"A photo of sink",
							"A photo of night_stand",
							"A photo of stool",
							"A photo of bookshelf",
							"A photo of dresser",
							"A photo of toilet",
							"A photo of fridge",
							"A photo of microwave",
							"A photo of counter",
							"A photo of bathtub",
							"A photo of scanner",
							'A photo of ground',
							'A photo of wall',
							'A photo of floor',
							"An unclear image",
							"A photo of background",
							"There is no object in this image",]
		elif args.dataset_name in ["scannet"]:
			self.eval_text = ["A photo of toilet",
							"A photo of bed",
							"A photo of chair",
							"A photo of sofa",
							"A photo of dresser",
							"A photo of table",
							"A photo of cabinet",
							"A photo of bookshelf",
							"A photo of pillow",
							"A photo of sink",
							"A photo of bathtub",
							"A photo of refridgerator",
							"A photo of desk",
							"A photo of night stand",
							"A photo of counter",
							"A photo of door",
							"A photo of curtain",
							"A photo of box",
							"A photo of lamp",
							"A photo of bag",
							'A photo of ground',
							'A photo of wall',
							'A photo of floor',
							"An unclear image",
							"A photo of background",
							"There is no object in this image",]
		else:
			print("Error: No such dataset")
			traceback.print_stack()
			exit()

		eval_text = clip.tokenize(self.eval_text).to(device)
		self.eval_text_feats = self.batch_encode_text(eval_text)\
		# self.text_feats = self.img_model.encode_text(text).detach()

		self.eval_text_num = self.eval_text_feats.shape[0] - 6
		self.eval_text_label = torch.arange(self.eval_text_num, dtype=torch.int).to(device)

	def batch_encode_text(self, text):
		batch_size = 20

		text_num = text.shape[0]
		cur_start = 0
		cur_end = 0

		all_text_feats = []
		while cur_end < text_num:
			# print(cur_end)
			cur_start = cur_end
			cur_end += batch_size
			if cur_end >= text_num:
				cur_end = text_num
			
			cur_text = text[cur_start:cur_end,:]
			cur_text_feats = self.img_model.encode_text(cur_text).detach()
			all_text_feats.append(cur_text_feats)

		all_text_feats = torch.cat(all_text_feats, dim=0)
		# print(all_text_feats.shape)
		return all_text_feats

	def accuracy(self, image_features, text_features, img_label):
		image_features = image_features / image_features.norm(dim=1, keepdim=True)
		text_features = text_features / text_features.norm(dim=1, keepdim=True)

		logit_scale_ = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
		logit_scale = logit_scale_.exp()

		logits_per_image = logit_scale * image_features @ text_features.t()
		logits_per_text = logits_per_image.t()

		probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
		rst = np.argmax(probs, axis=1)

		img_label = img_label.reshape([-1]).detach().cpu().numpy()

		diff = rst-img_label

		correct_cnt = np.where(diff==0)[0]

		print(rst)
		print(img_label)
		input()
		print(rst-img_label)
		print(correct_cnt.shape[0]/diff.shape[0])
		input()

		# shape = [global_batch_size, global_batch_size]
		return


	def classify(self, image_features, text_features, verbose=False, img_label=None):
		image_features = image_features / image_features.norm(dim=1, keepdim=True)
		text_features = text_features / text_features.norm(dim=1, keepdim=True)

		logit_scale_ = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
		logit_scale = logit_scale_.exp()

		logits_per_image = logit_scale * image_features @ text_features.t()
		logits_per_text = logits_per_image.t()

		probs_ = logits_per_image.softmax(dim=-1)
		probs, rst = torch.max(probs_, dim=1)
		
		'''
		img_label = img_label.reshape([-1])
		diff = rst-img_label
		correct_cnt = torch.where(diff==0)[0]
		
		print(rst)
		print(img_label)
		input()
		print(rst-img_label)
		print(correct_cnt.shape[0]/diff.shape[0])
		input()
		'''
		if verbose:
			return probs, rst, probs_, logits_per_image
		else:
			return probs, rst

	def my_compute_box_3d(self, center, size, heading_angle):
		R = pc_util.rotz(-1 * heading_angle)
		l, w, h = size
		x_corners = [-l, l, l, -l, -l, l, l, -l]
		y_corners = [w, w, -w, -w, w, w, -w, -w]
		z_corners = [h, h, h, h, -h, -h, -h, -h]
		corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
		corners_3d[0, :] += center[0]
		corners_3d[1, :] += center[1]
		corners_3d[2, :] += center[2]
		return np.transpose(corners_3d)

	def compute_all_bbox_corners(self, pred_bboxes):
		batch_size, bbox_num, _ = pred_bboxes.shape

		all_corners = []
		for cur_bs in range(batch_size):
			cur_bs_corners = []
			for cur_bbox in range(bbox_num):
				cur_center = pred_bboxes[cur_bs, cur_bbox, :3]
				cur_size = pred_bboxes[cur_bs, cur_bbox, 3:6] / 2
				cur_heading = pred_bboxes[cur_bs, cur_bbox, 6]
				cur_corners = self.my_compute_box_3d(cur_center, cur_size, cur_heading)
				cur_bs_corners.append(cur_corners)

			cur_bs_corners = np.stack(cur_bs_corners, axis=0)
			all_corners.append(cur_bs_corners)

		all_corners = np.stack(all_corners, axis=0)

		return all_corners

	def restore_aug(self, pred_bboxes, Aug_param):
		'''
		bboxes_after=bboxes.copy()
		bboxes_after[:,3:6] *= 2
		bboxes_after[:, 6] *= -1
		np.savetxt("pc_after_aug.txt", point_cloud, fmt="%.3f")
		write_oriented_bbox(bboxes_after[:,:7], "gt_after_aug.ply")
		'''

		pred_bboxes_restore=pred_bboxes.copy()

		batch_size = pred_bboxes.shape[0]
		
		for bs_ind in range(batch_size):
			#print("hehe")
			Aug_Scale = Aug_param["Aug_Scale"][bs_ind, 0, ...].detach().cpu().numpy()
			Aug_Rot_Ang = Aug_param["Aug_Rot_Ang"][bs_ind].detach().cpu().numpy()
			Aug_Rot_Mat = Aug_param["Aug_Rot_Mat"][bs_ind,...].detach().cpu().numpy()
			Aug_Flip = Aug_param["Aug_Flip"][bs_ind].detach().cpu().numpy()
			
			'''
			print(Aug_Scale)
			print(Aug_Rot_Ang)
			print(Aug_Rot_Mat)
			print(Aug_Flip)
			'''

			#print(pred_bboxes_restore[bs_ind, 0, :])
			pred_bboxes_restore[bs_ind, :, :3] /= Aug_Scale
			pred_bboxes_restore[bs_ind, :, 3:6] /= Aug_Scale
			#print(pred_bboxes_restore[bs_ind, 0, :])
			pred_bboxes_restore[bs_ind, :, 6] += Aug_Rot_Ang
			pred_bboxes_restore[bs_ind, :, 0:3] = np.dot(pred_bboxes_restore[bs_ind, :, 0:3], Aug_Rot_Mat)

			if Aug_Flip == 1:
				pred_bboxes_restore[bs_ind, :, 6] = np.pi - pred_bboxes_restore[bs_ind, :, 6]
				pred_bboxes_restore[bs_ind, :, 0] = -1 * pred_bboxes_restore[bs_ind, :, 0]

		return pred_bboxes_restore

	def proj_pointcloud_into_image(self, xyz, pose, calib_K):
		pose = pose.detach().cpu().numpy()
		calib_K = calib_K.detach().cpu().numpy()

		padding = np.ones([xyz.shape[0], 1])
		xyzp = np.concatenate([xyz, padding], axis=1)
		xyz = (pose @ xyzp.T)[:3,:].T

		intrinsic_image = calib_K[:3,:3]
		xyz_uniform = xyz/xyz[:,2:3]
		xyz_uniform = xyz_uniform.T

		uv = intrinsic_image @ xyz_uniform

		uv /= uv[2:3, :]
		uv = np.around(uv).astype(np.int)

		uv = uv.T

		return uv[:,:2]


	def compute_all_bbox_corners_2d(self, pred_bboxes_corners, calib_param):
		'''
		print("calib")
		print(pred_bboxes_corners.shape)
		print(calib_param)
		print(calib_param["calib_K"].shape)
		print(calib_param["calib_Rtilt"].shape)
		'''

		calib_Rtilt = calib_param["calib_Rtilt"]
		calib_K = calib_param["calib_K"]

		batch_size, box_num, _, _ = pred_bboxes_corners.shape

		all_corners_2d = []
		for bs_ind in range(batch_size):
			#calib = SUNRGBD_Calibration(calib_Rtilt[0,:,:], calib_K[0, :, :])
			cur_calib_Rtilt = calib_Rtilt[bs_ind,:,:]
			cur_calib_K = calib_K[bs_ind, :, :]

			cur_batch_corners_2d = []
			for box_ind in range(box_num):
				cur_corners_3d = pred_bboxes_corners[bs_ind, box_ind, :, :]

				cur_corners_2d = self.proj_pointcloud_into_image(cur_corners_3d, cur_calib_Rtilt, cur_calib_K)
				cur_batch_corners_2d.append(cur_corners_2d)

			cur_batch_corners_2d = np.stack(cur_batch_corners_2d, axis=0)
			all_corners_2d.append(cur_batch_corners_2d)

		all_corners_2d = np.stack(all_corners_2d, axis=0)
		#print(all_corners_2d.shape)
		return all_corners_2d

	def img_preprocess(self, img):
		# clip normalize
		def resize(img, size):
			img_h, img_w, _ = img.shape
			if img_h > img_w:
				new_w = int(img_w * size / img_h)
				new_h = size
			else:
				new_w = size
				new_h = int(img_h * size / img_w)

			dsize = (new_h, new_w)
			transform = T.Resize(dsize, interpolation=BICUBIC)

			img = img.permute([2,0,1])
			img = transform(img)
			img = img.permute([1,2,0])
			
			return img

		def center_crop(img, dim):
			"""Returns center cropped image
			Args:
			img: image to be center cropped
			dim: dimensions (width, height) to be cropped
			"""
			width, height = img.shape[1], img.shape[0]

			# process crop width and height for max available dimension
			crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
			crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
			mid_x, mid_y = int(width/2), int(height/2)
			cw2, ch2 = int(crop_width/2), int(crop_height/2) 
			crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
			return crop_img

		def padding(img, dim):
			rst_img = torch.zeros([dim[0], dim[1], 3], dtype=img.dtype, device=img.device)
			h, w, _ = img.shape

			top_left = (np.array(dim)/2 - np.array([h,w])/2).astype(np.int)
			down_right = (top_left + np.array([h,w])).astype(np.int)

			rst_img[top_left[0]:down_right[0], top_left[1]:down_right[1], :] = img.clone()
			return rst_img

		img = resize(img, self.patch_size)
		img = padding(img, [self.patch_size,self.patch_size])
		return img

	def img_denorm(self, img):
		img = img.detach().cpu().numpy()
		imagenet_std = np.array([0.26862954, 0.26130258, 0.27577711])
		imagenet_mean = np.array([0.48145466, 0.4578275, 0.40821073])
		img = (img * imagenet_std + imagenet_mean) * 255
		#img = ((img/255) - imagenet_mean) / imagenet_std
		return img

	def crop_patches(self, image, corners_2d, image_size, corners_3d, point_cloud, pred_bboxes=None):

		def is_valid_patch(top_left, down_right, cur_corner_3d, cur_point_cloud):
			patch_ori_size = down_right - top_left
			patch_ori_size = patch_ori_size[0] * patch_ori_size[1]

			if np.isnan(top_left[0]) or np.isnan(top_left[1]) or np.isnan(down_right[0]) or np.isnan(down_right[1]):
				return False, np.array([0,0]), np.array([10,10])

			if top_left[0] < 0:
				top_left[0] = 0

			if top_left[1] < 0:
				top_left[1] = 0

			if top_left[0] >= img_width:
				return False, np.array([0,0]), np.array([10,10])

			if top_left[1] >= img_height:
				return False, np.array([0,0]), np.array([10,10])

			if down_right[0] > img_width:
				down_right[0] = img_width - 1

			if down_right[1] > img_height:
				down_right[1] = img_height - 1

			if down_right[0] <= 0:
				return False, np.array([0,0]), np.array([10,10])

			if down_right[1] <= 0:
				return False, np.array([0,0]), np.array([10,10])

			patch_fixed_size = down_right - top_left

			if patch_fixed_size[0] < 10 or patch_fixed_size[1] < 10:
				return False, np.array([0,0]), np.array([10,10])

			patch_fixed_size = patch_fixed_size[0] * patch_fixed_size[1]

			if patch_fixed_size/patch_ori_size < 0.8:
				return False, top_left, down_right

			# omit if there is no points in the bounding box
			cur_corner_3d = corners_3d[bs_ind, box_ind, :, :]
			pc, _ = extract_pc_in_box3d(cur_point_cloud, cur_corner_3d)
			if pc.shape[0] < 100:
				return False, top_left, down_right
			return True, top_left, down_right


		batch_size, box_num, _, _ = corners_2d.shape

		all_patch = []

		all_valid = np.zeros([batch_size, box_num], dtype=np.bool)

		for bs_ind in range(batch_size):
			cur_bs_img = image[bs_ind, ...]
			img_width = image_size[bs_ind, 0]
			img_height = image_size[bs_ind, 1]

			# print(img_width, " ", img_height)
			# cur_bs_img_ = self.img_denorm(cur_bs_img)
			# cv2.imwrite("image.jpg", cur_bs_img_)

			cur_bs_patch = []

			for box_ind in range(box_num):
				cur_corners_2d = corners_2d[bs_ind, box_ind, :, :]
				top_left = np.min(cur_corners_2d, axis=0)
				down_right = np.max(cur_corners_2d, axis=0)

				valid_flag, top_left, down_right = is_valid_patch(top_left, down_right, corners_3d[bs_ind, box_ind, :, :], point_cloud[bs_ind,:,:])
				all_valid[bs_ind, box_ind] = valid_flag

				# print(valid_flag)
				# print(top_left)
				# print(down_right)
				# input()

				# patch_ori_size = down_right - top_left
				# patch_ori_size = patch_ori_size[0] * patch_ori_size[1]

				# if np.isnan(top_left[0]) or np.isnan(top_left[1]) or np.isnan(down_right[0]) or np.isnan(down_right[1]):
				# 	continue

				# if top_left[0] < 0:
				# 	top_left[0] = 0

				# if top_left[1] < 0:
				# 	top_left[1] = 0

				# if top_left[0] >= img_width:
				# 	continue

				# if top_left[1] >= img_height:
				# 	continue

				# if down_right[0] > img_width:
				# 	down_right[0] = img_width - 1

				# if down_right[1] > img_height:
				# 	down_right[1] = img_height - 1

				# if down_right[0] <= 0:
				# 	continue

				# if down_right[1] <= 0:
				# 	continue

				# patch_fixed_size = down_right - top_left
				# patch_fixed_size = patch_fixed_size[0] * patch_fixed_size[1]

				# if patch_fixed_size/patch_ori_size < 0.8:
				# 	continue

				# # omit if there is no points in the bounding box
				# cur_corner_3d = corners_3d[bs_ind, box_ind, :, :]
				# pc, _ = extract_pc_in_box3d(point_cloud, cur_corner_3d)
				# if pc.shape[0] < 100:
				# 	continue

				# #print(patch_fixed_size/patch_ori_size)
				# #print(top_left)
				# #print(down_right)

				top_left = (int(top_left[0]), int(top_left[1]))
				down_right = (int(down_right[0]), int(down_right[1]))

				cur_patch = cur_bs_img[top_left[1]:down_right[1], top_left[0]:down_right[0], :].clone()


				#print(cur_patch.shape)
				#cur_patch_denorm = img_denorm(cur_patch)
				#print(cur_patch_denorm.shape)
				#cv2.imwrite("patch_%03d_%03d_%03d.jpg"%(bs_ind, box_ind, len(all_patch)), cur_patch_denorm)
				cur_patch = self.img_preprocess(cur_patch)
				#print(cur_patch.shape)
				#cv2.imwrite("patch_%03d_%03d_1.jpg"%(bs_ind, box_ind), cur_patch.detach().cpu().numpy())

				cur_bs_patch.append(cur_patch)

				
				'''
				cur_patch = cur_bs_img[top_left[1]:down_right[1], top_left[0]:down_right[0], :]
				cur_patch_ = img_denorm(cur_patch)
				cv2.imwrite("patch_%03d_%03d.jpg"%(bs_ind, box_ind), cur_patch_)


				pair_3d_bbox = np.expand_dims(pred_bboxes[bs_ind, box_ind, ...], axis=0)
				print(pair_3d_bbox.shape)

				pair_3d_bbox[..., 6] *= -1
				write_oriented_bbox(pair_3d_bbox, "pair_3d_bbox_%03d_%03d.ply"%(bs_ind, box_ind))

				print("valid")
				input()
				'''
			cur_bs_patch = torch.stack(cur_bs_patch, dim=0)
			all_patch.append(cur_bs_patch)

		all_patch = torch.stack(all_patch, dim=0)

		all_patch = all_patch.permute(0,1,4,2,3)
		return all_patch, all_valid


	def collect_valid_pc_img_feat(self, in_pc_feat, valid_3d_bbox, pair_img_cnt, pair_img_porb, pair_img_label, pair_img_feat):
		'''
		print(in_pc_feat.shape)
		print(valid_3d_bbox)
		print(pair_img_cnt)
		print(pair_img_porb)
		print(pair_img_label)
		'''

		valid_3d_bbox_num = 0
		for bs_ind in range(len(valid_3d_bbox)):
			valid_3d_bbox_num+=len(valid_3d_bbox[bs_ind])

		#print(valid_3d_bbox_num)
		
		assert valid_3d_bbox_num == pair_img_cnt

		bbox_counter=0

		out_pc_feat = []
		out_label = []
		out_prob = []
		out_img_feat = []
		
		
		for bs_ind in range(len(valid_3d_bbox)):
			cur_bs_valid_bbox = valid_3d_bbox[bs_ind]

			for bbox_ind in range(len(cur_bs_valid_bbox)):
				
				cur_pc_feat = in_pc_feat[cur_bs_valid_bbox[bbox_ind], bs_ind, :]
				cur_label = pair_img_label[bbox_counter]
				cur_prob = pair_img_porb[bbox_counter]
				cur_img_feat = pair_img_feat[bbox_counter, :]

				out_img_feat.append(cur_img_feat)
				out_pc_feat.append(cur_pc_feat)
				out_label.append(cur_label)
				out_prob.append(cur_prob)
				bbox_counter += 1


		out_pc_feat = torch.stack(out_pc_feat, dim=0)
		out_label = torch.stack(out_label, dim=0)
		out_prob = torch.stack(out_prob, dim=0)
		out_img_feat = torch.stack(out_img_feat, dim=0)

		assert valid_3d_bbox_num == pair_img_cnt and valid_3d_bbox_num == bbox_counter
		

		pair_img_output={"pair_img_feat": out_img_feat,
						 "pair_img_label": out_label,
						 "pair_img_prob": out_prob,
						}
		
		pc_output={"pc_feat": out_pc_feat,
						 "pc_label": out_label,
						 "pc_prob": out_prob,
						}
		return pair_img_output, pc_output

	def classify_pc(self, pc_query_feat, text_feat, text_num):
		query_num, batch_size, feat_dim = pc_query_feat.shape
		pc_query_feat = pc_query_feat.reshape([-1, feat_dim])

		pc_all_porb, pc_all_label, pc_all_porb_ori, _ = self.classify(pc_query_feat.half(), text_feat, verbose=True)

		pc_all_label = pc_all_label.reshape([query_num, batch_size])
		pc_all_porb = pc_all_porb.reshape([query_num, batch_size])
		pc_all_porb_ori = pc_all_porb_ori.reshape([query_num, batch_size, -1])

		pc_all_logits = torch.zeros([query_num, batch_size, text_num+1], device=pc_all_porb_ori.device)
		pc_all_logits[..., :text_num] = torch.log(pc_all_porb_ori[...,:text_num])
		pc_all_logits[..., text_num] = torch.log(torch.sum(pc_all_porb_ori[...,text_num:], dim=-1))

		#print(pc_all_porb_ori[0,0,:])
		#print(pc_all_logits.softmax(dim=-1)[0,0,:])
		#exit()

		pc_all_logits = pc_all_logits.permute(1,0,2)
		pc_all_porb = pc_all_porb.permute(1,0)
		pc_all_porb_ori = pc_all_porb_ori[:,:,:text_num].permute(1,0,2)
		pc_all_label = pc_all_label.permute(1,0)
		return pc_all_logits, pc_all_porb, pc_all_porb_ori, pc_all_label
		
	def pred_bbox_nms(self, pred_bboxes_corners, valid_3d_bbox, pair_img_porb, pair_img_label, pair_img_output):
		batch_size, query_num, _, _ = pred_bboxes_corners.shape

		new_valid_3d_bbox = []

		bbox_counter = 0
		picked_counter = []
		for bs_ind in range(len(valid_3d_bbox)):
			boxes_3d_with_prob = []
			cur_bs_valid_bbox = valid_3d_bbox[bs_ind]

			bbox_counter_list=[]
			for bbox_ind in range(len(cur_bs_valid_bbox)):

				cur_bbox_with_prob = np.zeros(7)
				cur_bbox_ind = cur_bs_valid_bbox[bbox_ind]
				cur_bbox_corner = pred_bboxes_corners[bs_ind, cur_bbox_ind, :, :]
				#print(cur_bbox_corner.shape)

				cur_bbox_with_prob[0] = np.min(cur_bbox_corner[:, 0])
				cur_bbox_with_prob[1] = np.min(cur_bbox_corner[:, 1])
				cur_bbox_with_prob[2] = np.min(cur_bbox_corner[:, 2])
				cur_bbox_with_prob[3] = np.max(cur_bbox_corner[:, 0])
				cur_bbox_with_prob[4] = np.max(cur_bbox_corner[:, 1])
				cur_bbox_with_prob[5] = np.max(cur_bbox_corner[:, 2])
				cur_bbox_with_prob[6] = pair_img_porb[bbox_counter]
				bbox_counter_list.append(bbox_counter)

				boxes_3d_with_prob.append(cur_bbox_with_prob)
				bbox_counter += 1

			if len(boxes_3d_with_prob) <= 0:
				pick=[]
			else:
				boxes_3d_with_prob = np.stack(boxes_3d_with_prob, axis=0)
				pick = nms_3d_faster(boxes_3d_with_prob, overlap_threshold = 0.05)

			cur_bs_valid_bbox = np.array(cur_bs_valid_bbox)
			new_valid_3d_bbox.append(cur_bs_valid_bbox[pick].tolist())
			#print(new_valid_3d_bbox)
			
			bbox_counter_list = np.array(bbox_counter_list)
			picked_counter.extend(bbox_counter_list[pick].tolist())

			#print(pair_img_porb[bbox_counter_list[pick]])
			#print(picked_counter)

			#print(boxes_3d_with_prob[pick,:])
			#input()

		#print(picked_counter)
		#print(pair_img_porb[picked_counter])
		#print(new_valid_3d_bbox)

		pair_img_porb = pair_img_porb[picked_counter]
		pair_img_label = pair_img_label[picked_counter]
		pair_img_output = pair_img_output[picked_counter, :]
		pair_img_cnt = pair_img_porb.shape[0]

		#print(pair_img_label)
		#print(pair_img_output)
		#input(pair_img_cnt)

		return pair_img_porb, pair_img_label, pair_img_output, new_valid_3d_bbox, pair_img_cnt

	def collect_matched_patch(self, cur_patches, assignments):
		batch_size, query_num, _, _, _ = cur_patches.shape
		all_matched_patch = []

		for cur_bs in range(batch_size):
			cur_assignments = assignments["assignments"][cur_bs]

			cur_matched_cnt = cur_assignments[0].shape[0]
			#print(cur_matched_cnt)

			for cur_match in range(cur_matched_cnt):
				cur_match_patch_ind = cur_assignments[0][cur_match]
				cur_match_gt_ind = cur_assignments[1][cur_match]

				cur_match_patch = cur_patches[cur_bs, cur_match_patch_ind, ...]
				all_matched_patch.append(cur_match_patch)

		all_matched_patch = torch.stack(all_matched_patch, dim=0)
		return all_matched_patch

	def collect_matched_pred_bbox(self, pred_bboxes, assignments):
		batch_size, query_num, _ = pred_bboxes.shape
		all_matched_bbox = []

		for cur_bs in range(batch_size):
			cur_assignments = assignments["assignments"][cur_bs]

			cur_matched_cnt = cur_assignments[0].shape[0]
			#print(cur_matched_cnt)

			for cur_match in range(cur_matched_cnt):
				cur_match_bbox_ind = cur_assignments[0][cur_match]
				cur_match_gt_ind = cur_assignments[1][cur_match]

				cur_match_bbox = pred_bboxes[cur_bs, cur_match_bbox_ind, ...]
				all_matched_bbox.append(cur_match_bbox)

		all_matched_bbox = np.stack(all_matched_bbox, axis=0)
		return all_matched_bbox

	def collect_matched_query_feat(self, pc_query_feat, targets, assignments):
		# print(pc_query_feat.shape)
		# input()
		# print(assignments)
		# input()

		# for key,val in targets.items():
		# 	print(key)

		# input()
		# print(targets["gt_box_sem_cls_label"])
		# input()
		query_num, batch_size, feat_dim = pc_query_feat.shape

		all_matched_query = []
		all_matched_label = []
		all_matched_prob = []


		for cur_bs in range(batch_size):
			cur_assignments = assignments["assignments"][cur_bs]
			cur_matched_cnt = cur_assignments[0].shape[0]
			#print(cur_matched_cnt)

			for cur_match in range(cur_matched_cnt):
				cur_match_query_ind = cur_assignments[0][cur_match]
				cur_match_gt_ind = cur_assignments[1][cur_match]

				cur_match_query = pc_query_feat[cur_match_query_ind, cur_bs, :]
				cur_match_gt = targets["gt_box_sem_cls_label"][cur_bs, cur_match_gt_ind]

				all_matched_query.append(cur_match_query)
				all_matched_label.append(cur_match_gt)
				all_matched_prob.append(torch.ones(1, device=cur_match_gt.device))

				# print(cur_match_query.shape)
				# print(cur_match_gt)
				# input()

		all_matched_query = torch.stack(all_matched_query, dim=0)
		all_matched_label = torch.stack(all_matched_label, dim=0)
		all_matched_prob = torch.stack(all_matched_prob, dim=0).reshape(-1)

		pc_feat_label={"pc_feat": all_matched_query,
					"pc_label": all_matched_label,
					"pc_prob": all_matched_prob,
					}
		return pc_feat_label
		
	
	def forward(self, pc_input, img_input, criterion=None, targets=None, train_loc_only=True):
		pc_output = self.pc_model(pc_input)
		
		if self.training:
			if not train_loc_only:
				# prepare text feat
				text_output_before_clip_header = self.text_feats
				# text_output = self.text_feats @ self.text_clip_header.half()
				# text_output = text_output / text_output.norm(dim=1, keepdim=True)
				text_output = self.text_feats / self.text_feats.norm(dim=1, keepdim=True)
				text_feat_label = {"text_feat": text_output[:self.text_num, :],
										"text_label": self.text_label}

				# classify point cloud querys
				pc_sem_cls_logits, pc_objectness_prob, pc_sem_cls_prob, pc_all_label = self.classify_pc(pc_output["pc_query_feat"], text_output, self.text_num)
				loss_pc, loss_dict, assignments = criterion(pc_output, targets, return_assignments=True)

				# prepare image feat
				pred_bboxes = torch.cat([pc_output["outputs"]["center_unnormalized"], pc_output["outputs"]["size_unnormalized"], torch.unsqueeze(pc_output["outputs"]["angle_continuous"], dim=-1)], dim=-1).detach().cpu().numpy()
				point_cloud = pc_input["point_clouds"].detach().cpu().numpy()
				pred_bboxes_corners_aug = self.compute_all_bbox_corners(pred_bboxes)
				pred_bboxes = self.restore_aug(pred_bboxes, pc_input["Aug_param"])
				pred_bboxes_corners = self.compute_all_bbox_corners(pred_bboxes)
				pred_bboxes_corners_2d = self.compute_all_bbox_corners_2d(pred_bboxes_corners, img_input["calib"])

				patches, valid_3d_bbox = self.crop_patches(img_input["image"], pred_bboxes_corners_2d, img_input["calib"]["image_size"], pred_bboxes_corners_aug, point_cloud, pred_bboxes)

				# assert valid_3d_bbox
				# valid_3d_bbox_cnt = np.sum(valid_3d_bbox, axis=1)
				# for cur_valid_3d_bbox_cnt in valid_3d_bbox_cnt:
				# 	assert cur_valid_3d_bbox_cnt > 0

				patches = self.collect_matched_patch(patches, assignments)
				pc_feat = self.collect_matched_query_feat(pc_output["pc_query_feat"], targets, assignments)["pc_feat"]
				

				# for validation
				# patches = patches.permute(0,2,3,1)
				# pred_bboxes_ = self.collect_matched_pred_bbox(pred_bboxes, assignments)

				# print(patches.shape)
				# print(pred_bboxes_.shape)
				# print(self.img_denorm(patches[0,...]).shape)

				# for patch_ind in range(patches.shape[0]):
				# 	cur_patch = self.img_denorm(patches[patch_ind,...])
				# 	cur_bbox = pred_bboxes_[patch_ind:patch_ind+1,...]

				# 	print(cur_patch.shape)
				# 	print(cur_bbox.shape)
				# 	cv2.imwrite("patch_%03d.jpg"%(patch_ind), cur_patch)
				# 	write_oriented_bbox(cur_bbox, "pair_3d_bbox_%03d.ply"%(patch_ind))
					
				# exit()

				pair_img_cnt = patches.shape[0]
				img_output = self.img_model.encode_image(patches)

				pair_img_output_before_clip_header = img_output[:pair_img_cnt, :]

				# img_output = img_output @ self.img_clip_header.half()
				img_output = img_output / img_output.norm(dim=1, keepdim=True)

				pair_img_output = img_output[:pair_img_cnt, :]

				pair_img_porb, pair_img_label = self.classify(pair_img_output_before_clip_header, text_output_before_clip_header)

				pc_feat_label={"pc_feat": pc_feat,
						"pc_label": pair_img_label,
						"pc_prob": pair_img_porb,
						}

				pair_feat_label={"pair_img_feat": pair_img_output,
						"pair_img_label": pair_img_label,
						"pair_img_prob": pair_img_porb,
						}

				# for key,val in pc_feat_label.items():
				# 	print(key)
				# 	print(val.shape)

				# for key,val in pair_feat_label.items():
				# 	print(key)
				# 	print(val.shape)

				# for key,val in text_feat_label.items():
				# 	print(key)
				# 	print(val.shape)

				# input()	

				return pc_output, loss_pc, loss_dict, pair_feat_label, pc_feat_label, text_feat_label
			else:
				loss_pc, loss_dict, assignments = criterion(pc_output, targets, return_assignments=True)
				return pc_output, loss_pc, loss_dict, None, None, None


		else:
			eval_text_output_before_clip_header = self.eval_text_feats
			eval_text_output = self.eval_text_feats / self.eval_text_feats.norm(dim=1, keepdim=True)
			eval_text_feat_label = {"text_feat": eval_text_output[:self.eval_text_num, :],
								"text_label": self.eval_text_label}

			# classify point cloud querys
			pc_sem_cls_logits, pc_objectness_prob, pc_sem_cls_prob, pc_all_label = self.classify_pc(pc_output["pc_query_feat"], eval_text_output, self.eval_text_num)
			pc_output["outputs"]["sem_cls_logits"] = pc_sem_cls_logits
			pc_output["outputs"]["objectness_prob"] = pc_objectness_prob
			pc_output["outputs"]["sem_cls_prob"] = pc_sem_cls_prob
			return pc_output
	

def build_img_encoder(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, preprocess = clip.load(args.clip_model, device=device)
    #encoder, preprocess = clip.load("RN50", device=device)
    return encoder

def build_3detr(args, dataset_config):
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    pc_model = Model3DETR(
        args,
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
    )
    output_processor = BoxProcessor(dataset_config)
    
    # Build Image Branch
    img_model = build_img_encoder(args)
    
    model = DETR_3D_2D(args, pc_model=pc_model, img_model=img_model)

    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
        
    return model, output_processor
