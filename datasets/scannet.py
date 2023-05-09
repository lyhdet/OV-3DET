# Copyright (c) Facebook, Inc. and its affiliates.


""" 
Modified from https://github.com/facebookresearch/votenet
Dataset for 3D object detection on SUN RGB-D (with support of vote supervision).

A sunrgbd oriented bounding box is parameterized by (cx,cy,cz), (l,w,h) -- (dx,dy,dz) in upright depth coord
(Z is up, Y is forward, X is right ward), heading angle (from +X rotating to -Y) and semantic class

Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Return heading class, heading residual, size class and size residual for 3D bounding boxes.
Oriented bounding box is parameterized by (cx,cy,cz), (l,w,h), heading_angle and semantic class label.
(cx,cy,cz) is in upright depth coordinate
(l,h,w) are *half length* of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

Author: Charles R. Qi
Date: 2019

"""
import os
import sys
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio  # to load .mat files for depth points
import cv2
import random

import utils.pc_util as pc_util
from utils.sunrgbd_pc_util import write_oriented_bbox
from utils.sunrgbd_utils import sunrgbd_object, SUNObject3d, compute_box_3d, draw_projected_box3d
from utils.random_cuboid import RandomCuboid
from utils.pc_util import shift_scale_points, scale_points
from utils.box_util import (
    flip_axis_to_camera_tensor,
    get_3d_box_batch_tensor,
    flip_axis_to_camera_np,
    get_3d_box_batch_np,
)


MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1
DATA_PATH_V1 = "" ## Replace with path to dataset
DATA_PATH_V2 = "" ## Not used in the codebase.


class ScannetDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 365
        self.num_angle_bin = 12
        self.max_num_obj = 64
        #self.open_class = [0,1,2,3,4,5,6,7,8,9]
        self.open_class = []
        #self.open_class_top_k = 10
        self.type2class = {"human": 0,
                            "sneakers": 1,
                            "chair": 2,
                            "hat": 3,
                            "lamp": 4,
                            "bottle": 5,
                            "cabinet/shelf": 6,
                            "cup": 7,
                            "car": 8,
                            "glasses": 9,
                            "picture/frame": 10,
                            "desk": 11,
                            "handbag": 12,
                            "street lights": 13,
                            "book": 14,
                            "plate": 15,
                            "helmet": 16,
                            "leather shoes": 17,
                            "pillow": 18,
                            "glove": 19,
                            "potted plant": 20,
                            "bracelet": 21,
                            "flower": 22,
                            "monitor": 23,
                            "storage box": 24,
                            "plants pot/vase": 25,
                            "bench": 26,
                            "wine glass": 27,
                            "boots": 28,
                            "dining table": 29,
                            "umbrella": 30,
                            "boat": 31,
                            "flag": 32,
                            "speaker": 33,
                            "trash bin/can": 34,
                            "stool": 35,
                            "backpack": 36,
                            "sofa": 37,
                            "belt": 38,
                            "carpet": 39,
                            "basket": 40,
                            "towel/napkin": 41,
                            "slippers": 42,
                            "bowl": 43,
                            "barrel/bucket": 44,
                            "coffee table": 45,
                            "suv": 46,
                            "toy": 47,
                            "tie": 48,
                            "bed": 49,
                            "traffic light": 50,
                            "pen/pencil": 51,
                            "microphone": 52,
                            "sandals": 53,
                            "canned": 54,
                            "necklace": 55,
                            "mirror": 56,
                            "faucet": 57,
                            "bicycle": 58,
                            "bread": 59,
                            "high heels": 60,
                            "ring": 61,
                            "van": 62,
                            "watch": 63,
                            "combine with bowl": 64,
                            "sink": 65,
                            "horse": 66,
                            "fish": 67,
                            "apple": 68,
                            "traffic sign": 69,
                            "camera": 70,
                            "candle": 71,
                            "stuffed animal": 72,
                            "cake": 73,
                            "motorbike/motorcycle": 74,
                            "wild bird": 75,
                            "laptop": 76,
                            "knife": 77,
                            "cellphone": 78,
                            "paddle": 79,
                            "truck": 80,
                            "cow": 81,
                            "power outlet": 82,
                            "clock": 83,
                            "drum": 84,
                            "fork": 85,
                            "bus": 86,
                            "hanger": 87,
                            "nightstand": 88,
                            "pot/pan": 89,
                            "sheep": 90,
                            "guitar": 91,
                            "traffic cone": 92,
                            "tea pot": 93,
                            "keyboard": 94,
                            "tripod": 95,
                            "hockey stick": 96,
                            "fan": 97,
                            "dog": 98,
                            "spoon": 99,
                            "blackboard/whiteboard": 100,
                            "balloon": 101,
                            "air conditioner": 102,
                            "cymbal": 103,
                            "mouse": 104,
                            "telephone": 105,
                            "pickup truck": 106,
                            "orange": 107,
                            "banana": 108,
                            "airplane": 109,
                            "luggage": 110,
                            "skis": 111,
                            "soccer": 112,
                            "trolley": 113,
                            "oven": 114,
                            "remote": 115,
                            "combine with glove": 116,
                            "paper towel": 117,
                            "refrigerator": 118,
                            "train": 119,
                            "tomato": 120,
                            "machinery vehicle": 121,
                            "tent": 122,
                            "shampoo/shower gel": 123,
                            "head phone": 124,
                            "lantern": 125,
                            "donut": 126,
                            "cleaning products": 127,
                            "sailboat": 128,
                            "tangerine": 129,
                            "pizza": 130,
                            "kite": 131,
                            "computer box": 132,
                            "elephant": 133,
                            "toiletries": 134,
                            "gas stove": 135,
                            "broccoli": 136,
                            "toilet": 137,
                            "stroller": 138,
                            "shovel": 139,
                            "baseball bat": 140,
                            "microwave": 141,
                            "skateboard": 142,
                            "surfboard": 143,
                            "surveillance camera": 144,
                            "gun": 145,
                            "Life saver": 146,
                            "cat": 147,
                            "lemon": 148,
                            "liquid soap": 149,
                            "zebra": 150,
                            "duck": 151,
                            "sports car": 152,
                            "giraffe": 153,
                            "pumpkin": 154,
                            "Accordion/keyboard/piano": 155,
                            "radiator": 156,
                            "converter": 157,
                            "tissue": 158,
                            "carrot": 159,
                            "washing machine": 160,
                            "vent": 161,
                            "cookies": 162,
                            "cutting/chopping board": 163,
                            "tennis racket": 164,
                            "candy": 165,
                            "skating and skiing shoes": 166,
                            "scissors": 167,
                            "folder": 168,
                            "baseball": 169,
                            "strawberry": 170,
                            "bow tie": 171,
                            "pigeon": 172,
                            "pepper": 173,
                            "coffee machine": 174,
                            "bathtub": 175,
                            "snowboard": 176,
                            "suitcase": 177,
                            "grapes": 178,
                            "ladder": 179,
                            "pear": 180,
                            "american football": 181,
                            "basketball": 182,
                            "potato": 183,
                            "paint brush": 184,
                            "printer": 185,
                            "billiards": 186,
                            "fire hydrant": 187,
                            "goose": 188,
                            "projector": 189,
                            "sausage": 190,
                            "fire extinguisher": 191,
                            "extension cord": 192,
                            "facial mask": 193,
                            "tennis ball": 194,
                            "chopsticks": 195,
                            "Electronic stove and gas st": 196,
                            "pie": 197,
                            "frisbee": 198,
                            "kettle": 199,
                            "hamburger": 200,
                            "golf club": 201,
                            "cucumber": 202,
                            "clutch": 203,
                            "blender": 204,
                            "tong": 205,
                            "slide": 206,
                            "hot dog": 207,
                            "toothbrush": 208,
                            "facial cleanser": 209,
                            "mango": 210,
                            "deer": 211,
                            "egg": 212,
                            "violin": 213,
                            "marker": 214,
                            "ship": 215,
                            "chicken": 216,
                            "onion": 217,
                            "ice cream": 218,
                            "tape": 219,
                            "wheelchair": 220,
                            "plum": 221,
                            "bar soap": 222,
                            "scale": 223,
                            "watermelon": 224,
                            "cabbage": 225,
                            "router/modem": 226,
                            "golf ball": 227,
                            "pine apple": 228,
                            "crane": 229,
                            "fire truck": 230,
                            "peach": 231,
                            "cello": 232,
                            "notepaper": 233,
                            "tricycle": 234,
                            "toaster": 235,
                            "helicopter": 236,
                            "green beans": 237,
                            "brush": 238,
                            "carriage": 239,
                            "cigar": 240,
                            "earphone": 241,
                            "penguin": 242,
                            "hurdle": 243,
                            "swing": 244,
                            "radio": 245,
                            "CD": 246,
                            "parking meter": 247,
                            "swan": 248,
                            "garlic": 249,
                            "french fries": 250,
                            "horn": 251,
                            "avocado": 252,
                            "saxophone": 253,
                            "trumpet": 254,
                            "sandwich": 255,
                            "cue": 256,
                            "kiwi fruit": 257,
                            "bear": 258,
                            "fishing rod": 259,
                            "cherry": 260,
                            "tablet": 261,
                            "green vegetables": 262,
                            "nuts": 263,
                            "corn": 264,
                            "key": 265,
                            "screwdriver": 266,
                            "globe": 267,
                            "broom": 268,
                            "pliers": 269,
                            "hammer": 270,
                            "volleyball": 271,
                            "eggplant": 272,
                            "trophy": 273,
                            "board eraser": 274,
                            "dates": 275,
                            "rice": 276,
                            "tape measure/ruler": 277,
                            "dumbbell": 278,
                            "hamimelon": 279,
                            "stapler": 280,
                            "camel": 281,
                            "lettuce": 282,
                            "goldfish": 283,
                            "meat balls": 284,
                            "medal": 285,
                            "toothpaste": 286,
                            "antelope": 287,
                            "shrimp": 288,
                            "rickshaw": 289,
                            "trombone": 290,
                            "pomegranate": 291,
                            "coconut": 292,
                            "jellyfish": 293,
                            "mushroom": 294,
                            "calculator": 295,
                            "treadmill": 296,
                            "butterfly": 297,
                            "egg tart": 298,
                            "cheese": 299,
                            "pomelo": 300,
                            "pig": 301,
                            "race car": 302,
                            "rice cooker": 303,
                            "tuba": 304,
                            "crosswalk sign": 305,
                            "papaya": 306,
                            "hair dryer": 307,
                            "green onion": 308,
                            "chips": 309,
                            "dolphin": 310,
                            "sushi": 311,
                            "urinal": 312,
                            "donkey": 313,
                            "electric drill": 314,
                            "spring rolls": 315,
                            "tortoise/turtle": 316,
                            "parrot": 317,
                            "flute": 318,
                            "measuring cup": 319,
                            "shark": 320,
                            "steak": 321,
                            "poker card": 322,
                            "binoculars": 323,
                            "llama": 324,
                            "radish": 325,
                            "noodles": 326,
                            "mop": 327,
                            "yak": 328,
                            "crab": 329,
                            "microscope": 330,
                            "barbell": 331,
                            "Bread/bun": 332,
                            "baozi": 333,
                            "lion": 334,
                            "red cabbage": 335,
                            "polar bear": 336,
                            "lighter": 337,
                            "mangosteen": 338,
                            "seal": 339,
                            "comb": 340,
                            "eraser": 341,
                            "pitaya": 342,
                            "scallop": 343,
                            "pencil case": 344,
                            "saw": 345,
                            "table tennis  paddle": 346,
                            "okra": 347,
                            "starfish": 348,
                            "monkey": 349,
                            "eagle": 350,
                            "durian": 351,
                            "rabbit": 352,
                            "game board": 353,
                            "french horn": 354,
                            "ambulance": 355,
                            "asparagus": 356,
                            "hoverboard": 357,
                            "pasta": 358,
                            "target": 359,
                            "hotair balloon": 360,
                            "chainsaw": 361,
                            "lobster": 362,
                            "iron": 363,
                            "flashlight": 364,}
        
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.type2onehotclass = {"human": 0,
                            "sneakers": 1,
                            "chair": 2,
                            "hat": 3,
                            "lamp": 4,
                            "bottle": 5,
                            "cabinet/shelf": 6,
                            "cup": 7,
                            "car": 8,
                            "glasses": 9,
                            "picture/frame": 10,
                            "desk": 11,
                            "handbag": 12,
                            "street lights": 13,
                            "book": 14,
                            "plate": 15,
                            "helmet": 16,
                            "leather shoes": 17,
                            "pillow": 18,
                            "glove": 19,
                            "potted plant": 20,
                            "bracelet": 21,
                            "flower": 22,
                            "monitor": 23,
                            "storage box": 24,
                            "plants pot/vase": 25,
                            "bench": 26,
                            "wine glass": 27,
                            "boots": 28,
                            "dining table": 29,
                            "umbrella": 30,
                            "boat": 31,
                            "flag": 32,
                            "speaker": 33,
                            "trash bin/can": 34,
                            "stool": 35,
                            "backpack": 36,
                            "sofa": 37,
                            "belt": 38,
                            "carpet": 39,
                            "basket": 40,
                            "towel/napkin": 41,
                            "slippers": 42,
                            "bowl": 43,
                            "barrel/bucket": 44,
                            "coffee table": 45,
                            "suv": 46,
                            "toy": 47,
                            "tie": 48,
                            "bed": 49,
                            "traffic light": 50,
                            "pen/pencil": 51,
                            "microphone": 52,
                            "sandals": 53,
                            "canned": 54,
                            "necklace": 55,
                            "mirror": 56,
                            "faucet": 57,
                            "bicycle": 58,
                            "bread": 59,
                            "high heels": 60,
                            "ring": 61,
                            "van": 62,
                            "watch": 63,
                            "combine with bowl": 64,
                            "sink": 65,
                            "horse": 66,
                            "fish": 67,
                            "apple": 68,
                            "traffic sign": 69,
                            "camera": 70,
                            "candle": 71,
                            "stuffed animal": 72,
                            "cake": 73,
                            "motorbike/motorcycle": 74,
                            "wild bird": 75,
                            "laptop": 76,
                            "knife": 77,
                            "cellphone": 78,
                            "paddle": 79,
                            "truck": 80,
                            "cow": 81,
                            "power outlet": 82,
                            "clock": 83,
                            "drum": 84,
                            "fork": 85,
                            "bus": 86,
                            "hanger": 87,
                            "nightstand": 88,
                            "pot/pan": 89,
                            "sheep": 90,
                            "guitar": 91,
                            "traffic cone": 92,
                            "tea pot": 93,
                            "keyboard": 94,
                            "tripod": 95,
                            "hockey stick": 96,
                            "fan": 97,
                            "dog": 98,
                            "spoon": 99,
                            "blackboard/whiteboard": 100,
                            "balloon": 101,
                            "air conditioner": 102,
                            "cymbal": 103,
                            "mouse": 104,
                            "telephone": 105,
                            "pickup truck": 106,
                            "orange": 107,
                            "banana": 108,
                            "airplane": 109,
                            "luggage": 110,
                            "skis": 111,
                            "soccer": 112,
                            "trolley": 113,
                            "oven": 114,
                            "remote": 115,
                            "combine with glove": 116,
                            "paper towel": 117,
                            "refrigerator": 118,
                            "train": 119,
                            "tomato": 120,
                            "machinery vehicle": 121,
                            "tent": 122,
                            "shampoo/shower gel": 123,
                            "head phone": 124,
                            "lantern": 125,
                            "donut": 126,
                            "cleaning products": 127,
                            "sailboat": 128,
                            "tangerine": 129,
                            "pizza": 130,
                            "kite": 131,
                            "computer box": 132,
                            "elephant": 133,
                            "toiletries": 134,
                            "gas stove": 135,
                            "broccoli": 136,
                            "toilet": 137,
                            "stroller": 138,
                            "shovel": 139,
                            "baseball bat": 140,
                            "microwave": 141,
                            "skateboard": 142,
                            "surfboard": 143,
                            "surveillance camera": 144,
                            "gun": 145,
                            "Life saver": 146,
                            "cat": 147,
                            "lemon": 148,
                            "liquid soap": 149,
                            "zebra": 150,
                            "duck": 151,
                            "sports car": 152,
                            "giraffe": 153,
                            "pumpkin": 154,
                            "Accordion/keyboard/piano": 155,
                            "radiator": 156,
                            "converter": 157,
                            "tissue": 158,
                            "carrot": 159,
                            "washing machine": 160,
                            "vent": 161,
                            "cookies": 162,
                            "cutting/chopping board": 163,
                            "tennis racket": 164,
                            "candy": 165,
                            "skating and skiing shoes": 166,
                            "scissors": 167,
                            "folder": 168,
                            "baseball": 169,
                            "strawberry": 170,
                            "bow tie": 171,
                            "pigeon": 172,
                            "pepper": 173,
                            "coffee machine": 174,
                            "bathtub": 175,
                            "snowboard": 176,
                            "suitcase": 177,
                            "grapes": 178,
                            "ladder": 179,
                            "pear": 180,
                            "american football": 181,
                            "basketball": 182,
                            "potato": 183,
                            "paint brush": 184,
                            "printer": 185,
                            "billiards": 186,
                            "fire hydrant": 187,
                            "goose": 188,
                            "projector": 189,
                            "sausage": 190,
                            "fire extinguisher": 191,
                            "extension cord": 192,
                            "facial mask": 193,
                            "tennis ball": 194,
                            "chopsticks": 195,
                            "Electronic stove and gas st": 196,
                            "pie": 197,
                            "frisbee": 198,
                            "kettle": 199,
                            "hamburger": 200,
                            "golf club": 201,
                            "cucumber": 202,
                            "clutch": 203,
                            "blender": 204,
                            "tong": 205,
                            "slide": 206,
                            "hot dog": 207,
                            "toothbrush": 208,
                            "facial cleanser": 209,
                            "mango": 210,
                            "deer": 211,
                            "egg": 212,
                            "violin": 213,
                            "marker": 214,
                            "ship": 215,
                            "chicken": 216,
                            "onion": 217,
                            "ice cream": 218,
                            "tape": 219,
                            "wheelchair": 220,
                            "plum": 221,
                            "bar soap": 222,
                            "scale": 223,
                            "watermelon": 224,
                            "cabbage": 225,
                            "router/modem": 226,
                            "golf ball": 227,
                            "pine apple": 228,
                            "crane": 229,
                            "fire truck": 230,
                            "peach": 231,
                            "cello": 232,
                            "notepaper": 233,
                            "tricycle": 234,
                            "toaster": 235,
                            "helicopter": 236,
                            "green beans": 237,
                            "brush": 238,
                            "carriage": 239,
                            "cigar": 240,
                            "earphone": 241,
                            "penguin": 242,
                            "hurdle": 243,
                            "swing": 244,
                            "radio": 245,
                            "CD": 246,
                            "parking meter": 247,
                            "swan": 248,
                            "garlic": 249,
                            "french fries": 250,
                            "horn": 251,
                            "avocado": 252,
                            "saxophone": 253,
                            "trumpet": 254,
                            "sandwich": 255,
                            "cue": 256,
                            "kiwi fruit": 257,
                            "bear": 258,
                            "fishing rod": 259,
                            "cherry": 260,
                            "tablet": 261,
                            "green vegetables": 262,
                            "nuts": 263,
                            "corn": 264,
                            "key": 265,
                            "screwdriver": 266,
                            "globe": 267,
                            "broom": 268,
                            "pliers": 269,
                            "hammer": 270,
                            "volleyball": 271,
                            "eggplant": 272,
                            "trophy": 273,
                            "board eraser": 274,
                            "dates": 275,
                            "rice": 276,
                            "tape measure/ruler": 277,
                            "dumbbell": 278,
                            "hamimelon": 279,
                            "stapler": 280,
                            "camel": 281,
                            "lettuce": 282,
                            "goldfish": 283,
                            "meat balls": 284,
                            "medal": 285,
                            "toothpaste": 286,
                            "antelope": 287,
                            "shrimp": 288,
                            "rickshaw": 289,
                            "trombone": 290,
                            "pomegranate": 291,
                            "coconut": 292,
                            "jellyfish": 293,
                            "mushroom": 294,
                            "calculator": 295,
                            "treadmill": 296,
                            "butterfly": 297,
                            "egg tart": 298,
                            "cheese": 299,
                            "pomelo": 300,
                            "pig": 301,
                            "race car": 302,
                            "rice cooker": 303,
                            "tuba": 304,
                            "crosswalk sign": 305,
                            "papaya": 306,
                            "hair dryer": 307,
                            "green onion": 308,
                            "chips": 309,
                            "dolphin": 310,
                            "sushi": 311,
                            "urinal": 312,
                            "donkey": 313,
                            "electric drill": 314,
                            "spring rolls": 315,
                            "tortoise/turtle": 316,
                            "parrot": 317,
                            "flute": 318,
                            "measuring cup": 319,
                            "shark": 320,
                            "steak": 321,
                            "poker card": 322,
                            "binoculars": 323,
                            "llama": 324,
                            "radish": 325,
                            "noodles": 326,
                            "mop": 327,
                            "yak": 328,
                            "crab": 329,
                            "microscope": 330,
                            "barbell": 331,
                            "Bread/bun": 332,
                            "baozi": 333,
                            "lion": 334,
                            "red cabbage": 335,
                            "polar bear": 336,
                            "lighter": 337,
                            "mangosteen": 338,
                            "seal": 339,
                            "comb": 340,
                            "eraser": 341,
                            "pitaya": 342,
                            "scallop": 343,
                            "pencil case": 344,
                            "saw": 345,
                            "table tennis  paddle": 346,
                            "okra": 347,
                            "starfish": 348,
                            "monkey": 349,
                            "eagle": 350,
                            "durian": 351,
                            "rabbit": 352,
                            "game board": 353,
                            "french horn": 354,
                            "ambulance": 355,
                            "asparagus": 356,
                            "hoverboard": 357,
                            "pasta": 358,
                            "target": 359,
                            "hotair balloon": 360,
                            "chainsaw": 361,
                            "lobster": 362,
                            "iron": 363,
                            "flashlight": 364,}

        self.eval_type2class = {
                            "toilet": 0,
                            "bed": 1,
                            "chair": 2,
                            "sofa": 3,
                            "dresser": 4,
                            "table": 5,
                            "cabinet": 6,
                            "bookshelf": 7,
                            "pillow": 8,
                            "sink": 9,
                            "bathtub": 10,
                            "refridgerator": 11,
                            "desk": 12,
                            "night stand": 13,
                            "counter": 14,
                            "door": 15,
                            "curtain": 16,
                            "box": 17,
                            "lamp": 18,
                            "bag": 19,}
        self.eval_class2type = {self.eval_type2class[t]: t for t in self.eval_type2class}

    def angle2class(self, angle):
        """Convert continuous angle to discrete class
        [optinal] also small regression number from
        class center angle to current angle.

        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        returns class [0,1,...,N-1] and a residual number such that
            class*(2pi/N) + number = angle
        """
        num_class = self.num_angle_bin
        angle = angle % (2 * np.pi)
        assert angle >= 0 and angle <= 2 * np.pi
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (
            class_id * angle_per_class + angle_per_class / 2
        )
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        """Inverse function to angle2class"""
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def class2angle_batch(self, pred_cls, residual, to_label_format=True):
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format:
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        return self.class2angle_batch(pred_cls, residual, to_label_format)

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

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


def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)

def get_color_label(xyz, intrinsic_image, rgb):
    height, width, _ = rgb.shape
    intrinsic_image = intrinsic_image[:3,:3]

    xyz_uniform = xyz/xyz[:,2:3]
    xyz_uniform = xyz_uniform.T

    uv = intrinsic_image @ xyz_uniform

    uv /= uv[2:3, :]
    uv = np.around(uv).astype(np.int)
    uv = uv.T

    uv[:, 0] = np.clip(uv[:, 0], 0, width-1)
    uv[:, 1] = np.clip(uv[:, 1], 0, height-1)

    uv_ind = uv[:, 1]*width + uv[:, 0]
    
    pc_rgb = np.take_along_axis(rgb.reshape([-1,3]), np.expand_dims(uv_ind, axis=1), axis=0)
    
    return pc_rgb


def save_img_with_bbox(img, bbox, filename, class2type):
	for ind in range(bbox.shape[0]):
		top_left = (int(bbox[ind, 0]), int(bbox[ind,1]))
		down_right = (int(bbox[ind, 2]), int(bbox[ind,3]))
		
		cv2.rectangle(img, top_left, down_right, (0,255,0), 2)
		cv2.putText(img, '%d %s'%(ind, class2type[bbox[ind,4]]), (max(int(bbox[ind,0]),15), max(int(bbox[ind,1]),15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
		cv2.imwrite(filename, img)

def img_norm(img):
	imagenet_std = np.array([0.26862954, 0.26130258, 0.27577711])
	imagenet_mean = np.array([0.48145466, 0.4578275, 0.40821073])
	img = ((img/255) - imagenet_mean) / imagenet_std
	return img

def flip_horizon(img, bbox):
	img = cv2.flip(img, 1)
	out_bbox = bbox.copy()
	out_bbox[:,2] = img.shape[1] - bbox[:,0]
	out_bbox[:,0] = img.shape[1] - bbox[:,2]
	return img, out_bbox
	
def flip_vertical(img, bbox):
	img = cv2.flip(img, 0)
	out_bbox = bbox.copy()
	out_bbox[:,3] = img.shape[0] - bbox[:,1]
	out_bbox[:,1] = img.shape[0] - bbox[:,3]
	return img, out_bbox

def random_rotate(img, bbox):
	img_width = img.shape[1]
	img_hight = img.shape[0]
	bbox_num = bbox.shape[0]
	
	center = (img_width/2, img_hight/2)
	angle = random.gauss(0, 10)
	rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
	
	corner = np.ones([4, bbox_num, 3])
	#left_top
	corner[0, :, :2] = bbox[:,:2]
		
	#right_down
	corner[1, :, :2] = bbox[:,2:4]
	
	#left_down
	left_down_ind = [0,3]
	corner[2, :, :2] = bbox[:,left_down_ind]
	
	#right_top
	right_top_ind = [2,1]
	corner[3, :, :2] = bbox[:,right_top_ind]
	
	rotated_corner = np.matmul(corner, rot_mat.T)
	
	out_bbox = np.zeros([bbox_num, 5])
	
	out_bbox[:, 0] = np.min(rotated_corner[: ,:, 0], axis=0)
	out_bbox[:, 1] = np.min(rotated_corner[: ,:, 1], axis=0)
	out_bbox[:, 2] = np.max(rotated_corner[: ,:, 0], axis=0)
	out_bbox[:, 3] = np.max(rotated_corner[: ,:, 1], axis=0)
	out_bbox[:, 4] = bbox[:,4]
	
	width_ind = [0, 2]
	heigh_ind = [1, 3]
	
	out_bbox[:, width_ind] = np.clip(out_bbox[:, width_ind], 0, img_width-1)
	out_bbox[:, heigh_ind] = np.clip(out_bbox[:, heigh_ind], 0, img_hight-1)
	
	out_img = cv2.warpAffine(img, rot_mat, (img_width, img_hight))
	
	return out_img, out_bbox

def scale_img_bbox(img, bbox, scale):
	scaled_width = int(img.shape[1] * scale) - 1
	scaled_hight = int(img.shape[0] * scale) - 1
	if scaled_width < 10:
		scaled_width = 10
	if scaled_hight < 10:
		scaled_hight = 10
	
	dsize =( scaled_width, scaled_hight )
	try:
		img = cv2.resize(img, dsize)
	except:
		print(dsize)
		print(img.shape)
		exit()
	
	bbox[:,:4] *= scale
	
	img_width = img.shape[1]
	img_hight = img.shape[0]	
	width_ind = [0, 2]
	heigh_ind = [1, 3]
	
	bbox[:, width_ind] = np.clip(bbox[:, width_ind], 0, img_width-1)
	bbox[:, heigh_ind] = np.clip(bbox[:, heigh_ind], 0, img_hight-1)
	return img, bbox

def random_scale(img, bbox):
	scale = random.uniform(1, 2)
	if random.random() < 0.5:
		scale = 1/scale
	
	img, bbox = scale_img_bbox(img, bbox, scale)
	
	return img, bbox
	
def img_det_aug(img, bbox, split, class2type):
	if split in ["train"]:
		# Random Horizontally Flip
		if random.random() < 0.5:
			img, bbox = flip_horizon(img, bbox)
		
		# Random Horizontally Flip
		if random.random() < 0.5:
			img, bbox = flip_vertical(img, bbox)
		
		# Random Rotate
		if random.random() < 0.5:
			img, bbox = random_rotate(img, bbox)
		#save_img_with_bbox(img.copy(), bbox, "after_rotate.jpg", class2type)
		
		# Random Scale
		if random.random() < 0.5:
			img, bbox = random_scale(img, bbox)
		
	# Norm is for both training and testing		
	img = img_norm(img)
	
	return img, bbox
	
def img_preprocess(img):
    # clip normalize
    def resize(img, size):
        img_h, img_w, _ = img.shape
        if img_h > img_w:
            new_w = int(img_w * size / img_h)
            new_h = size
        else:
            new_w = size
            new_h = int(img_h * size / img_w)

        dsize = (new_w, new_h)
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)
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
    	rst_img = np.zeros([dim[0], dim[1], 3], dtype=img.dtype)
    	h, w, _ = img.shape

    	top_left = (np.array(dim)/2 - np.array([h,w])/2).astype(np.int)
    	down_right = (top_left + np.array([h,w])).astype(np.int)

    	rst_img[top_left[0]:down_right[0], top_left[1]:down_right[1], :] = img.copy()
    	return rst_img
    	

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = resize(img, 224)
    img = padding(img, [224,224])
    #cv2.imwrite("image_cv2.jpg", img)
    
    img = img/255
    img_mean = np.array([0.48145466, 0.4578275, 0.40821073])
    img_std = np.array([0.26862954, 0.26130258, 0.27577711])
    img = img - img_mean
    img = img / img_std
    
    return img



class ScannetDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        split_set="train",
        root_dir=None,
        num_points=20000,
        use_color=False,
        use_height=False,
        use_v1=True,
        augment=False,
        use_random_cuboid=False,
        random_cuboid_min_points=30000,
    ):
        assert num_points <= 50000
        assert split_set in ["train", "val", "trainval"]
        
        self.dataset_config = dataset_config
        self.use_v1 = use_v1
        self.split_set = split_set

        if root_dir is None:
            root_dir = DATA_PATH_V1 if use_v1 else DATA_PATH_V2

        self.data_path = root_dir + "_%s" % (split_set)
        
        if split_set in ["train"]:
            self.scan_names = sorted(
                list(
                    set([os.path.basename(x)[0:19] for x in os.listdir(self.data_path) if ("intrinsic" not in x) and ("pc" in x)])
                )
            )

            # self.scan_names = self.scan_names[0:len(self.scan_names):150]
                        
            
        elif split_set in ["val"]:
            self.scan_names = sorted(
                list(
                    set([os.path.basename(x)[0:19] for x in os.listdir(self.data_path) if ("intrinsic" not in x) and ("pc" in x)])
                )
            )
            # self.scan_names = self.scan_names[0:len(self.scan_names):50]
        
        elif split_set in ["trainval"]:
            # combine names from both
            sub_splits = ["train", "val"]
            all_paths = []
            for sub_split in sub_splits:
                data_path = self.data_path.replace("trainval", sub_split)
                basenames = sorted(
                    list(set([os.path.basename(x)[0:6] for x in os.listdir(data_path)]))
                )
                basenames = [os.path.join(data_path, x) for x in basenames]
                all_paths.extend(basenames)
            all_paths.sort()
            self.scan_names = all_paths

        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(
            min_points=random_cuboid_min_points,
            aspect=0.75,
            min_crop=0.75,
            max_crop=1.0,
        )
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        self.max_num_obj = 64

    def __len__(self):
        return len(self.scan_names)
            
    def __getitem__(self, idx):
        while True:
            scan_name = self.scan_names[idx]
            if scan_name.startswith("/"):
                scan_path = scan_name
            else:
                scan_path = os.path.join(self.data_path, scan_name)

            if not os.path.exists(scan_path + "_bbox.npy"):
                idx = random.randint(0, len(self.scan_names)-1)
                continue

            bboxes = np.load(scan_path + "_bbox.npy")
            
            if bboxes.shape[0] <= 0 and self.split_set in ["train"]:
                idx = random.randint(0, len(self.scan_names)-1)
            else:
                break
        
        point_cloud = np.load(scan_path + "_pc.npy").astype(np.float32)  # Nx6
        bboxes = np.load(scan_path + "_bbox.npy")  # K,8
        pose = np.linalg.inv(load_matrix_from_txt(scan_path + "_pose.txt"))
        image_intrinsic = load_matrix_from_txt(scan_path[:-7] + "_image_intrinsic.txt")

        # validate map point to image
        # img = cv2.imread(scan_path + ".jpg")
        # cv2.imwrite("cur_img.png", img)
        # padding = np.ones([point_cloud.shape[0], 1])
        # point_cloud = np.concatenate([point_cloud[:,:3], padding], axis=1)
        # point_cloud = (np.linalg.inv(pose) @ point_cloud.T)[:3,:].T
        # rgb = get_color_label(point_cloud, image_intrinsic, img)
        # point_cloud = np.concatenate([point_cloud, rgb], axis=1)
        # np.savetxt("cur_pc.txt", point_cloud, fmt="%.3f")
        # exit()

        bbox_num = bboxes.shape[0]
        if bbox_num > 64:
            bboxes=bboxes[:64,:]
            bbox_num = bboxes.shape[0]

        # bboxes[:,-1] = 17
        # invalid_cls_idx = np.where(bboxes[:,-1]>19)[0]
        # bboxes[invalid_cls_idx,-1] = 0

        bboxes_ori=np.zeros([self.max_num_obj, 8])
        bboxes_ori[:bbox_num, :] = bboxes.copy()

        
        img = cv2.imread(scan_path + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img_norm(img)

        image = np.zeros([968,1296,3], dtype=np.float32)
        img_width = img.shape[1]
        img_height = img.shape[0]
        image_size = np.array([img_width, img_height])
        image[:img_height, :img_width, :3] = img
        

        # preprocess point cloud data
        if not self.use_color:
            point_cloud = point_cloud[:, 0:3]
        else:
            assert point_cloud.shape[1] == 6
            point_cloud = point_cloud[:, 0:6]
            point_cloud[:, 3:] = point_cloud[:, 3:] - MEAN_COLOR_RGB

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate(
                [point_cloud, np.expand_dims(height, 1)], 1
            )  # (N,4) or (N,7)

        # print(bboxes_ori)
        # exit()



        # ------------------------------- DATA AUGMENTATION ------------------------------
        Aug_Flip = 0
        Aug_Rot = 0
        Aug_Scale = 1
        Aug_Rot_Ang = 0
        Aug_Rot_Mat = 0

        if self.augment:
            if np.random.random() > 0.5:
                #Aug Flip flag
                Aug_Flip = 1

                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                bboxes[:, 0] = -1 * bboxes[:, 0]
                bboxes[:, 6] = np.pi - bboxes[:, 6]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
            rot_mat = pc_util.rotz(rot_angle)
            
            # Aug Rot flag
            Aug_Rot_Ang = rot_angle
            Aug_Rot_Mat = rot_mat

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 0:3] = np.dot(bboxes[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 6] -= rot_angle

            # Augment RGB color
            if self.use_color:
                rgb_color = point_cloud[:, 3:6] + MEAN_COLOR_RGB
                rgb_color *= (
                    1 + 0.4 * np.random.random(3) - 0.2
                )  # brightness change for each channel
                rgb_color += (
                    0.1 * np.random.random(3) - 0.05
                )  # color shift for each channel
                rgb_color += np.expand_dims(
                    (0.05 * np.random.random(point_cloud.shape[0]) - 0.025), -1
                )  # jittering on each pixel
                rgb_color = np.clip(rgb_color, 0, 1)
                # randomly drop out 30% of the points' colors
                rgb_color *= np.expand_dims(
                    np.random.random(point_cloud.shape[0]) > 0.3, -1
                )
                point_cloud[:, 3:6] = rgb_color - MEAN_COLOR_RGB

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random() * 0.3 + 0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            point_cloud[:, 0:3] *= scale_ratio
            bboxes[:, 0:3] *= scale_ratio
            bboxes[:, 3:6] *= scale_ratio

            # Aug Scale flag
            Aug_Scale = scale_ratio

            if self.use_height:
                point_cloud[:, -1] *= scale_ratio[0, 0]

            if self.use_random_cuboid:
                point_cloud, bboxes, _ = self.random_cuboid_augmentor(
                    point_cloud, bboxes
                )

        Aug_param = {"Aug_Flip": Aug_Flip,
                    "Aug_Scale": Aug_Scale,
                    "Aug_Rot_Ang": Aug_Rot_Ang,
                    "Aug_Rot_Mat": Aug_Rot_Mat,}
        # restore
        '''
        bboxes_after=bboxes.copy()
        bboxes_after[:,3:6] *= 2
        bboxes_after[:, 6] *= -1
        np.savetxt("pc_after_aug.txt", point_cloud, fmt="%.3f")
        write_oriented_bbox(bboxes_after[:,:7], "gt_after_aug.ply")


        print("\n\n")
        print(bboxes[0,:])
        bboxes_return=bboxes.copy()
        
        bboxes_return[:,:3] /= Aug_Scale
        bboxes_return[:,3:6] /= Aug_Scale
        print(bboxes_return[0,:])

        bboxes_return[:, 6] += Aug_Rot_Ang
        print(bboxes_return[0,:])
        bboxes_return[:, 0:3] = np.dot(bboxes_return[:, 0:3], Aug_Rot_Mat)
        print(bboxes_return[0,:])

        if Aug_Flip == 1:
            bboxes_return[:, 6] = np.pi - bboxes_return[:, 6]
            bboxes_return[:, 0] = -1 * bboxes_return[:, 0]
            print("here")

        print(bboxes_return[0,:])


        bboxes_return[:,3:6] *= 2
        bboxes_return[:, 6] *= -1
        write_oriented_bbox(bboxes_return[:,:7], "bboxes_return.ply")

        exit()
        '''
        

        # ------------------------------- LABELS ------------------------------
        angle_classes = np.zeros((self.max_num_obj,), dtype=np.float32)
        angle_residuals = np.zeros((self.max_num_obj,), dtype=np.float32)
        raw_angles = np.zeros((self.max_num_obj,), dtype=np.float32)
        raw_sizes = np.zeros((self.max_num_obj, 3), dtype=np.float32)
        label_mask = np.zeros((self.max_num_obj))
        label_mask[0 : bboxes.shape[0]] = 1
        max_bboxes = np.zeros((self.max_num_obj, 8))
        max_bboxes[0 : bboxes.shape[0], :] = bboxes

        target_bboxes_mask = label_mask
        target_bboxes = np.zeros((self.max_num_obj, 6))

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            raw_angles[i] = bbox[6] % 2 * np.pi
            box3d_size = bbox[3:6] * 2
            raw_sizes[i, :] = box3d_size
            angle_class, angle_residual = self.dataset_config.angle2class(bbox[6])
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            corners_3d = self.dataset_config.my_compute_box_3d(
                bbox[0:3], bbox[3:6], bbox[6]
            )
            # compute axis aligned box
            xmin = np.min(corners_3d[:, 0])
            ymin = np.min(corners_3d[:, 1])
            zmin = np.min(corners_3d[:, 2])
            xmax = np.max(corners_3d[:, 0])
            ymax = np.max(corners_3d[:, 1])
            zmax = np.max(corners_3d[:, 2])
            target_bbox = np.array(
                [
                    (xmin + xmax) / 2,
                    (ymin + ymax) / 2,
                    (zmin + zmax) / 2,
                    xmax - xmin,
                    ymax - ymin,
                    zmax - zmin,
                ]
            )
            target_bboxes[i, :] = target_bbox

        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True
        )

        point_cloud_dims_min = point_cloud.min(axis=0)
        point_cloud_dims_max = point_cloud.max(axis=0)

        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]

        # re-encode angles to be consistent with VoteNet eval
        angle_classes = angle_classes.astype(np.int64)
        angle_residuals = angle_residuals.astype(np.float32)
        raw_angles = self.dataset_config.class2angle_batch(
            angle_classes, angle_residuals
        )

        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(
            np.float32
        )
        target_bboxes_semcls = np.zeros((self.max_num_obj))
        target_bboxes_semcls[0 : bboxes.shape[0]] = bboxes[:, -1]  # from 0 to 9
        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["gt_angle_class_label"] = angle_classes
        ret_dict["gt_angle_residual_label"] = angle_residuals
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max

        ret_dict["image"] = image
        #ret_dict["image_size"] = image_size
        ret_dict["calib"] = {"calib_K": image_intrinsic, "calib_Rtilt": pose, "image_size": image_size}
        ret_dict["bboxes_ori"] = bboxes_ori
        ret_dict["bbox_num"] = bbox_num
        ret_dict["Aug_param"] = Aug_param
        
        return ret_dict
