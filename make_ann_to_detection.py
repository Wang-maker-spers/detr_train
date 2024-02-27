
import json
import torchvision

from util.misc import is_dist_avail_and_initialized

def make_ann_to_detection():
    new_name = "coco_detection.json"
    new_context = {}
    path = "../data/coco/annotations/coco_dete_1.json"
    with open(path) as f:
        context = json.load(f)
        new_context["info"] = context["info"]
        new_context["images"] = context["images"]
        new_context["categories"] = context["categories"]
        new_context["annotations"] = []
        for i in context["annotations"]:
            ann_dict = {}
            ann_dict["area"] = i["area"]
            ann_dict["iscrowd"] = i["iscrowd"]
            ann_dict["image_id"] = i["image_id"]
            ann_dict["bbox"] = i["bbox"]
            ann_dict["category_id"] = i["category_id"]
            ann_dict["id"] = i["id"]
            new_context["annotations"].append(ann_dict)
            
    with open(new_name, 'w') as file:
        json.dump(new_context, file)


def coco_detection_to_val():
    new_name = "coco_detection_val.json"
    new_context = {}
    path = "../data/coco/annotations/coco_detection.json"
    with open(path) as f:
        context = json.load(f)
        new_context["info"] = context["info"]
        new_context["images"] = context["images"][:500]
        new_context["categories"] = context["categories"]
        new_context["annotations"] =  context["annotations"]

    with open(new_name, 'w') as file:
        json.dump(new_context, file)


import torch
import torch.nn.functional as F
import torchvision
from torch import nn

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


# backbone = getattr(torchvision.models, 'resnet50')(replace_stride_with_dilation=[False, False, False],pretrained=True, norm_layer=FrozenBatchNorm2d)
# for name, parameter in backbone.named_parameters():
#     print(name,parameter)


def test_list_change(a):
    a[0]+=10
    

def to_fugai_cocojson():
    new_name = "coco_detection0.json"
    for i in range(10):
        new_context = {}
        new_context["test"] = i
        with open(new_name, 'w') as file:
            json.dump(new_context, file)

from models.matcher import HungarianMatcher

hm = HungarianMatcher(1,5,2)

outputs = {"pred_logits":torch.rand((1,100,92)),"pred_boxes":torch.rand((1,100,4))}

targets_dict = {
    'boxes': torch.tensor([[0.8301, 0.5701, 0.3337, 0.7882],
        [0.4096, 0.5024, 0.5822, 0.9951],
        [0.0741, 0.5300, 0.1466, 0.5304],
        [0.9546, 0.8439, 0.0908, 0.3121]]),
 'labels': torch.tensor([62, 62, 62, 67]),
 'image_id': torch.tensor([228144]),
 'area': torch.tensor([113670.9453, 203102.0312,  56520.0664,  21672.1914]),
 'iscrowd': torch.tensor([0, 0, 0, 0]),
 'orig_size': torch.tensor([426, 640]),
 'size': torch.tensor([ 768, 1153])}
# print(targets_dict["boxes"].shape[0])

targets = (targets_dict,)

# i["boxes"].add_(1)  for i in targets

# print(targets)
a = torch.ones((4,4))
b = torch.ones((4,4))+0.5

b[:,:2] = b[:,:2] + (a[:,:2] - b[:,:2])/100


def the_number_of_bbox():
    path = "out/coco_detection.json"
    pic_id = []
    with open(path) as f:
        context = json.load(f)
    for i in context["images"]:
        pic_id.append(i["id"])
    the_number = 0
    for i in context["annotations"]:
        if i["image_id"] in pic_id:
            the_number+=1
        
    print("you xiao de bbox de shu liang :",the_number)


the_number_of_bbox()
        
        
  

























