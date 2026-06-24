"""torchvision detector ZOO registration."""

from cracks_yolo.zoo.torchvision.faster_rcnn import FasterRCNNModel
from cracks_yolo.zoo.torchvision.fcos import FCOSModel
from cracks_yolo.zoo.torchvision.mask_rcnn import MaskRCNNModel
from cracks_yolo.zoo.torchvision.retinanet import RetinaNetModel
from cracks_yolo.zoo.torchvision.ssd300 import SSD300Model
from cracks_yolo.zoo.torchvision.ssdlite320 import SSDlite320Model

ZOO: dict[str, type] = {
    "retinanet_r50": RetinaNetModel,
    "faster_rcnn_r50": FasterRCNNModel,
    "mask_rcnn_r50": MaskRCNNModel,
    "fcos_r50": FCOSModel,
    "ssd300_vgg16": SSD300Model,
    "ssdlite320_mobilenetv3": SSDlite320Model,
}
