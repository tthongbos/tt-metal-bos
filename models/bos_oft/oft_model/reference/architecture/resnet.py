import argparse
import ast
import io
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from PIL import Image


def _load_pretrained(model, pretrained):
    model_dict = model.state_dict()
    # Convert pretrained weights to the model's dtype
    dtype = next(model.parameters()).dtype
    pretrained = {k: v.to(dtype) if v.is_floating_point() else v for k, v in pretrained.items() if k in model_dict}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1, dtype=torch.float32):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        dtype=dtype,
    )


def conv1x1(in_planes, out_planes, stride=1, dtype=torch.float32):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        dtype=dtype,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dtype=torch.float32,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dtype=dtype)
        self.bn1 = nn.GroupNorm(16, planes, dtype=dtype)  # self.bn1=nn.BatchNorm2d(planes,dtype=dtype)
        self.conv2 = conv3x3(planes, planes, stride=1, dtype=dtype)
        self.bn2 = nn.GroupNorm(16, planes, dtype=dtype)  # self.bn2=nn.BatchNorm2d(planes,dtype=dtype)
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride, dtype=dtype),
                nn.GroupNorm(16, planes),  # nn.BatchNorm2d(planes, dtype=dtype)
            )
        else:
            self.downsample = None
        self.to(dtype)

    def forward(self, x, *, return_intermediates=False):
        outs = {} if return_intermediates else None
        identity = x

        out = self.conv1(x)
        if return_intermediates:
            outs["conv1"] = out

        out = self.bn1(out)
        if return_intermediates:
            outs["grnorm1"] = out
            outs["grnorm"] = out

        out = F.relu(out, inplace=not return_intermediates)
        if return_intermediates:
            outs["relu1"] = out

        out = self.conv2(out)
        if return_intermediates:
            outs["conv2"] = out

        out = self.bn2(out)
        if return_intermediates:
            outs["grnorm2"] = out

        if self.downsample is not None:
            identity = self.downsample(x)
            if return_intermediates:
                outs["downsample"] = identity

        out = out + identity
        if return_intermediates:
            outs["add"] = out

        out = F.relu(out, inplace=not return_intermediates)
        if return_intermediates:
            outs["relu2"] = out
            outs["out"] = out
            return out, outs

        return out


class ResNetLayer(nn.Module):
    def __init__(self, block, layers, dtype=torch.float32, num_classes=1000):
        super(ResNetLayer, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, dtype=dtype)
        # out_size = (input_size - kernel_size + 2*padding) // stride + 1
        self.bn1 = nn.GroupNorm(16, self.inplanes, dtype=dtype)
        self.dtype = dtype
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dtype=dtype)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dtype=dtype)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dtype=dtype)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dtype=dtype)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes, dtype=dtype)
        self.to(dtype)

    def _make_layer(self, block, planes, blocks, stride=1, dtype=torch.float32):
        layers = [block(self.inplanes, planes, stride, dtype=dtype)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dtype=dtype))

        return nn.Sequential(*layers)

    @staticmethod
    def _run_layer_with_intermediates(layer, x):
        layer_outs = {}
        for block_index, block in enumerate(layer, start=1):
            x, block_outs = block(x, return_intermediates=True)
            layer_outs[f"block{block_index}"] = block_outs
        return x, layer_outs

    def forward_feature_pyramid(self, x):
        outs = {}
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        x = self.conv1(x)

        outs["stem_conv"] = x
        x = self.bn1(x)
        outs["stem_bn"] = x
        x = F.relu(x, inplace=False)
        outs["stem_relu"] = x
        x = self.maxpool(x)
        outs["stem_pool"] = x

        x, layer1_outs = self._run_layer_with_intermediates(self.layer1, x)
        outs["layer1"] = x
        outs["layer1_blocks"] = layer1_outs

        feats8, layer2_outs = self._run_layer_with_intermediates(self.layer2, x)
        outs["layer2"] = feats8
        outs["layer2_blocks"] = layer2_outs

        feats16, layer3_outs = self._run_layer_with_intermediates(self.layer3, feats8)
        outs["layer3"] = feats16
        outs["layer3_blocks"] = layer3_outs

        feats32, layer4_outs = self._run_layer_with_intermediates(self.layer4, feats16)
        outs["layer4"] = feats32
        outs["layer4_blocks"] = layer4_outs
        return feats8, feats16, feats32, outs

    def forward_features(self, x):
        return self.forward_feature_pyramid(x)[2]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(pretrained=False, dtype=torch.float32, **kwargs):
    model = ResNetLayer(BasicBlock, [2, 2, 2, 2], dtype=dtype, **kwargs)
    if pretrained:
        _load_pretrained(model, model_zoo.load_url(model_urls["resnet18"]))
    return model


def resnet34(pretrained=False, dtype=torch.float32, **kwargs):
    model = ResNetLayer(BasicBlock, [3, 4, 6, 3], dtype=dtype, **kwargs)
    if pretrained:
        _load_pretrained(model, model_zoo.load_url(model_urls["resnet34"]))
    return model


#######################################
##                                    #
##        TEST CODE BELOW             #
##                                    #
#######################################


def _resize_short_side(image, short_side=256):
    from PIL import Image

    width, height = image.size
    if width < height:
        new_width = short_side
        new_height = round(height * short_side / width)
    else:
        new_height = short_side
        new_width = round(width * short_side / height)

    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    return image.resize((new_width, new_height), resample=resample)


def _center_crop(image, crop_size=224):
    width, height = image.size
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))


def _is_url(value):
    return urlparse(str(value)).scheme in {"http", "https", "file"}


def _open_image(image_source):
    parsed = urlparse(str(image_source))
    if parsed.scheme in {"http", "https"}:
        request = Request(str(image_source), headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")

    if parsed.scheme == "file":
        return Image.open(unquote(parsed.path)).convert("RGB")

    return Image.open(image_source).convert("RGB")


def _preprocess_image(image_source, dtype=torch.float32):
    image = _open_image(image_source)
    image = _resize_short_side(image)
    image = _center_crop(image)

    image_tensor = torch.tensor(list(image.getdata()), dtype=dtype).view(224, 224, 3)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=dtype).view(1, 3, 1, 1)
    return (image_tensor - mean) / std


def _load_imagenet_labels(labels_path):
    labels_path = Path(labels_path)
    labels = ast.literal_eval(labels_path.read_text())
    return {int(index): name for index, name in labels.items()}


def _predict_image(image_path, labels_path, pretrained=False, topk=5):
    torch.manual_seed(0)
    labels = _load_imagenet_labels(labels_path)

    model = resnet18(pretrained=pretrained, dtype=torch.float16)
    model.eval()

    x = _preprocess_image(image_path, dtype=torch.float16)
    with torch.no_grad():
        logits = model(x)
        probabilities = F.softmax(logits, dim=1)
        topk = max(1, min(topk, logits.shape[1]))
        top_probs, top_indices = torch.topk(probabilities, k=topk, dim=1)

    assert logits.shape == (1, 1000), f"Expected output shape (1, 1000), got {tuple(logits.shape)}"
    assert not torch.isnan(logits).any(), "Output has NaN values"

    print("ResNet image predict test passed")
    print(f"image source: {image_path}")
    print(f"input shape:  {tuple(x.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"pretrained:   {pretrained}")
    if not pretrained:
        print("note: pretrained=False, so class predictions only test the pipeline and are not meaningful.")

    print("top predictions:")
    for rank, (class_index, probability) in enumerate(zip(top_indices[0], top_probs[0]), start=1):
        class_index = class_index.item()
        class_name = labels.get(class_index, "unknown")
        print(f"{rank}. class_index={class_index}, prob={probability.item():.6f}, label={class_name}")


def _resnet_test():
    parser = argparse.ArgumentParser(description="Run ResNet image prediction test.")
    parser.add_argument("--image", default=None, help="Path, file:// URL, or http(s) URL to an RGB image.")
    parser.add_argument("--labels", default=None, help="Path to ImageNet class labels.")
    parser.add_argument("--topk", type=int, default=5, help="Number of predictions to print.")
    parser.add_argument("--pretrained", action="store_true", help="Download/use pretrained ImageNet weights.")
    args = parser.parse_args()

    _predict_image(
        image_path=args.image,
        labels_path=args.labels,
        pretrained=args.pretrained,
        topk=args.topk,
    )


if __name__ == "__main__":
    _resnet_test()
