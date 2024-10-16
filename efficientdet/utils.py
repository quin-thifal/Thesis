import torch
import itertools
import numpy as np
import torch.nn as nn

# Bounding Box Transformation
class BBoxTransform(nn.Module):
    # Calculate center and dimensions of anchors
    def forward(self, anchors, regression):
        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 2] - anchors[..., 0]
        wa = anchors[..., 3] - anchors[..., 1]
        # Apply regression to get the final bounding box coordinates
        w = regression[..., 3].exp() * wa
        h = regression[..., 2].exp() * ha
        y_centers = regression[..., 0] * ha + y_centers_a
        x_centers = regression[..., 1] * wa + x_centers_a
        ymin = y_centers - h / 2.
        xmin = x_centers - w / 2.
        ymax = y_centers + h / 2.
        xmax = x_centers + w / 2.
        return torch.stack([xmin, ymin, xmax, ymax], dim=2)

# Box Clipping
class ClipBoxes(nn.Module):
    def __init__(self):
        super(ClipBoxes, self).__init__()
    # Get image dimensions
    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape
        # Clip box coordinates to be within image boundaries
        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)
        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)
        return boxes

# Anchor Generation
class Anchors(nn.Module):
    # Initialize anchor parameters
    def __init__(self, anchor_scale=4., pyramid_levels=None, **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels
        self.strides = kwargs.get('strides', [2 ** x for x in self.pyramid_levels])
        self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        self.ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.last_anchors = {}
        self.last_shape = None
    # Generate anchor boxes based on image shape
    def forward(self, image, dtype=torch.float32):
        image_shape = image.shape[2:]
        if image_shape == self.last_shape and image.device in self.last_anchors:
            return self.last_anchors[image.device]
        if self.last_shape is None or self.last_shape != image_shape:
            self.last_shape = image_shape
        # Set dtype
        if dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32
        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if image_shape[1] % stride != 0:
                    raise ValueError('Input size must be divided by the stride.')
                # Calculate anchor sizes and positions
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0
                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))
        anchor_boxes = np.vstack(boxes_all)
        anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(image.device)
        anchor_boxes = anchor_boxes.unsqueeze(0)
        self.last_anchors[image.device] = anchor_boxes
        return anchor_boxes