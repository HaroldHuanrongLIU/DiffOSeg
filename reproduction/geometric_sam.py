"""Continuous training views with native-grid supervision for single-slice SAM."""
import math

import torch
from torch.nn import functional as F

from reproduction import native_local


def random_grids(batch, generator):
    values = torch.rand((batch, 5), generator=generator, device=generator.device)
    center = 63.5 + 8 * (values[:, :2] - 0.5)
    scale = 0.75 + 0.5 * values[:, 2]
    angle = (values[:, 3] - 0.5) * math.pi / 2
    flip = torch.where(values[:, 4] < 0.5, -1.0, 1.0)
    matrix = native_local.matrix_for(scale, angle, flip)
    return native_local.grids(center, matrix, 128)


def native_predictions(model, image, guide, rotation, grid, back):
    def sample(value, coordinates):
        return F.grid_sample(value.float(), coordinates, padding_mode="border", align_corners=True)

    local_image = sample(image, grid)
    local_guide = sample(guide, grid)
    logits, raw = model(None, local_guide, local_image, rotation)
    # Keep existing single-slice guidance where the transformed view has no pixels.
    covered = back.abs().amax(-1)[:, None] <= 1
    return tuple(torch.where(covered, sample(prediction, back), guide) for prediction in (logits, raw))
