import unittest

import torch

from reproduction.prompt_preference import boxes_from_logits, image_input


class PromptGeometryTests(unittest.TestCase):
    def test_box_uses_the_ct_guide_at_both_resolutions(self):
        guide = torch.full((2, 4, 128, 128), -1.0)
        guide[0, :, 20:31, 40:51] = 1
        native = torch.tensor([[38.0, 18.0, 52.0, 32.0],
                               [46.0, 46.0, 82.0, 82.0]])
        for resolution in (512, 1024):
            expected = (native + 0.5) * (resolution / 128) - 0.5
            torch.testing.assert_close(
                boxes_from_logits(guide, False, resolution), expected
            )

    def test_normalization_remains_per_image(self):
        image = torch.arange(128 * 128).reshape(1, 1, 128, 128).float() % 256
        distractor = torch.zeros_like(image)
        for resolution in (512, 1024):
            for rotation in (0, 1):
                alone = image_input(image, rotation, resolution)
                batch = image_input(torch.cat((image, distractor)), rotation, resolution)
                self.assertEqual(tuple(alone.shape), (1, 3, resolution, resolution))
                torch.testing.assert_close(alone, batch[:1])
