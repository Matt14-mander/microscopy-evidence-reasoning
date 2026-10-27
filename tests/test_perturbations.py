import unittest

import numpy as np
from PIL import Image

from src.data.schema import Region
from src.perturbations.transforms import apply_condition, control_box


class PerturbationTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (20, 16), "white")
        self.region = Region(0, (2, 2, 7, 6))

    def test_exact_mask_area_and_no_mutation(self):
        masked, meta = apply_condition([self.image], [self.region], {"kind": "evidence_mask"}, 7)
        pixels = np.array(masked[0])
        self.assertEqual(int((pixels[:, :, 0] == 0).sum()), 20)
        self.assertTrue(np.all(np.array(self.image) == 255))
        self.assertEqual(meta["regions"][0]["box"], [2, 2, 7, 6])

    def test_control_shape_disjoint_and_deterministic(self):
        for seed in range(30):
            box = control_box(self.image.size, self.region.box, seed)
            x0, y0, x1, y1 = box
            self.assertEqual((x1 - x0, y1 - y0), (5, 4))
            self.assertTrue(x1 <= 2 or x0 >= 7 or y1 <= 2 or y0 >= 6)
            self.assertEqual(box, control_box(self.image.size, self.region.box, seed))

    def test_no_room_and_missing_annotation_fail(self):
        with self.assertRaises(ValueError):
            control_box((10, 10), (0, 0, 10, 10), 0)
        with self.assertRaises(ValueError):
            apply_condition([self.image], [], {"kind": "evidence_mask"}, 0)

    def test_text_only_and_multiple_images(self):
        output, _ = apply_condition([self.image], [], {"kind": "text_only"}, 0)
        self.assertEqual(output, ())
        images, _ = apply_condition([self.image, self.image], [Region(1, (2, 2, 7, 6))], {"kind": "evidence_mask"}, 0)
        self.assertEqual(images[0].getpixel((3, 3)), (255, 255, 255))
        self.assertEqual(images[1].getpixel((3, 3)), (0, 0, 0))

    def test_blur_and_resolution_change_content_preserve_size(self):
        pattern = np.zeros((16, 20, 3), dtype=np.uint8)
        pattern[::2, ::2] = 255
        image = Image.fromarray(pattern)
        for condition in ({"kind": "blur", "radius": 2}, {"kind": "resolution", "scale": 0.5}):
            with self.subTest(condition=condition):
                output, _ = apply_condition([image], [], condition, 0)
                self.assertEqual(output[0].size, image.size)
                self.assertFalse(np.array_equal(np.array(output[0]), pattern))

    def test_invalid_parameters(self):
        for condition in ({"kind": "blur", "radius": 0}, {"kind": "resolution", "scale": 2},
                          {"kind": "unknown"}, {"kind": "baseline", "raduis": 2}):
            with self.subTest(condition=condition), self.assertRaises(ValueError):
                apply_condition([self.image], [], condition, 0)
