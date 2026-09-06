import unittest

import numpy as np

from vision_roi import crop_recognition_input, camera_coordinates


class VisionRoiTest(unittest.TestCase):
    def test_dashboard_excludes_both_outside_regions(self):
        # 320x240 camera -> 176x163 dashboard: central 259x240 camera pixels.
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :30] = 255
        frame[:, 289:] = 255
        frame[100, 100] = (1, 2, 3)
        cropped, roi = crop_recognition_input(frame, "30,0,259,240")
        self.assertEqual(cropped.shape, (240, 259, 3))
        self.assertFalse(np.any(cropped == 255))
        self.assertEqual(cropped[100, 70].tolist(), [1, 2, 3])
        self.assertFalse(np.shares_memory(frame, cropped))
        self.assertEqual(roi, (30, 0, 259, 240))

    def test_boxes_return_to_camera_coordinates_without_mutating_cache(self):
        cached = {"face": {"bbox": {"x": 4, "y": 6, "width": 20, "height": 30}}}
        response = camera_coordinates(cached, (30, 12, 259, 210))
        self.assertEqual(response["face"]["bbox"]["x"], 34)
        self.assertEqual(response["face"]["bbox"]["y"], 18)
        self.assertEqual(cached["face"]["bbox"]["x"], 4)

    def test_invalid_roi_is_rejected_not_replaced_by_full_frame(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for roi in ("", "1,2,3", "-1,0,100,100", "0,0,0,240", "0,0,321,240", "a,b,c,d"):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                crop_recognition_input(frame, roi)

    def test_legacy_client_and_no_face(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        image, roi = crop_recognition_input(frame, None)
        self.assertIs(image, frame)
        self.assertIsNone(roi)
        self.assertEqual(camera_coordinates({"face": None}, (30, 0, 259, 240)), {"face": None})


if __name__ == "__main__":
    unittest.main()
