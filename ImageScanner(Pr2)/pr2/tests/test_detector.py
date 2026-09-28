import unittest

from app import detector


class DetectorTests(unittest.TestCase):
    def test_load_model_returns_single_instance(self):
        model_a = detector.load_model()
        model_b = detector.load_model()
        self.assertIs(model_a, model_b)

    def test_empty_bytes_raise_detection_error(self):
        with self.assertRaises(detector.DetectionError):
            detector.detect(b"")

    def test_invalid_image_bytes_raise_detection_error(self):
        with self.assertRaises(detector.DetectionError):
            detector.detect(b"not-an-image")


if __name__ == "__main__":
    unittest.main()
