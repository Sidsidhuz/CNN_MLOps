from __future__ import annotations

import unittest

from app.main import app, get_predictor


class SmokeTests(unittest.TestCase):
    def test_home_route_is_registered(self) -> None:
        self.assertTrue(any(route.path == "/" for route in app.routes))

    def test_predictor_is_available(self) -> None:
        predictor = get_predictor()
        self.assertIsInstance(predictor.available_crops(), list)


if __name__ == "__main__":
    unittest.main()