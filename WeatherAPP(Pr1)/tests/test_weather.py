import unittest
from unittest.mock import patch

from app.weather import WeatherError, find_city, get_current_weather


class WeatherLogicTests(unittest.TestCase):
    @patch("app.weather.requests.get")
    def test_find_city_returns_coordinates_for_known_city(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "results": [
                {
                    "name": "Kyiv",
                    "latitude": 50.45,
                    "longitude": 30.52,
                }
            ]
        }

        result = find_city("Kyiv")

        self.assertEqual(result["city"], "Kyiv")
        self.assertAlmostEqual(result["latitude"], 50.45)
        self.assertAlmostEqual(result["longitude"], 30.52)

    @patch("app.weather.requests.get")
    def test_get_current_weather_raises_for_unknown_city(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"error": "not found"}

        with self.assertRaises(WeatherError) as ctx:
            get_current_weather("Unknown City")

        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
