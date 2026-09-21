"""Модуль інтеграції із зовнішнім API погоди.

Тут зосереджена вся робота з HTTP: адреси сервісів, параметри запиту,
timeout, перевірка статусів і розбір JSON. Веб-рівень отримує готовий
результат або зрозумілу помилку і нічого не знає про requests.
"""

from __future__ import annotations

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 10


class WeatherError(Exception):
    """Помилка отримання погоди з HTTP-статусом для веб-рівня."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _extract_error_message(payload):
    if not isinstance(payload, dict):
        return ""
    for key in ("reason", "message", "error"):
        value = payload.get(key)
        if value:
            if isinstance(value, str):
                return value
            return str(value)
    return ""


def _request_json(url: str, params: dict, *, timeout: int = REQUEST_TIMEOUT_SECONDS):
    """Виконати GET-запит і повернути JSON або згенерувати WeatherError."""
    try:
        response = requests.get(url, params=params, timeout=timeout)
    except requests.Timeout as exc:
        raise WeatherError("Сервіс погоди не відповідає вчасно. Спробуйте ще раз пізніше.", 504) from exc
    except requests.RequestException as exc:
        raise WeatherError("Сервіс погоди тимчасово недоступний.", 503) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise WeatherError("Сервіс погоди повернув некоректну відповідь.", 502) from exc

    if 400 <= response.status_code < 500:
        detail = _extract_error_message(payload)
        if response.status_code == 404:
            raise WeatherError("Місто не знайдено. Перевірте назву." , 404)
        if detail:
            raise WeatherError(f"Некоректний запит: {detail}", 400)
        raise WeatherError("Некоректний запит до сервісу погоди.", 400)

    if response.status_code >= 500:
        detail = _extract_error_message(payload)
        if detail:
            raise WeatherError(f"Сервіс погоди тимчасово недоступний: {detail}", 503)
        raise WeatherError("Сервіс погоди тимчасово недоступний.", 503)

    return payload


def find_city(name: str):
    """Знайти координати міста за назвою."""
    clean_name = (name or "").strip()
    if not clean_name:
        raise WeatherError("Введіть назву міста.", 400)

    data = _request_json(
        GEOCODING_URL,
        {
            "name": clean_name,
            "count": 1,
            "language": "uk",
            "format": "json",
        },
    )

    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list) or not results:
        raise WeatherError("Місто не знайдено. Перевірте назву.", 404)

    city = results[0]
    latitude = city.get("latitude")
    longitude = city.get("longitude")
    if latitude is None or longitude is None:
        raise WeatherError("Сервіс погоди повернув неповну інформацію про місто.", 502)

    return {
        "city": city.get("name") or clean_name,
        "latitude": float(latitude),
        "longitude": float(longitude),
    }


def get_current_weather(city: str):
    """Повернути поточну погоду в місті: температуру й швидкість вітру."""
    city_data = find_city(city)

    forecast = _request_json(
        FORECAST_URL,
        {
            "latitude": city_data["latitude"],
            "longitude": city_data["longitude"],
            "current": "temperature_2m,wind_speed_10m",
            "timezone": "auto",
        },
    )

    current = forecast.get("current") if isinstance(forecast, dict) else None
    units = forecast.get("current_units") if isinstance(forecast, dict) else {}

    if not isinstance(current, dict):
        raise WeatherError("Сервіс погоди повернув неповну інформацію про поточну погоду.", 502)

    temperature = current.get("temperature_2m")
    wind_speed = current.get("wind_speed_10m")

    if temperature is None or wind_speed is None:
        raise WeatherError("У відповіді сервісу відсутні потрібні дані про погоду.", 502)

    return {
        "city": city_data["city"],
        "temperature": float(temperature),
        "wind_speed": float(wind_speed),
        "temperature_unit": units.get("temperature_2m", "°C"),
        "wind_speed_unit": units.get("wind_speed_10m", "km/h"),
    }
