"""
pytest를 활용한 Pydantic 스키마 검증 테스트.

pytest

"""

import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError


MODULE_PATH = Path(__file__).with_name("광주_1반_김지호.py")
SPEC = importlib.util.spec_from_file_location("weather_data_validation_pipeline", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
main_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(main_module)


def test_weather_rejects_invalid_probability() -> None:
    """강수확률이 0~100 밖이면 ValidationError가 발생해야 한다."""
    with pytest.raises(ValidationError):
        main_module.WeatherResponse.model_validate(
            {
                "latitude": 37.5665,
                "longitude": 126.9780,
                "timezone": "Asia/Seoul",
                "hourly": {
                    "time": ["2026-07-20T00:00"],
                    "temperature_2m": [25.0],
                    "precipitation_probability": [101],
                },
            }
        )


def test_ip_requires_success_status() -> None:
    """ip-api 실패 응답은 ValidationError로 처리해야 한다."""
    with pytest.raises(ValidationError):
        main_module.IpResponse.model_validate(
            {
                "status": "fail",
                "query": "8.8.8.8",
                "country": "United States",
                "city": "Mountain View",
                "lat": 37.4,
                "lon": -122.1,
            }
        )
