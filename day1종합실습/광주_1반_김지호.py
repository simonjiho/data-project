"""
Day 1 종합 실습: 비동기 API 수집·검증·저장·성능 비교 파이프라인.
본 코드는 데이터 분석을 위한 Python Day 1 종합 실습입니다.

작성자: 김지호
작성일자: 26.07.20


코드의 작업 내용은 다음과 같습니다.
- Open-Meteo, Countries.dev, ip-api를 asyncio.gather()로 동시에 호출
- API별 응답을 Pydantic v2 모델로 검증하고 공통 DataFrame으로 변환
- 검증 데이터를 CSV와 Parquet으로 저장하고 읽기·쓰기 시간 및 파일 크기 비교
- 네트워크 오류 발생 시 재시도하고 파이프라인 오류를 로깅


주 학습내용은 다음과 같습니다.
- asyncio, async, await를 활용한 비동기 처리
- httpx.AsyncClient를 활용한 여러 API 동시 수집
- pandas를 활용한 CSV·Parquet 파일 저장 및 재로딩
- 데코레이터를 활용한 실행 시간 측정 및 API 재시도 기능 분리
- pytest와 Ruff를 활용한 테스트 및 코드 품질 검사


새로 학습한 내용을 정리한 바는 다음과 같습니다.

- tasks 리스트 앞의 *는 리스트 내부 항목을 위치 인수(*args)로 전달하는 unpacking


- async
    - 비동기 함수나 비동기 컨텍스트를 정의할 때 사용하는 키워드
    - async def로 정의한 함수를 코루틴 함수라고 부름
    - 코루틴 함수를 호출하면 함수 본문이 즉시 실행되는 것이 아니라 코루틴 객체를 반환
    - 생성된 코루틴은 await, asyncio.run(), asyncio.create_task() 등을 통해 실행
    - async를 사용했다고 자동으로 여러 작업이 동시에 실행되는 것은 아님



- await        
    - 코루틴, Task 등 비동기 작업의 완료 결과를 받을 때 사용하는 키워드
    - 일반적으로 async def로 정의한 함수 내부에서만 사용 가능
    - 해당 작업이 끝나지 않았다면 현재 코루틴을 잠시 중지
    - 중지하는 동안 이벤트 루프에 제어권을 반환하여 다른 비동기 작업이 실행될 수 있게 함
    - 작업이 완료되면 중지된 위치로 돌아와 결과를 반환
    - await를 여러 번 순서대로 작성하면 작업도 기본적으로 순차 실행
    - 여러 작업을 동시에 진행하려면 asyncio.gather() 또는 asyncio.create_task() 등을 사용
    - 코루틴을 만들고 await하지 않으면 실행되지 않거나 경고가 발생할 수 있음
    


- Parquet
    - Parquet은 column 기반 형식으로 대용량 데이터의 압축과 선택적 column 조회에 유리
    - 작은 데이터에서는 schema·metadata 생성 비용 때문에 Parquet이 CSV보다 더 느리거나 클 수 있음
    
"""

import asyncio
import json
import logging
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Callable, TypeVar

import httpx
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator



logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("day1-pipeline")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CSV_PATH = OUTPUT_DIR / "collected_data.csv"
PARQUET_PATH = OUTPUT_DIR / "collected_data.parquet"

API_URLS = {
    "weather": (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=37.5665&longitude=126.9780"
        "&hourly=temperature_2m,precipitation_probability"
        "&forecast_days=3&timezone=Asia%2FSeoul"
    ),
    "country": "https://countries.dev/alpha/KOR",
    "ip": "http://ip-api.com/json/8.8.8.8",
}

ResultType = TypeVar("ResultType")


def timer(func: Callable[..., ResultType]) -> Callable[..., ResultType]:
    """함수 실행 시간을 출력하는 데코레이터"""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> ResultType:
        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed = perf_counter() - start
        print(f"{func.__name__}: {elapsed:.6f}초")
        return result

    return wrapper


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
) -> Callable[
    [Callable[..., Awaitable[ResultType]]],
    Callable[..., Awaitable[ResultType]],
]:
    """일시적인 API 오류 발생 시 비동기 함수를 재시도하는 데코레이터"""

    def decorator(
        func: Callable[..., Awaitable[ResultType]],
    ) -> Callable[..., Awaitable[ResultType]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> ResultType:
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except httpx.HTTPStatusError as error:
                    status_code = error.response.status_code
                    retryable = status_code == 429 or status_code >= 500
                    if not retryable or attempt == max_attempts:
                        raise
                    logger.warning(
                        "%s 재시도 %d/%d - HTTP %d",
                        func.__name__,
                        attempt,
                        max_attempts,
                        status_code,
                    )
                except httpx.RequestError as error:
                    if attempt == max_attempts:
                        raise
                    logger.warning(
                        "%s 재시도 %d/%d - %s",
                        func.__name__,
                        attempt,
                        max_attempts,
                        error,
                    )

                await asyncio.sleep(delay * attempt)

            raise RuntimeError("재시도 횟수를 모두 소진했습니다.")

        return wrapper

    return decorator


class HourlyWeather(BaseModel):
    """시간대별 시각, 기온, 강수확률 배열"""

    time: list[str] = Field(min_length=1)
    temperature_2m: list[float] = Field(min_length=1)
    precipitation_probability: list[int] = Field(min_length=1)

    @field_validator("precipitation_probability")
    @classmethod
    def check_probability(cls, values: list[int]) -> list[int]:
        if any(value < 0 or value > 100 for value in values):
            raise ValueError("강수확률은 0~100 범위여야 합니다.")
        return values


class WeatherResponse(BaseModel):
    """Open-Meteo 응답에서 필요한 필드"""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str = Field(min_length=1)
    hourly: HourlyWeather

    @field_validator("hourly")
    @classmethod
    def check_hourly_lengths(cls, hourly: HourlyWeather) -> HourlyWeather:
        lengths = {
            len(hourly.time),
            len(hourly.temperature_2m),
            len(hourly.precipitation_probability),
        }
        if len(lengths) != 1:
            raise ValueError("시간·기온·강수확률 배열 길이가 서로 다릅니다.")
        return hourly


class CountryResponse(BaseModel):
    """Countries.dev 응답에서 필요한 한국 국가 정보"""

    name: str = Field(min_length=1)
    alpha2_code: str = Field(
        min_length=2,
        max_length=2,
        validation_alias="alpha2Code",
    )
    region: str = Field(min_length=1)
    population: int = Field(gt=0)
    capital: list[str] | str | None = None


class IpResponse(BaseModel):
    """ip-api 응답에서 필요한 IP 기반 지역 정보"""

    status: str
    query: str
    country: str = Field(min_length=1)
    city: str = Field(min_length=1)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)

    @field_validator("status")
    @classmethod
    def check_status(cls, value: str) -> str:
        if value != "success":
            raise ValueError("ip-api 응답 상태가 success가 아닙니다.")
        return value


@retry(max_attempts=3, delay=1.0)
async def request_json(
    client: httpx.AsyncClient,
    url: str,
) -> dict[str, Any]:
    """HTTP 요청 후 JSON 객체를 반환"""
    response = await client.get(url)
    response.raise_for_status()  # 4xx, 5xx 응답을 예외로 변환
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("JSON 최상위 값이 객체가 아닙니다.")
    return data


async def fetch(
    client: httpx.AsyncClient,
    name: str,
    url: str,
) -> tuple[str, dict[str, Any] | None, str | None]:
    """API 하나를 호출하고 실패 정보를 데이터로 반환"""
    try:
        data = await request_json(client, url)
        logger.info("%s API 수집 성공", name)
        return name, data, None
    except (httpx.HTTPError, json.JSONDecodeError, ValueError) as error:
        logger.error("%s API 수집 실패: %s", name, error)
        return name, None, str(error)


async def fetch_all() -> dict[str, dict[str, Any]]:
    """세 API를 asyncio.gather()로 동시에 수집"""
    timeout = httpx.Timeout(15.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        tasks = [fetch(client, name, url) for name, url in API_URLS.items()]
        results = await asyncio.gather(*tasks) # task들이 모두 작업 완료 될 때까지 대기

    collected: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for name, data, error in results:
        if data is not None:
            collected[name] = data
        else:
            failures.append(f"{name}: {error}")

    if failures:
        raise RuntimeError("일부 API 수집 실패 - " + " | ".join(failures))
    return collected


def validate_and_flatten(raw: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """API 응답을 Pydantic으로 검증하고 저장 가능한 공통 행으로 변환"""
    weather = WeatherResponse.model_validate(raw["weather"])
    country = CountryResponse.model_validate(raw["country"])
    ip_info = IpResponse.model_validate(raw["ip"])

    rows: list[dict[str, Any]] = []
    for time, temperature, probability in zip(
        weather.hourly.time,
        weather.hourly.temperature_2m,
        weather.hourly.precipitation_probability,
        strict=True,
    ):
        rows.append(
            {
                "source": "open-meteo",
                "time": time,
                "name": "Seoul",
                "temperature": temperature,
                "precipitation_probability": probability,
                "detail": weather.timezone,
            }
        )

    capital = country.capital
    if isinstance(capital, list):
        capital = ", ".join(capital)
    rows.append(
        {
            "source": "countries.dev",
            "time": None,
            "name": country.name,
            "temperature": None,
            "precipitation_probability": None,
            "detail": (
                f"{country.alpha2_code}/{country.region}/{capital}/{country.population}"
            ),
        }
    )
    rows.append(
        {
            "source": "ip-api",
            "time": None,
            "name": f"{ip_info.city}, {ip_info.country}",
            "temperature": None,
            "precipitation_probability": None,
            "detail": f"{ip_info.query}/{ip_info.lat}/{ip_info.lon}",
        }
    )
    return pd.DataFrame(rows)


@timer
def write_csv(df: pd.DataFrame, path: Path) -> None:
    """DataFrame을 CSV로 저장"""
    df.to_csv(path, index=False, encoding="utf-8-sig")


@timer
def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """DataFrame을 Parquet으로 저장"""
    df.to_parquet(path, index=False)


@timer
def read_csv(path: Path) -> pd.DataFrame:
    """CSV를 DataFrame으로 로드"""
    return pd.read_csv(path)


@timer
def read_parquet(path: Path) -> pd.DataFrame:
    """Parquet을 DataFrame으로 로드"""
    return pd.read_parquet(path)


def save_and_compare(df: pd.DataFrame) -> None:
    """CSV·Parquet 저장/읽기 시간과 파일 크기 측정"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[CSV / Parquet 성능 비교]")
    write_csv(df, CSV_PATH)
    write_parquet(df, PARQUET_PATH)
    csv_reloaded = read_csv(CSV_PATH)
    parquet_reloaded = read_parquet(PARQUET_PATH)

    assert len(csv_reloaded) == len(df)
    assert len(parquet_reloaded) == len(df)

    print(f"CSV 크기: {CSV_PATH.stat().st_size:,} bytes")
    print(f"Parquet 크기: {PARQUET_PATH.stat().st_size:,} bytes")





if __name__ == "__main__":
    """종합 실습 파이프라인 실행"""
    try:
        raw = asyncio.run(fetch_all())
        validated = validate_and_flatten(raw)
        save_and_compare(validated)
        print(f"\n검증·저장 완료: {len(validated)}행")
        print(f"결과 폴더: {OUTPUT_DIR}")
    except (RuntimeError, ValidationError, OSError, ValueError) as error:
        logger.exception("파이프라인 실행 실패: %s", error)
        raise SystemExit(1) from error


    # 대용량 데이터 성능 측정용
    target_rows = 1_000_000
    repeat_count = (target_rows + len(validated) - 1) // len(validated)
    large_df = pd.concat([validated] * repeat_count, ignore_index=True).iloc[:target_rows]
    print(f"\n[대용량 데이터 성능 측정: {len(large_df):,}행]")
    save_and_compare(large_df)
