"""
Practice 2: 파일 I/O, 예외 처리, Pydantic 검증 파이프라인.

Python_Practice2_Data.json을 읽어 SalesRecord 스키마로 검증하고,
정상 레코드는 CSV로, 오류 정보는 JSON으로 저장한 뒤 다시 읽어 검증

작성자: 김지호
작성일자: 26.07.20

본 코드는 데이터 분석을 위한 python 실습2입니다.

코드의 작업 내용은 다음과 같습니다.
- Python_Practice2_Data.json을 읽어 SalesRecord 스키마로 검증
- 정상 레코드는 CSV로, 오류 정보는 JSON으로 저장한 뒤 다시 읽어 검증


주 학습내용은 다음과 같습니다.
- csv, json 포맷 파일의 FileIO
- Pydantic을 사용한 검증 파이프라인 설계
- 예외 처리



새로 학습한 내용을 정리한 바는 다음과 같습니다.

- @field_validator(attr1, attr2, ...)
    - attr1, att2에 적용하는 field validator
    - 기존의 Field를 이용한 validation보다 좀 더 복잡한 validation을 수행 시 사용
    - @classmethod와 같이 사용하는 것이 표준
    - Pydantic v2에서는 첫 번째 매개변수 이름이 cls인 검증기를 보고 클래스 메서드 형태로 자동 처리해줄 수 있다.

    
- @classmethod
    - 객체의 method가 아닌 class의 method로 작용, 해당 데코레이터의 함수는 첫번째 인자가 무조건 클래스 자신이고, 입력하지 않아도 첫번째 인자로 자신을 전달.



"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger("practice2")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
INPUT_PATH = DATA_DIR / "Python_Practice2_Data.json"
OUTPUT_DIR = BASE_DIR / "output"
VALID_CSV_PATH = OUTPUT_DIR / "valid_sales.csv"
ERROR_JSON_PATH = OUTPUT_DIR / "validation_errors.json"


def safe_load_csv(path: Path) -> list[dict[str, str]] | None:
    """csv 파일을 로드"""
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        logger.info("CSV 로딩 성공: %s (%d건)", path, len(rows))
        return rows
    except FileNotFoundError:
        logger.error("CSV 파일을 찾을 수 없습니다: %s", path)
        return None
    except (OSError, csv.Error) as error:
        logger.error("CSV 로딩 실패: %s (%s)", path, error)
        return None
    finally:
        logger.info("로딩 종료")


def safe_load_json(path: Path) -> list[dict[str, Any]] | None:
    """json 파일을 로드"""
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
            raise ValueError("최상위 데이터는 dict로 이루어진 리스트여야 합니다.")

        logger.info("JSON 로딩 성공: %s (%d건)", path, len(data))
        return data
    except FileNotFoundError:
        logger.error("JSON 파일을 찾을 수 없습니다: %s", path)
        return None
    except (OSError, json.JSONDecodeError, ValueError) as error:
        logger.error("JSON 로딩 실패: %s (%s)", path, error)
        return None
    finally:
        logger.info("로딩 종료")


class SalesRecord(BaseModel): # 검증용 레코드

    month: str = Field(min_length=1, description="필수")
    region: str = Field(min_length=1, description="필수")
    amount: float = Field(ge=1, description="양수")
    category: str | None = None # 생략 가능한 attribute

    @field_validator("month", "region")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        """공백만 입력된 month와 region도 거부"""
        value = value.strip()
        if not value: # 빈 문자열이면 raise error
            raise ValueError("비어 있을 수 없습니다.")
        return value


def validate_records(
    raw_data: list[dict[str, Any]],
) -> tuple[list[SalesRecord], list[dict[str, Any]]]:
    """원본 데이터를 검증해 valid와 errors로 분리"""
    valid: list[SalesRecord] = []
    errors: list[dict[str, Any]] = []

    # enumerate로 행 번호와 데이터를 함께 추출
    for row_number, row in enumerate(raw_data):
        try:
            # dict의 키·값을 Pydantic 모델의 키워드 인자(**kargs)로 전달
            valid.append(SalesRecord(**row))
        except ValidationError as error:
            errors.append({"row": row_number, "error": str(error)})
            logger.warning("%d행 검증 실패: %s", row_number, error)

    return valid, errors


def save_valid_csv(records: list[SalesRecord], path: Path) -> None:
    """검증에 성공한 레코드를 CSV로 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["month", "region", "amount", "category"]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record.model_dump() for record in records)



def save_errors_json(errors: list[dict[str, Any]], path: Path) -> None:
    """검증 오류 목록을 JSON으로 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), # ensure_ascii=False -> 한글이 깨지지 않음
        encoding="utf-8",
    )





if __name__ == "__main__":
    """로딩, 검증, 저장, 재로딩 확인을 순서대로 실행"""
    # 존재하지 않는 파일에서는 None을 반환하는지 확인
    assert safe_load_csv(BASE_DIR / "does_not_exist.csv") is None

    raw_data = safe_load_json(INPUT_PATH)
    if raw_data is None:
        raise SystemExit("원본 데이터를 읽지 못해 프로그램을 종료합니다.")

    valid, errors = validate_records(raw_data)

    assert len(valid) + len(errors) == len(raw_data)

    save_valid_csv(valid, VALID_CSV_PATH)
    save_errors_json(errors, ERROR_JSON_PATH)

    reloaded = safe_load_csv(VALID_CSV_PATH)
    assert reloaded is not None
    assert len(reloaded) == len(valid)

    print(f"유효: {len(valid)}건, 오류: {len(errors)}건")
    print(f"재로딩: {len(reloaded)}건")
    print(f"정상 데이터 저장: {VALID_CSV_PATH}")
    print(f"오류 데이터 저장: {ERROR_JSON_PATH}")
