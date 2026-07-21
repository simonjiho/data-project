"""
Practice 3: Pandas EDA, Polars Lazy API, DuckDB 성능 비교.

sales_100k.csv를 읽어 기본 EDA와 IQR 이상치 처리를 수행하고,
Pandas, Polars, DuckDB로 같은 집계를 실행한 뒤 처리 시간을 비교

작성자: 김지호
작성일자: 26.07.21

본 코드는 데이터 분석을 위한 python 실습3입니다.

실행 예시
    python 광주_1반_김지호.py sales_100k.csv --number 5

코드의 작업 내용은 다음과 같습니다.
- sales_100k.csv의 데이터 구조와 컬럼별 결측치를 확인
- amount 컬럼의 IQR 정상 범위를 계산하고 범위 밖의 행을 제외
- region, category별 total, mean, count를 세 가지 도구로 집계
- 세 도구의 집계 결과가 같은지 검증
- 동일한 반복 횟수로 각 도구의 실행 시간을 측정하여 비교


주 학습내용은 다음과 같습니다.
- Pandas를 사용한 기본 EDA와 named aggregation
- IQR을 이용한 이상치 처리
- Polars scan_csv와 collect를 사용한 Lazy API
- DuckDB Relation API를 사용한 지연 실행 방식
- timeit을 사용한 동일 조건의 성능 비교
- 입력 파일, 필수 컬럼 및 데이터 처리 오류에 대한 예외 처리



새로 학습한 내용을 정리한 바는 다음과 같습니다.

- Lazy API
    - 연산을 호출할 때마다 즉시 실행하지 않고 실행 계획을 먼저 구성
    - Polars는 collect(), DuckDB Relation API는 df()를 호출할 때 실제 연산 수행

- IQR
    - Q3 - Q1로 데이터 가운데 50%의 범위를 나타냄
    - 1.5 * IQR 범위 밖의 값은 반드시 오류가 아니라 확인이 필요한 이상치 후보
    
- Apache Arrow: 여러 데이터 도구가 함께 사용할 수 있는 컬럼형 인메모리 표준
    - Polars와 DuckDB는 모두 Apache Arrow 형식의 입출력을 지원한다. 
    - 따라서 Arrow를 중간 교환 형식으로 사용하면 CSV나 Pandas 객체를 거치는 방식보다 직렬화와 데이터 복사를 줄일 수 있으며, 
    - 자료형과 메모리 구조가 호환되는 일부 변환에서는 제로카피가 가능하다. 
    - 다만 DuckDB는 내부적으로 자체 실행 형식을 사용하며, 모든 변환과 연산에서 제로카피가 보장되는 것은 아니다.


"""


import argparse
import timeit
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd
import polars as pl

REQUIRED_COLUMNS = {"region", "category", "amount"}


def parse_args() -> argparse.Namespace:
    """입력 CSV 경로와 성능 측정 반복 횟수를 읽습니다."""
    parser = argparse.ArgumentParser(
        description="Pandas, Polars Lazy, DuckDB로 동일한 매출 집계를 비교"
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "sales_100k.csv",
        help="sales_100k.csv 경로 (기본값: 프로젝트의 data 폴더)",
    )
    parser.add_argument(
        "--number",
        type=int,
        default=3,
        help="각 도구의 timeit 반복 횟수 (기본값: 3)",
    )
    args = parser.parse_args()
    if args.number < 1:
        parser.error("--number는 1 이상의 정수여야 합니다.")
    return args


def validate_input(csv_path: Path) -> Path:
    """파일 존재 여부와 필수 컬럼을 검사하고 절대 경로를 반환"""
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"입력 파일을 찾을 수 없습니다: {csv_path}\n"
            "sales_100k.csv를 프로젝트의 data 폴더에 넣거나 경로를 인자로 전달하세요."
        )


    columns = set(pd.read_csv(csv_path, nrows=0).columns)
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(sorted(missing))}")
    return csv_path


def pandas_eda(csv_path: Path) -> tuple[float, float]:
    """기본 EDA를 출력하고 amount의 IQR 정상 범위를 계산"""
    df = pd.read_csv(csv_path)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    print("\n[1. Pandas 기본 EDA]")
    print(f"데이터 크기: {df.shape[0]:,}행 x {df.shape[1]:,}열")
    print("\n- df.info()")
    df.info()
    print("\n- isnull().sum()")
    print(df.isnull().sum())
    print("\n- 수치형 기술통계")
    print(df.describe(include='all')) # 범주형 포함

    valid_amount = df["amount"].dropna()
    if valid_amount.empty:
        raise ValueError("amount 컬럼에 분석 가능한 숫자가 없습니다.")

    q1 = valid_amount.quantile(0.25)
    q3 = valid_amount.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    normal = df["amount"].between(lower, upper)
    print("\n- IQR 이상치 처리")
    print(f"정상 범위: {lower:,.2f} <= amount <= {upper:,.2f}")
    print(f"제거 전: {len(df):,}행")
    print(f"제거 후: {normal.sum():,}행")
    print(f"제거 수: {(~normal).sum():,}행 (결측/숫자 변환 실패 포함)")
    return float(lower), float(upper)


def aggregate_pandas(csv_path: Path, lower: float, upper: float) -> pd.DataFrame:
    """Pandas named aggregation으로 지역·카테고리별 매출을 집계"""
    df = pd.read_csv(csv_path)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    clean = df.loc[
        df["amount"].between(lower, upper)
        & df["region"].notna()
        & df["category"].notna()
    ]
    return (
        clean.groupby(["region", "category"], as_index=False)
        .agg(
            total=("amount", "sum"),
            mean=("amount", "mean"),
            count=("amount", "count"),
        )
        .sort_values("total", ascending=False, ignore_index=True)
    )


def aggregate_polars(csv_path: Path, lower: float, upper: float) -> pl.DataFrame:
    """Polars Lazy API의 scan-filter-group-agg-sort-collect 체인을 실행"""
    return (
        pl.scan_csv(csv_path, schema_overrides={"amount": pl.String})
        .with_columns(pl.col("amount").cast(pl.Float64, strict=False))
        .filter(
            pl.col("amount").is_between(lower, upper, closed="both")
            & pl.col("region").is_not_null()
            & pl.col("category").is_not_null()
        )
        .group_by(["region", "category"])
        .agg(
            pl.col("amount").sum().alias("total"),
            pl.col("amount").mean().alias("mean"),
            pl.col("amount").count().alias("count"),
        )
        .sort("total", descending=True)
        .collect()
    )


def aggregate_duckdb(csv_path: Path, lower: float, upper: float) -> pd.DataFrame:
    """DuckDB Relation API로 실행 계획을 만든 뒤 마지막에 DataFrame을 생성"""
    with duckdb.connect() as connection:
        # 각 메서드는 즉시 Pandas DataFrame을 만들지 않고 DuckDB의 Relation을
        # 반환합니다. 마지막 df() 호출 시 전체 실행 계획이 실제로 실행됩니다.
        sql = """
            WITH typed AS (
                SELECT
                    region,
                    category,
                    TRY_CAST(amount AS DOUBLE) AS amount
                FROM read_csv_auto(?, all_varchar = true)
            )
            SELECT
                region,
                category,
                SUM(amount) AS total,
                AVG(amount) AS mean,
                COUNT(amount) AS count
            FROM typed
            WHERE amount BETWEEN ? AND ?
              AND region IS NOT NULL
              AND category IS NOT NULL
            GROUP BY region, category
            ORDER BY total DESC
        """
        return connection.execute(
            sql, [str(csv_path), lower, upper]
        ).df()


def measure(
    name: str,
    function: Callable[[Path, float, float], object],
    csv_path: Path,
    lower: float,
    upper: float,
    number: int,
) -> float:
    """전달받은 함수를 timeit으로 동일 횟수 실행하고 소요 시간을 출력"""
    elapsed = timeit.timeit(
        lambda: function(csv_path, lower, upper),
        number=number,
    )
    print(f"{name:<8}: 총 {elapsed:.6f}초 / 평균 {elapsed / number:.6f}초")
    return elapsed


def normalize_result(result: object) -> pd.DataFrame:
    """세 도구의 결과를 Pandas 형식으로 맞춰 값 비교"""
    if isinstance(result, pl.DataFrame):
        result = result.to_pandas()
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"예상하지 못한 결과 타입입니다: {type(result).__name__}")
    return result.sort_values(["region", "category"]).reset_index(drop=True)


def verify_same_results(*results: object) -> None:
    """세 집계 결과의 그룹, 개수 및 수치가 같은지 검증"""
    expected = normalize_result(results[0])
    for result in results[1:]:
        actual = normalize_result(result)
        pd.testing.assert_frame_equal(
            expected,
            actual,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
        )


def main() -> None:
    """입력 검증, EDA, 세 도구 집계·검증·성능 측정을 순서대로 수행"""
    args = parse_args()
    csv_path = validate_input(args.csv_path)
    lower, upper = pandas_eda(csv_path)

    pandas_result = aggregate_pandas(csv_path, lower, upper)
    polars_result = aggregate_polars(csv_path, lower, upper)
    duckdb_result = aggregate_duckdb(csv_path, lower, upper)
    verify_same_results(pandas_result, polars_result, duckdb_result)

    print("\n[2. Pandas named aggregation 결과]")
    print(pandas_result.to_string(index=False))
    print("\n[3. Polars Lazy API 결과]")
    print(polars_result)
    print("\n[4. DuckDB 지연 실행 Relation API 결과]")
    print(duckdb_result.to_string(index=False))
    print("\n세 도구의 집계 결과가 같습니다.")

    print(f"\n[5. 실행 시간 비교: 각 {args.number}회]")
    measure("Pandas", aggregate_pandas, csv_path, lower, upper, args.number)
    measure("Polars", aggregate_polars, csv_path, lower, upper, args.number)
    measure("DuckDB", aggregate_duckdb, csv_path, lower, upper, args.number)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, TypeError, pd.errors.ParserError) as error:
        raise SystemExit(f"오류: {error}") from error
    except (duckdb.Error, pl.exceptions.PolarsError) as error:
        raise SystemExit(f"데이터 처리 오류: {error}") from error
