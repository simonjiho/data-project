"""Day 2 종합실습의 핵심 데이터 처리 테스트."""

from io import BytesIO
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import src.analysis_pipeline as analysis_pipeline  # noqa: E402
from src.analysis_pipeline import (  # noqa: E402
    COLUMNS,
    calculate_cramers_v,
    clean_pandas,
    create_report_environment,
    exclude_capital_outliers,
    format_p_value,
    run_rank_sum_test,
    run_t_test,
    validate_input,
)


def test_validate_input_downloads_missing_adult_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """입력 파일이 없을 때 UCI 원본을 지정 경로에 저장하는지 검증"""
    destination = tmp_path / "data" / "adult.data"
    sample_row = (
        b"39, State-gov, 77516, Bachelors, 13, Never-married, "
        b"Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, "
        b"United-States, <=50K\n"
    )

    def fake_urlopen(download_request, timeout):
        assert download_request.full_url == analysis_pipeline.ADULT_DATA_URL
        assert timeout == 30
        return BytesIO(sample_row)

    monkeypatch.setattr(analysis_pipeline.request, "urlopen", fake_urlopen)

    result = validate_input(destination)

    assert result == destination.resolve()
    assert destination.read_bytes() == sample_row


def test_report_template_and_custom_filters_are_available() -> None:
    """보고서 템플릿 로딩과 숫자 표시 필터 구성을 검증"""
    environment = create_report_environment()
    source, filename, _ = environment.loader.get_source(
        environment,
        "report.md.j2",
    )

    assert filename.endswith("report.md.j2")
    assert "# Adult Census Income 분석 보고서" in source
    assert environment.filters["number"](1234.5, 2) == "1,234.50"
    assert environment.filters["percent"](0.237, 1) == "23.7%"


def test_clean_pandas_removes_duplicate_and_empty_target() -> None:
    """중복 행과 target 결측 행 제거를 검증"""
    valid = [39, "Private", 1, "Bachelors", 13, "Single", "Tech", "Self", "White", "Male", 0, 0, 40, "US", "<=50K"]
    missing_target = [40, "Private", 2, "HS-grad", 9, "Single", "Sales", "Self", "White", "Female", 0, 0, 40, "US", None]
    frame = pd.DataFrame([valid, valid, missing_target], columns=COLUMNS)

    result = clean_pandas(frame)

    assert len(result) == 1
    assert result.iloc[0]["income"] == "<=50K"


def test_format_p_value_does_not_display_underflow_as_zero() -> None:
    """부동소수점 한계 아래 p-value 표시를 검증"""
    assert format_p_value(0.0) == "< 1e-300"
    assert format_p_value(0.012345) == "0.012345"


def test_cramers_v_is_zero_for_independent_table() -> None:
    """비율이 같은 분할표의 Cramér's V가 0인지 검증"""
    contingency = pd.DataFrame([[10, 20], [20, 40]])

    _, _, cramers_v, _ = calculate_cramers_v(contingency)

    assert cramers_v == 0.0


def test_rank_sum_detects_higher_values_in_high_income_group() -> None:
    """독립표본 순위합 검정의 방향과 효과크기를 검증"""
    frame = pd.DataFrame(
        {
            "income": [">50K"] * 3 + ["<=50K"] * 3,
            "hours_per_week": [50, 60, 70, 10, 20, 30],
        }
    )

    result = run_rank_sum_test(frame)

    assert result["high_median"] == 60.0
    assert result["low_median"] == 20.0
    assert result["rank_biserial"] == 1.0


def test_t_test_reports_confidence_interval_containing_mean_difference() -> None:
    """Welch t-test의 신뢰구간·자유도·효과크기 산출을 검증"""
    frame = pd.DataFrame(
        {
            "income": [">50K"] * 5 + ["<=50K"] * 5,
            "hours_per_week": [48, 50, 52, 54, 56, 38, 40, 42, 44, 46],
        }
    )

    result = run_t_test(frame)

    assert result["mean_difference"] == 10.0
    assert result["ci_lower"] < result["mean_difference"] < result["ci_upper"]
    assert 0 < result["dof"] <= 8
    assert result["cohens_d"] > 0


def test_capital_outlier_filter_preserves_zero_and_removes_extreme_value() -> None:
    """자본손익의 구조적 0 보존과 양수 극단값 제외를 검증"""
    frame = pd.DataFrame(
        {
            "capital_gain": [0, 100, 110, 120, 130, 140, 10_000],
            "capital_loss": [0, 1_600, 1_700, 1_800, 1_900, 2_000, 2_100],
        }
    )

    result, information = exclude_capital_outliers(frame)

    assert 0 in result["capital_gain"].to_numpy()
    assert 10_000 not in result["capital_gain"].to_numpy()
    assert information["excluded_count"] == 1


def test_capital_outlier_filter_reports_income_composition_of_excluded_rows() -> None:
    """이상치 제외의 소득 그룹 편중 진단 산출을 검증"""
    frame = pd.DataFrame(
        {
            "capital_gain": [0, 100, 110, 120, 130, 140, 10_000],
            "capital_loss": [0, 0, 0, 0, 0, 0, 0],
            "income": ["<=50K"] * 6 + [">50K"],
        }
    )

    _, information = exclude_capital_outliers(frame)

    assert information["excluded_high_income_rate"] == 100.0
    assert information["kept_high_income_rate"] == 0.0
