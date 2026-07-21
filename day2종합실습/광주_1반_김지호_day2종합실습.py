"""
Day 2 종합 실습: Adult Census Income End-to-End 데이터 분석 프로젝트.

UCI Adult Census Income 데이터를 Pandas와 Polars로 비교 로딩하고,
EDA·시각화·통계 검정·ML Pipeline·자동 보고서 생성을 순서대로 수행

작성자: 김지호
작성일자: 26.07.21

본 코드는 데이터 분석을 위한 Day 2 종합 실습입니다.

코드의 작업 내용은 다음과 같습니다.
- Pandas와 Polars로 같은 데이터를 로딩하고 처리 시간과 결과를 비교
- 결측치·중복·자료형을 확인하고 재현 가능한 정제 데이터 생성
- 수치형 변수의 Boxplot과 IQR 기준 이상치 후보를 분석
- 자본손익의 0을 보존하고 양수값 내부 IQR 극단값을 분석 데이터에서 제외
- Seaborn 정적 차트와 Plotly 인터랙티브 차트 생성
- 주요 범주형 변수와 소득 그룹의 Count bar plot 생성
- 기술통계·상관계수·정규성·Wilcoxon rank-sum·Welch t-test·카이제곱·Cramér's V 해석
- ColumnTransformer와 세 분류 모델을 각각 Pipeline으로 구성해 동일 조건 비교
- 정확도·F1·ROC-AUC와 ROC Curve를 평가하고 최고 Pipeline을 joblib으로 저장·재로딩
- 분석 결과와 본인 의견을 Jinja2 템플릿으로 report.md에 자동 생성


주 학습내용은 다음과 같습니다.
- Pandas와 Polars의 데이터 로딩·정제 방식 비교
- 결측치 처리 기준을 Pipeline 안에 두어 데이터 누수 방지
- Boxplot과 사분위 범위로 이상치 후보를 찾고 변수 분포에 맞게 해석
- Seaborn과 Plotly를 사용한 분포·그룹 비교
- Shapiro-Wilk 검정과 Q-Q plot을 함께 사용한 정규성 진단
- 비모수 Mann-Whitney U(Wilcoxon rank-sum)와 Welch t-test 비교
- 분석 용어와 검정 목적을 보고서 첫머리에서 정리
- sklearn Pipeline을 사용한 전처리·분류 모델 통합
- 분석 산출물과 Markdown 보고서 자동 생성



새로 학습한 내용을 정리한 바는 다음과 같습니다.

- End-to-End 분석
    - 데이터 준비부터 결과 공유까지 각 단계를 하나의 재현 가능한 흐름으로 연결
    - 같은 입력과 설정으로 다시 실행하면 같은 평가 결과와 보고서를 생성

- 데이터 누수 방지
    - 결측값 대체와 스케일링 기준을 전체 데이터가 아닌 훈련 데이터에서만 학습
    - ColumnTransformer와 Pipeline으로 전처리 누락과 순서 변경을 방지

- 통계적 유의성과 실질적 의미
    - p-value로 차이의 검출 여부를 판단하되 평균 차이와 표본 수를 함께 해석
    - 통계적으로 유의하다는 결과가 반드시 업무적으로 큰 차이를 의미하지는 않음


실행 예시
    python 광주_1반_김지호_day2종합실습.py
"""

from pathlib import Path

from src.analysis_pipeline import run_pipeline


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "adult.data"
OUTPUT_DIR = BASE_DIR / "output"


def main() -> None:
    """Adult Census 종합 분석 Pipeline을 실행"""
    run_pipeline(DATA_PATH, OUTPUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, TypeError) as error:
        raise SystemExit(f"오류: {error}") from error
    except OSError as error:
        raise SystemExit(f"파일 처리 오류: {error}") from error
