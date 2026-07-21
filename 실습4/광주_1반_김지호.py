"""
Practice 4: 시각화 4종, 통계 검정, sklearn Pipeline.

실습 3의 sales_100k.csv 로딩과 IQR 이상치 처리 결과를 재사용하여
시각화, 가설 검정, 머신러닝 Pipeline 및 인터랙티브 차트를 생성

작성자: 김지호
작성일자: 26.07.21

본 코드는 데이터 분석을 위한 python 실습4입니다.

코드의 작업 내용은 다음과 같습니다.
- sales_100k.csv를 읽고 amount 컬럼의 IQR 이상치 후보를 제외
- 히스토그램, 박스플롯, 월별 라인(날짜가 없으면 막대), 상관 히트맵을 2×2로 저장
- 두 지역의 평균 매출 차이를 Welch t-test로 검정하고 p-value를 해석
- region과 category의 독립성을 카이제곱 검정으로 확인하고 Cramér's V로 관계 크기를 해석
- ColumnTransformer와 Ridge를 Pipeline으로 묶어 학습, 예측, 평가
- 학습한 Pipeline을 joblib 파일로 저장한 뒤 다시 로딩하여 결과 확인
- 지역·카테고리별 총매출 Plotly 막대 차트를 HTML 파일로 저장


주 학습내용은 다음과 같습니다.
- Matplotlib Figure와 Axes를 사용한 2×2 서브플롯
- Seaborn을 사용한 분포와 상관관계 시각화
- scipy.stats의 ttest_ind와 chi2_contingency를 사용한 가설 검정
- Cramér's V를 사용한 범주형 변수 간 관계 크기 측정
- p-value와 유의수준 0.05를 비교한 통계적 해석
- ColumnTransformer와 Pipeline을 사용한 데이터 누수 방지
- joblib을 사용한 Pipeline 저장과 재로딩
- Plotly Express를 사용한 인터랙티브 HTML 차트 생성



새로 학습한 내용을 정리한 바는 다음과 같습니다.

- Figure와 Axes
    - Figure는 전체 캔버스, Axes는 캔버스 안의 개별 차트 영역
    - plt.subplots(2, 2)를 사용하면 네 차트를 하나의 결과물로 비교 가능

- 가설 검정
    - p-value가 0.05보다 작으면 차이 또는 연관성이 없다는 귀무가설을 기각
    - p-value가 0.05 이상이면 귀무가설을 기각할 근거가 부족한 것이며 같다는 증명은 아님

- Cramér's V
    - 카이제곱 검정에서 두 범주형 변수 간 관계의 실질적인 크기를 측정
    - p-value는 관계의 검출 여부, Cramér's V는 관계가 얼마나 큰지를 설명

- sklearn Pipeline
    - 전처리와 모델을 하나의 객체로 묶어 학습 데이터와 평가 데이터에 같은 절차 적용
    - 전처리 기준을 학습 데이터에서만 계산하여 데이터 누수를 방지


실행 예시
    python 광주_1반_김지호.py sales_100k.csv
"""

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import font_manager
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REQUIRED_COLUMNS = {"region", "category", "amount"}
SIGNIFICANCE_LEVEL = 0.05
RANDOM_STATE = 42
DATE_COLUMN_CANDIDATES = (
    "date",
    "order_date",
    "sales_date",
    "transaction_date",
    "order_datetime",
)


def parse_args() -> argparse.Namespace:
    """명령행에서 입력 CSV 경로와 결과 저장 폴더를 확인"""
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="실습 3의 정제 데이터를 활용해 실습 4 분석을 수행합니다."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=base_dir.parent / "data" / "sales_100k.csv",
        help="sales_100k.csv 경로 (기본값: 프로젝트의 data 폴더)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "output",
        help="차트와 모델을 저장할 폴더 (기본값: 실습4/output)",
    )
    return parser.parse_args()


def validate_input(csv_path: Path) -> Path:
    """CSV 존재 여부와 실습 필수 컬럼을 검증"""
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


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """실습 3과 같은 IQR 기준으로 amount 이상치 후보와 결측 행을 제외"""
    df = pd.read_csv(csv_path)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    valid_amount = df["amount"].dropna()
    if valid_amount.empty:
        raise ValueError("amount 컬럼에 분석 가능한 숫자가 없습니다.")

    q1 = valid_amount.quantile(0.25)
    q3 = valid_amount.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    normal = df["amount"].between(lower, upper)
    clean = df.loc[
        normal & df["region"].notna() & df["category"].notna()
    ].copy()

    if clean.empty:
        raise ValueError("IQR 및 결측치 처리 후 남은 데이터가 없습니다.")

    print("\n[1. 실습 3 정제 결과 재사용]")
    print(f"IQR 정상 범위: {lower:,.2f} <= amount <= {upper:,.2f}")
    print(f"정제 전: {len(df):,}행")
    print(f"정제 후: {len(clean):,}행")
    print(f"제외 수: {len(df) - len(clean):,}행")
    return clean


def aggregate_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Plotly와 분석에 재사용할 지역·카테고리별 매출을 집계"""
    return (
        df.groupby(["region", "category"], as_index=False)
        .agg(
            total=("amount", "sum"),
            mean=("amount", "mean"),
            count=("amount", "count"),
        )
        .sort_values("total", ascending=False, ignore_index=True)
    )


def find_date_column(df: pd.DataFrame) -> str | None:
    """이름과 변환 성공률을 기준으로 월별 차트용 날짜 컬럼을 탐색"""
    lower_names = {str(column).lower(): str(column) for column in df.columns}
    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in lower_names:
            return lower_names[candidate]

    for column in df.select_dtypes(include="object").columns:
        if "date" not in str(column).lower() and "일자" not in str(column):
            continue
        converted = pd.to_datetime(df[column], errors="coerce")
        if converted.notna().mean() >= 0.8:
            return str(column)
    return None


def configure_korean_font() -> None:
    """현재 운영체제에 실제 설치된 한글 글꼴만 Matplotlib에 적용"""
    mac_font_path = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    if mac_font_path.is_file():
        font_manager.fontManager.addfont(str(mac_font_path))
        font_name = font_manager.FontProperties(fname=str(mac_font_path)).get_name()
        plt.rcParams["font.family"] = font_name
    else:
        installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
        candidates = ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR")
        selected_font = next(
            (font for font in candidates if font in installed_fonts),
            "DejaVu Sans",
        )
        plt.rcParams["font.family"] = selected_font

    plt.rcParams["axes.unicode_minus"] = False


def create_eda_visualization(df: pd.DataFrame, output_path: Path) -> None:
    """네 가지 EDA 차트를 하나의 2×2 Figure로 저장"""
    sns.set_theme(style="whitegrid")
    configure_korean_font()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    sns.histplot(data=df, x="amount", kde=True, ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("매출 금액 분포")
    axes[0, 0].set_xlabel("매출 금액")
    axes[0, 0].set_ylabel("거래 건수")

    sns.boxplot(data=df, x="region", y="amount", ax=axes[0, 1], color="teal")
    axes[0, 1].set_title("지역별 매출 금액 분포")
    axes[0, 1].set_xlabel("지역")
    axes[0, 1].set_ylabel("매출 금액")
    axes[0, 1].tick_params(axis="x", rotation=30)

    date_column = find_date_column(df)
    if date_column is not None:
        dates = pd.to_datetime(df[date_column], errors="coerce")
        monthly = (
            df.assign(month=dates.dt.to_period("M").dt.to_timestamp())
            .dropna(subset=["month"])
            .groupby("month", as_index=False)
            .agg(total=("amount", "sum"))
        )
        sns.lineplot(
            data=monthly,
            x="month",
            y="total",
            marker="o",
            ax=axes[1, 0],
            color="darkorange",
        )
        axes[1, 0].set_title("월별 총매출 추이")
        axes[1, 0].set_xlabel("월")
        axes[1, 0].set_ylabel("총매출")
        axes[1, 0].tick_params(axis="x", rotation=30)
    else:
        category_sales = (
            df.groupby("category", as_index=False)
            .agg(total=("amount", "sum"))
            .sort_values("total", ascending=False)
        )
        sns.barplot(
            data=category_sales,
            x="category",
            y="total",
            ax=axes[1, 0],
            color="darkorange",
        )
        axes[1, 0].set_title("카테고리별 총매출 (날짜 컬럼 없음)")
        axes[1, 0].set_xlabel("카테고리")
        axes[1, 0].set_ylabel("총매출")
        axes[1, 0].tick_params(axis="x", rotation=30)

    numeric = df.select_dtypes(include=np.number)
    identifier_columns = [
        column
        for column in numeric.columns
        if str(column).lower() == "id" or str(column).lower().endswith("_id")
    ]
    numeric = numeric.drop(columns=identifier_columns)
    if numeric.empty:
        raise ValueError("식별자를 제외하면 상관 히트맵을 만들 수치형 컬럼이 없습니다.")
    sns.heatmap(
        numeric.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("수치형 변수 상관관계")

    fig.suptitle("Sales 100K 탐색적 데이터 분석", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[2. EDA 시각화 저장]\n{output_path}")


def choose_two_regions(df: pd.DataFrame) -> tuple[object, object]:
    """서울·부산 또는 표본 수가 큰 두 지역을 선택"""
    regions = df["region"].dropna()
    unique = set(regions.unique())
    preferred_pairs = (("서울", "부산"), ("Seoul", "Busan"))
    for first, second in preferred_pairs:
        if first in unique and second in unique:
            return first, second

    counts = regions.value_counts()
    if len(counts) < 2:
        raise ValueError("t-test를 수행하려면 region에 두 개 이상의 그룹이 필요합니다.")
    return counts.index[0], counts.index[1]


def run_t_test(df: pd.DataFrame) -> None:
    """두 지역의 평균 매출 차이에 대해 Welch 독립표본 t-test를 수행"""
    first_region, second_region = choose_two_regions(df)
    first = df.loc[df["region"] == first_region, "amount"].dropna()
    second = df.loc[df["region"] == second_region, "amount"].dropna()
    if len(first) < 2 or len(second) < 2:
        raise ValueError("t-test를 수행하려면 각 지역에 두 건 이상의 매출이 필요합니다.")

    result = stats.ttest_ind(first, second, equal_var=False)
    if np.isnan(result.statistic) or np.isnan(result.pvalue):
        raise ValueError("데이터 분산이 없어 t-test 결과를 계산할 수 없습니다.")
    print("\n[3. 독립표본 t-test]")
    print(f"비교 지역: {first_region}({len(first):,}건) vs {second_region}({len(second):,}건)")
    print(f"평균 매출: {first.mean():,.2f} vs {second.mean():,.2f}")
    print(f"t 통계량: {result.statistic:.6f}")
    print(f"p-value: {result.pvalue:.6g}")
    if result.pvalue < SIGNIFICANCE_LEVEL:
        print("해석: p < 0.05이므로 두 지역의 평균 매출 차이가 통계적으로 유의합니다.")
    else:
        print("해석: p >= 0.05이므로 평균 매출 차이가 있다고 판단할 근거가 부족합니다.")


def run_chi_square_test(df: pd.DataFrame) -> None:
    """region과 category의 독립성과 관계 크기를 카이제곱·Cramér's V로 확인"""
    contingency = pd.crosstab(df["region"], df["category"])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        raise ValueError("카이제곱 검정에는 각 변수에 두 개 이상의 범주가 필요합니다.")

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    if np.isnan(chi2) or np.isnan(p_value):
        raise ValueError("카이제곱 검정 결과를 계산할 수 없습니다.")

    sample_size = int(contingency.to_numpy().sum())
    min_dimension = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = float(np.sqrt(chi2 / (sample_size * min_dimension)))
    small_expected_ratio = float((expected < 5).mean())

    print("\n[4. 카이제곱 독립성 검정: region × category]")
    print("분할표:")
    print(contingency)
    print(f"카이제곱 통계량: {chi2:.6f}")
    print(f"자유도: {dof}")
    print(f"p-value: {p_value:.6g}")
    print(f"Cramér's V: {cramers_v:.6f}")

    if p_value < SIGNIFICANCE_LEVEL:
        print("통계적 해석: p < 0.05이므로 지역과 카테고리가 완전히 독립이라는 귀무가설을 기각합니다.")
    else:
        print("통계적 해석: p >= 0.05이므로 지역과 카테고리의 연관성을 판단할 근거가 부족합니다.")

    if cramers_v < 0.05:
        effect_description = "매우 작습니다"
    elif cramers_v < 0.1:
        effect_description = "작습니다"
    elif cramers_v < 0.3:
        effect_description = "중간 정도입니다"
    else:
        effect_description = "큽니다"
    print(f"효과 크기 해석: Cramér's V가 {cramers_v:.4f} 입니다.")

    if p_value < SIGNIFICANCE_LEVEL and cramers_v < 0.05:
        print(
            "종합 인사이트: 표본이 매우 커서 지역과 카테고리 사이의 작은 분포 차이가 "
            "통계적으로 검출되었지만, 관계의 실질적인 크기는 매우 작습니다. "
            "실무적으로는 거의 독립에 가깝다고 볼 수 있습니다."
        )
    elif p_value < SIGNIFICANCE_LEVEL:
        print("종합 인사이트: 통계적 연관성이 있으며 효과 크기도 함께 고려해야 합니다.")
    else:
        print("종합 인사이트: 현재 표본에서는 두 변수의 연관성이 통계적으로 확인되지 않았습니다.")

    if small_expected_ratio > 0.2:
        warnings.warn(
            "기대 빈도가 5보다 작은 셀이 20%를 초과해 카이제곱 근사가 부정확할 수 있습니다.",
            stacklevel=2,
        )


def build_pipeline() -> Pipeline:
    """범주형 결측치 처리·원핫 인코딩과 Ridge 회귀 Pipeline을 구성"""
    categorical_features = ["region", "category"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("category", categorical_transformer, categorical_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def train_evaluate_and_save_pipeline(df: pd.DataFrame, model_path: Path) -> None:
    """Pipeline의 훈련·예측·평가·저장 및 재로딩 결과를 확인"""
    features = df[["region", "category"]]
    target = df["amount"]
    if len(df) < 10:
        raise ValueError("Pipeline 학습과 평가에는 최소 10행이 필요합니다.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    model = build_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    joblib.dump(model, model_path)
    loaded_model = joblib.load(model_path)
    loaded_predictions = loaded_model.predict(x_test)
    if not np.allclose(predictions, loaded_predictions):
        raise ValueError("저장 전후 Pipeline의 예측 결과가 일치하지 않습니다.")

    print("\n[5. sklearn Pipeline]")
    print(f"학습 데이터: {len(x_train):,}행 / 평가 데이터: {len(x_test):,}행")
    print(f"R² score: {r2:.6f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"모델 저장 및 재로딩 확인: {model_path}")


def create_plotly_chart(summary: pd.DataFrame, output_path: Path) -> None:
    """지역·카테고리별 총매출 인터랙티브 막대 차트를 HTML로 저장"""
    figure = px.bar(
        summary,
        x="region",
        y="total",
        color="category",
        barmode="group",
        hover_data={"mean": ":,.2f", "count": True, "total": ":,.2f"},
        labels={
            "region": "지역",
            "category": "카테고리",
            "total": "총매출",
            "mean": "평균 매출",
            "count": "거래 건수",
        },
        title="지역·카테고리별 총매출",
    )
    figure.update_layout(template="plotly_white")
    figure.write_html(output_path, include_plotlyjs=True)
    print(f"\n[6. Plotly 인터랙티브 차트 저장]\n{output_path}")


def main() -> None:
    """정제·시각화·검정·Pipeline·결과 저장을 순서대로 수행"""
    args = parse_args()
    csv_path = validate_input(args.csv_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clean = load_and_clean(csv_path)
    summary = aggregate_sales(clean)
    create_eda_visualization(clean, output_dir / "eda_2x2.png")
    run_t_test(clean)
    run_chi_square_test(clean)
    train_evaluate_and_save_pipeline(clean, output_dir / "sales_pipeline.joblib")
    create_plotly_chart(summary, output_dir / "sales_by_region_category.html")
    print("\n실습 4의 모든 작업을 완료했습니다.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, TypeError, pd.errors.ParserError) as error:
        raise SystemExit(f"오류: {error}") from error
    except OSError as error:
        raise SystemExit(f"파일 처리 오류: {error}") from error
