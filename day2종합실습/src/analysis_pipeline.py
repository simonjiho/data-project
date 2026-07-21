"""Adult Census 데이터의 준비·분석·모델링·보고서 생성 기능."""

import time
from pathlib import Path
from urllib import error, request

import joblib
import matplotlib
from jinja2 import Environment, FileSystemLoader, StrictUndefined

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import seaborn as sns
from matplotlib import font_manager
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

NUMERIC_COLUMNS = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

BOXPLOT_COLUMNS = [
    "age",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

TARGET_COLUMN = "income"
POSITIVE_CLASS = ">50K"
RANDOM_STATE = 42
SIGNIFICANCE_LEVEL = 0.05
REPORT_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"
ADULT_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)


def download_input(data_path: Path) -> Path:
    """UCI 공식 URL에서 Adult Census 원본 파일을 안전하게 다운로드"""
    data_path = data_path.expanduser().resolve()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = data_path.with_name(f".{data_path.name}.download")
    download_request = request.Request(
        ADULT_DATA_URL,
        headers={"User-Agent": "data-project/1.0"},
    )

    print(f"입력 파일이 없어 UCI에서 다운로드합니다: {ADULT_DATA_URL}")
    try:
        with request.urlopen(download_request, timeout=30) as response:
            content = response.read()

        first_row = next((row for row in content.splitlines() if row.strip()), b"")
        if first_row.count(b",") != len(COLUMNS) - 1:
            raise ValueError(
                "다운로드한 파일이 Adult Census 원본 형식(15개 컬럼)과 다릅니다."
            )

        temporary_path.write_bytes(content)
        temporary_path.replace(data_path)
    except (error.URLError, TimeoutError) as download_error:
        temporary_path.unlink(missing_ok=True)
        raise OSError(
            f"UCI Adult Census 데이터 다운로드에 실패했습니다: {ADULT_DATA_URL}"
        ) from download_error
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"입력 파일 저장 완료: {data_path}")
    return data_path


def validate_input(data_path: Path) -> Path:
    """Adult Census 입력 파일을 준비하고 비어 있지 않은지 검증"""
    data_path = data_path.expanduser().resolve()
    if not data_path.is_file():
        download_input(data_path)
    if data_path.stat().st_size == 0:
        raise ValueError(f"입력 파일이 비어 있습니다: {data_path}")
    return data_path


def load_with_pandas(data_path: Path) -> pd.DataFrame:
    """Adult Census 데이터를 Pandas DataFrame으로 로딩"""
    return pd.read_csv(
        data_path,
        header=None,
        names=COLUMNS,
        na_values="?",
        skipinitialspace=True,
    )


def load_with_polars(data_path: Path) -> pl.DataFrame:
    """Adult Census 데이터를 Polars DataFrame으로 로딩하고 자료형을 변환"""
    frame = pl.read_csv(
        data_path,
        has_header=False,
        new_columns=COLUMNS,
        null_values=["?", " ?"],
        schema_overrides={column: pl.String for column in COLUMNS},
    )
    return frame.with_columns(
        pl.col(pl.String).str.strip_chars(),
    ).with_columns(
        pl.col(NUMERIC_COLUMNS + ["fnlwgt"]).cast(pl.Int64, strict=False),
    )


def clean_pandas(frame: pd.DataFrame) -> pd.DataFrame:
    """문자열 공백과 중복을 제거하고 필수값·자료형을 정리"""
    clean = frame.copy()
    text_columns = clean.select_dtypes(include=["object", "string"]).columns
    clean[text_columns] = clean[text_columns].apply(
        lambda column: column.str.strip()
    )
    clean = clean.replace("?", np.nan)
    for column in NUMERIC_COLUMNS + ["fnlwgt"]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=[TARGET_COLUMN]).drop_duplicates().reset_index(drop=True)
    return clean


def clean_polars(frame: pl.DataFrame) -> pl.DataFrame:
    """Polars에서 필수값과 중복을 제거"""
    return frame.filter(pl.col(TARGET_COLUMN).is_not_null()).unique(maintain_order=True)


def compare_loaders(data_path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """Pandas와 Polars 로딩·정제 결과 및 실행 시간을 비교"""
    pandas_start = time.perf_counter()
    pandas_frame = clean_pandas(load_with_pandas(data_path))
    pandas_seconds = time.perf_counter() - pandas_start

    polars_start = time.perf_counter()
    polars_frame = clean_polars(load_with_polars(data_path))
    polars_seconds = time.perf_counter() - polars_start

    if pandas_frame.shape != polars_frame.shape:
        raise ValueError(
            "Pandas와 Polars 정제 결과의 크기가 다릅니다: "
            f"{pandas_frame.shape} vs {polars_frame.shape}"
        )

    print("\n[1. Pandas·Polars 로딩 및 정제 비교]")
    print(f"Pandas: {pandas_frame.shape}, {pandas_seconds:.6f}초")
    print(f"Polars: {polars_frame.shape}, {polars_seconds:.6f}초")
    print("두 도구의 정제 후 행·열 수가 같습니다.")
    print(
        "공통 정제: 컬럼명 지정, 문자열 공백 제거, '?' 결측 변환, "
        "수치형 변환, income 결측 제거, 중복 행 제거"
    )
    return pandas_frame, {
        "Pandas": pandas_seconds,
        "Polars": polars_seconds,
    }


def print_eda(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """결측치·기술통계·상관계수를 계산하고 출력"""
    missing = frame.isna().sum().sort_values(ascending=False)
    description = frame[NUMERIC_COLUMNS].describe().T
    correlation = frame[NUMERIC_COLUMNS].corr()

    print("\n[2. 기본 EDA]")
    print(f"데이터 크기: {frame.shape[0]:,}행 × {frame.shape[1]:,}열")
    print(f"중복 제거 후 중복 행: {frame.duplicated().sum():,}건")
    print("\n컬럼별 결측치:")
    print(missing[missing > 0] if missing.sum() else "결측치 없음")
    print("\n기술통계:")
    print(description)
    print("\n상관계수:")
    print(correlation.round(3))
    return description, correlation


def create_data_quality_visuals(
    frame: pd.DataFrame,
    missing_path: Path,
    boxplot_path: Path,
) -> pd.DataFrame:
    """결측치 분포와 수치형 변수 boxplot을 저장하고 IQR 요약을 반환"""
    configure_korean_font()
    missing = frame.isna().sum()
    missing = missing[missing > 0].sort_values()
    missing_rates = missing / len(frame) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(missing.index, missing.values, color="#457B9D")
    ax.bar_label(
        bars,
        labels=[f"{count:,}건 ({missing_rates[column]:.1f}%)" for column, count in missing.items()],
        padding=5,
    )
    ax.set_title("컬럼별 결측치", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("결측 건수")
    ax.set_ylabel("컬럼")
    ax.set_xlim(0, max(missing.max() * 1.28, 1))
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(missing_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    labels = {
        "age": "나이",
        "education_num": "교육수준",
        "capital_gain": "자본이득",
        "capital_loss": "자본손실",
        "hours_per_week": "주당 근무시간",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, column in zip(axes.flat, BOXPLOT_COLUMNS):
        sns.boxplot(
            data=frame,
            x=column,
            ax=ax,
            color="#76A9C5",
            flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.25},
        )
        ax.set_title(labels[column])
        ax.set_xlabel("")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("수치형 변수 Boxplot과 이상치 후보", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for column in BOXPLOT_COLUMNS:
        values = frame[column].dropna()
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int((~values.between(lower, upper)).sum())
        rows.append(
            {
                "column": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower": lower,
                "upper": upper,
                "outlier_count": outlier_count,
                "outlier_rate": outlier_count / len(values) * 100,
            }
        )
    summary = pd.DataFrame(rows)
    print(f"\n[3. 데이터 품질 시각화 저장]\n{missing_path}\n{boxplot_path}")
    print("\nIQR 기준 이상치 후보:")
    print(summary.round(3).to_string(index=False))
    return summary


def exclude_capital_outliers(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """0을 보존하고 양수 자본손익 내부의 IQR 극단값을 제외"""
    masks: dict[str, pd.Series] = {}
    thresholds: dict[str, tuple[float, float]] = {}
    for column in ("capital_gain", "capital_loss"):
        positive = frame.loc[frame[column] > 0, column]
        q1 = float(positive.quantile(0.25))
        q3 = float(positive.quantile(0.75))
        iqr = q3 - q1
        lower = max(0.0, q1 - 1.5 * iqr)
        upper = q3 + 1.5 * iqr
        thresholds[column] = (lower, upper)
        masks[column] = (frame[column] > 0) & ~frame[column].between(lower, upper)

    excluded_mask = masks["capital_gain"] | masks["capital_loss"]
    analysis_frame = frame.loc[~excluded_mask].copy().reset_index(drop=True)
    excluded = int(excluded_mask.sum())
    information: dict[str, float | int] = {
        "gain_lower": thresholds["capital_gain"][0],
        "gain_upper": thresholds["capital_gain"][1],
        "loss_lower": thresholds["capital_loss"][0],
        "loss_upper": thresholds["capital_loss"][1],
        "gain_excluded": int(masks["capital_gain"].sum()),
        "loss_excluded": int(masks["capital_loss"].sum()),
        "before_count": len(frame),
        "after_count": len(analysis_frame),
        "excluded_count": excluded,
        "excluded_rate": excluded / len(frame) * 100,
    }
    if TARGET_COLUMN in frame.columns and excluded > 0:
        # 극단 자본손익 제거가 특정 소득 그룹을 편중 제거하는지 진단
        information["excluded_high_income_rate"] = float(
            (frame.loc[excluded_mask, TARGET_COLUMN] == POSITIVE_CLASS).mean() * 100
        )
        information["kept_high_income_rate"] = float(
            (analysis_frame[TARGET_COLUMN] == POSITIVE_CLASS).mean() * 100
        )

    print("\n[4. 자본손익 IQR 이상치 제외]")
    print(
        "제외 기준: 자본손익의 0은 유지하고, 양수값에서 계산한 "
        "Q1 - 1.5×IQR 미만 또는 Q3 + 1.5×IQR 초과 행 제외"
    )
    print(
        f"제외 전 {len(frame):,}행 → 제외 후 {len(analysis_frame):,}행 "
        f"({excluded:,}행, {information['excluded_rate']:.2f}% 제외)"
    )
    if "excluded_high_income_rate" in information:
        print(
            f"제외 행의 >50K 비율: {information['excluded_high_income_rate']:.1f}% "
            f"(유지 데이터: {information['kept_high_income_rate']:.1f}%)"
        )
    return analysis_frame, information


def create_capital_outlier_comparison(
    before: pd.DataFrame,
    after: pd.DataFrame,
    output_path: Path,
) -> None:
    """자본손익 양수값의 이상치 제외 전후 Boxplot을 저장"""
    sns.set_theme(style="whitegrid")
    configure_korean_font()
    labels = {
        "capital_gain": "자본이득 양수값",
        "capital_loss": "자본손실 양수값",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, column in zip(axes, ("capital_gain", "capital_loss")):
        comparison = pd.concat(
            [
                pd.DataFrame(
                    {"구분": "제외 전", "값": before.loc[before[column] > 0, column]}
                ),
                pd.DataFrame(
                    {"구분": "제외 후", "값": after.loc[after[column] > 0, column]}
                ),
            ],
            ignore_index=True,
        )
        sns.boxplot(data=comparison, x="구분", y="값", hue="구분", ax=ax)
        ax.set_yscale("log")
        ax.set_title(labels[column])
        ax.set_xlabel("")
        ax.set_ylabel("값 (로그 스케일)")
    fig.suptitle(
        "자본손익 IQR 이상치 제외 전후 비교",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def configure_korean_font() -> None:
    """현재 운영체제에 실제 설치된 한글 글꼴만 Matplotlib에 적용"""
    mac_font_path = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    if mac_font_path.is_file():
        font_manager.fontManager.addfont(str(mac_font_path))
        font_name = font_manager.FontProperties(fname=str(mac_font_path)).get_name()
    else:
        installed = {font.name for font in font_manager.fontManager.ttflist}
        candidates = ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR")
        font_name = next((name for name in candidates if name in installed), "DejaVu Sans")
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False


def create_categorical_countplots(frame: pd.DataFrame, output_path: Path) -> None:
    """주요 범주형 변수와 소득 그룹의 빈도를 Count bar plot으로 저장"""
    sns.set_theme(style="whitegrid")
    configure_korean_font()
    plot_frame = frame.copy()
    columns = {
        "workclass": "근로 형태",
        "marital_status": "혼인 상태",
        "race": "인종",
        "sex": "성별",
    }
    for column in columns:
        plot_frame[column] = plot_frame[column].fillna("결측")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    palette = {"<=50K": "#76A9C5", ">50K": "#1D3557"}
    legend_handles = None
    legend_labels = None
    for ax, (column, title) in zip(axes.flat, columns.items()):
        order = plot_frame[column].value_counts().index
        sns.countplot(
            data=plot_frame,
            y=column,
            hue=TARGET_COLUMN,
            hue_order=["<=50K", ">50K"],
            order=order,
            palette=palette,
            ax=ax,
        )
        ax.set_title(f"{title}별 소득 그룹 건수", fontsize=14, fontweight="bold")
        ax.set_xlabel("건수")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    fig.legend(
        legend_handles,
        legend_labels,
        title="소득 그룹",
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(
        "주요 범주형 변수 Count bar plot",
        fontsize=19,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_correlation_heatmap(correlation: pd.DataFrame, output_path: Path) -> None:
    """수치형 변수 상관계수를 Seaborn 히트맵으로 저장"""
    sns.set_theme(style="white")
    configure_korean_font()
    labels = {
        "age": "나이",
        "education_num": "교육수준",
        "capital_gain": "자본이득",
        "capital_loss": "자본손실",
        "hours_per_week": "주당 근무시간",
    }
    display_correlation = correlation.rename(index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        display_correlation,
        annot=True,
        cmap="RdBu_r",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "상관계수", "shrink": 0.82},
        ax=ax,
    )
    ax.set_title("수치형 변수 상관관계", fontsize=16, fontweight="bold", pad=16)
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[5. 상관계수 히트맵 저장]\n{output_path}")


def create_plotly_chart(frame: pd.DataFrame, output_path: Path) -> None:
    """직업별 고소득 비율을 Plotly 인터랙티브 차트로 저장"""
    chart_data = (
        frame.dropna(subset=["occupation"])
        .assign(is_high_income=lambda data: (data[TARGET_COLUMN] == POSITIVE_CLASS).astype(int))
        .groupby("occupation", as_index=False)
        .agg(high_income_rate=("is_high_income", "mean"), count=("is_high_income", "size"))
        .query("count >= 100")
        .sort_values("high_income_rate", ascending=False)
    )
    chart_data["high_income_rate"] *= 100
    chart_data["income_rate_band"] = pd.cut(
        chart_data["high_income_rate"],
        bins=[0, 15, 25, 40, 100],
        labels=["15% 미만", "15~25%", "25~40%", "40% 이상"],
        include_lowest=True,
    )

    figure = px.bar(
        chart_data,
        x="occupation",
        y="high_income_rate",
        color="income_rate_band",
        text="high_income_rate",
        hover_data={"count": ":,", "high_income_rate": ":.2f"},
        color_discrete_map={
            "40% 이상": "#1D3557",
            "25~40%": "#457B9D",
            "15~25%": "#76A9C5",
            "15% 미만": "#B8C4CE",
        },
        category_orders={
            "occupation": chart_data["occupation"].tolist(),
            "income_rate_band": ["40% 이상", "25~40%", "15~25%", "15% 미만"],
        },
        labels={
            "occupation": "직업",
            "high_income_rate": "고소득 비율(%)",
            "count": "표본 수",
            "income_rate_band": "고소득 비율 구간",
        },
        title="직업별 고소득(>50K) 비율",
    )
    figure.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        marker_line_width=0,
        cliponaxis=False,
    )
    figure.update_layout(
        template="plotly_white",
        xaxis_tickangle=-35,
        yaxis_range=[0, chart_data["high_income_rate"].max() * 1.12],
        legend_title_text="고소득 비율",
        hoverlabel=dict(bgcolor="white"),
        bargap=0.24,
    )
    figure.write_html(output_path, include_plotlyjs=True)
    print(f"\n[6. Plotly 인터랙티브 차트 저장]\n{output_path}")


def format_p_value(p_value: float) -> str:
    """매우 작은 p-value를 0으로 오해하지 않도록 표시"""
    return "< 1e-300" if p_value == 0 else f"{p_value:.6g}"


def run_normality_analysis(
    frame: pd.DataFrame,
    output_path: Path,
) -> dict[str, dict[str, float | str]]:
    """소득 그룹별 정규성을 Shapiro-Wilk 검정과 Boxplot·Q-Q plot으로 진단"""
    groups = {
        ">50K": frame.loc[
            frame[TARGET_COLUMN] == POSITIVE_CLASS,
            "hours_per_week",
        ].dropna(),
        "<=50K": frame.loc[
            frame[TARGET_COLUMN] != POSITIVE_CLASS,
            "hours_per_week",
        ].dropna(),
    }
    results: dict[str, dict[str, float | str]] = {}

    sns.set_theme(style="whitegrid")
    configure_korean_font()
    fig = plt.figure(figsize=(14, 11))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.1])
    boxplot_ax = fig.add_subplot(grid[0, :])
    qq_axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    sns.boxplot(
        data=frame,
        x="hours_per_week",
        y=TARGET_COLUMN,
        order=[">50K", "<=50K"],
        hue=TARGET_COLUMN,
        hue_order=[">50K", "<=50K"],
        palette={">50K": "#457B9D", "<=50K": "#E07A5F"},
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#1D3557",
            "markersize": 7,
        },
        flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.25},
        ax=boxplot_ax,
    )
    boxplot_ax.set_title(
        "소득 그룹별 주당 근무시간 Boxplot (◇ 평균)",
        fontsize=15,
        fontweight="bold",
    )
    boxplot_ax.set_xlabel("주당 근무시간")
    boxplot_ax.set_ylabel("소득 그룹")
    boxplot_ax.set_xlim(0, 100)
    boxplot_ax.grid(axis="x", alpha=0.25)

    for column_index, (group_name, values) in enumerate(groups.items()):
        sample_size = min(len(values), 5_000)
        sample = values.sample(sample_size, random_state=RANDOM_STATE)
        shapiro = stats.shapiro(sample)
        normality = (
            "정규성 가설 기각 못함"
            if shapiro.pvalue >= SIGNIFICANCE_LEVEL
            else "정규성 가설 기각(비정규)"
        )
        results[group_name] = {
            "sample_size": sample_size,
            "statistic": float(shapiro.statistic),
            "p_value": float(shapiro.pvalue),
            "p_value_text": format_p_value(float(shapiro.pvalue)),
            "interpretation": normality,
        }

        qq_ax = qq_axes[column_index]
        stats.probplot(sample, dist="norm", plot=qq_ax)
        qq_ax.set_title(
            f"{group_name} Q-Q plot (표본 {sample_size:,}개)"
        )
        qq_ax.set_xlabel("이론적 정규분포 분위수")
        qq_ax.set_ylabel("관측값")

    fig.suptitle(
        "소득 그룹별 주당 근무시간 정규성 진단",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[7. 정규성 검정과 Q-Q plot 저장]\n{output_path}")
    for group_name, result in results.items():
        print(
            f"{group_name}: Shapiro-Wilk W={result['statistic']:.6f}, "
            f"p-value={result['p_value_text']} ({result['interpretation']}, "
            f"표본 {result['sample_size']:,}개)"
        )
    return results


def run_rank_sum_test(frame: pd.DataFrame) -> dict[str, float | str]:
    """독립된 두 소득 그룹을 Mann-Whitney U(Wilcoxon rank-sum)로 비교"""
    high = frame.loc[frame[TARGET_COLUMN] == POSITIVE_CLASS, "hours_per_week"].dropna()
    low = frame.loc[frame[TARGET_COLUMN] != POSITIVE_CLASS, "hours_per_week"].dropna()
    result = stats.mannwhitneyu(high, low, alternative="two-sided")
    rank_biserial = 2 * float(result.statistic) / (len(high) * len(low)) - 1
    magnitude = abs(rank_biserial)
    if magnitude < 0.1:
        effect = "매우 작은 편"
    elif magnitude < 0.3:
        effect = "작은 편"
    elif magnitude < 0.5:
        effect = "중간 정도"
    else:
        effect = "큰 편"
    interpretation = (
        "p < 0.05이므로 두 소득 그룹의 주당 근무시간 분포가 같다는 "
        "귀무가설을 기각합니다."
        if result.pvalue < SIGNIFICANCE_LEVEL
        else "p >= 0.05이므로 두 그룹의 분포 차이를 판단할 근거가 부족합니다."
    )
    p_value_text = format_p_value(float(result.pvalue))

    print("\n[8. Mann-Whitney U (Wilcoxon rank-sum) 검정]")
    print(f">50K 중앙값: {high.median():.2f}시간")
    print(f"<=50K 중앙값: {low.median():.2f}시간")
    print(f"U 통계량: {result.statistic:,.0f}")
    print(f"p-value: {p_value_text}")
    print(f"순위 양분 상관계수: {rank_biserial:.4f} ({effect})")
    print(f"해석: {interpretation}")
    return {
        "high_median": float(high.median()),
        "low_median": float(low.median()),
        "u_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "p_value_text": p_value_text,
        "rank_biserial": rank_biserial,
        "effect": effect,
        "interpretation": interpretation,
    }


def run_t_test(frame: pd.DataFrame) -> dict[str, float | str]:
    """두 소득 그룹의 주당 근무시간 차이를 Welch t-test로 검정"""
    high = frame.loc[frame[TARGET_COLUMN] == POSITIVE_CLASS, "hours_per_week"].dropna()
    low = frame.loc[frame[TARGET_COLUMN] != POSITIVE_CLASS, "hours_per_week"].dropna()
    result = stats.ttest_ind(high, low, equal_var=False)
    if np.isnan(result.statistic) or np.isnan(result.pvalue):
        raise ValueError("t-test 결과를 계산할 수 없습니다.")

    mean_difference = float(high.mean() - low.mean())
    high_count, low_count = len(high), len(low)
    high_variance = float(high.var(ddof=1))
    low_variance = float(low.var(ddof=1))
    standard_error = float(np.sqrt(high_variance / high_count + low_variance / low_count))
    # Welch-Satterthwaite 자유도
    dof = standard_error**4 / (
        (high_variance / high_count) ** 2 / (high_count - 1)
        + (low_variance / low_count) ** 2 / (low_count - 1)
    )
    t_critical = float(stats.t.ppf(1 - SIGNIFICANCE_LEVEL / 2, dof))
    ci_lower = mean_difference - t_critical * standard_error
    ci_upper = mean_difference + t_critical * standard_error
    pooled_std = float(
        np.sqrt(
            ((high_count - 1) * high_variance + (low_count - 1) * low_variance)
            / (high_count + low_count - 2)
        )
    )
    cohens_d = mean_difference / pooled_std
    d_magnitude = abs(cohens_d)
    if d_magnitude < 0.2:
        effect = "매우 작은 편"
    elif d_magnitude < 0.5:
        effect = "작은 편"
    elif d_magnitude < 0.8:
        effect = "중간 정도"
    else:
        effect = "큰 편"
    if result.pvalue < SIGNIFICANCE_LEVEL:
        interpretation = (
            "p < 0.05이므로 두 소득 그룹의 평균 주당 근무시간 차이는 "
            "통계적으로 유의합니다."
        )
    else:
        interpretation = (
            "p >= 0.05이므로 두 소득 그룹의 평균 주당 근무시간 차이가 "
            "있다고 판단할 근거가 부족합니다."
        )

    print("\n[9. Welch 독립표본 t-test]")
    print(f">50K 평균: {high.mean():.2f}시간")
    print(f"<=50K 평균: {low.mean():.2f}시간")
    print(f"평균 차이: {mean_difference:.2f}시간 (95% CI [{ci_lower:.2f}, {ci_upper:.2f}])")
    print(f"t 통계량: {result.statistic:.6f} (자유도 {dof:,.1f})")
    p_value_text = format_p_value(float(result.pvalue))
    print(f"p-value: {p_value_text}")
    print(f"Cohen's d: {cohens_d:.4f} ({effect})")
    print(f"해석: {interpretation}")
    return {
        "high_mean": float(high.mean()),
        "low_mean": float(low.mean()),
        "mean_difference": mean_difference,
        "t_statistic": float(result.statistic),
        "dof": float(dof),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cohens_d": cohens_d,
        "effect": effect,
        "p_value": float(result.pvalue),
        "p_value_text": p_value_text,
        "interpretation": interpretation,
    }


def calculate_cramers_v(contingency: pd.DataFrame) -> tuple[float, float, float, int]:
    """분할표의 카이제곱 통계량·p-value·Cramér's V·자유도를 계산"""
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
    sample_size = int(contingency.to_numpy().sum())
    min_dimension = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    if sample_size == 0 or min_dimension == 0:
        raise ValueError("Cramér's V 계산에는 각 변수에 두 개 이상의 범주가 필요합니다.")
    cramers_v = float(np.sqrt(chi2 / (sample_size * min_dimension)))
    return float(chi2), float(p_value), cramers_v, int(dof)


def run_chi_square_test(frame: pd.DataFrame) -> dict[str, float | str]:
    """교육수준과 소득 그룹의 독립성 및 관계 크기를 검정"""
    contingency = pd.crosstab(frame["education"], frame[TARGET_COLUMN])
    chi2, p_value, cramers_v, dof = calculate_cramers_v(contingency)
    p_value_text = format_p_value(p_value)

    # 카이제곱 근사 가정 점검: 기대빈도 5 미만 셀 확인
    expected = stats.contingency.expected_freq(contingency)
    min_expected = float(expected.min())
    small_cell_count = int((expected < 5).sum())
    total_cell_count = int(expected.size)

    if p_value < SIGNIFICANCE_LEVEL:
        significance = (
            "p < 0.05이므로 교육수준과 소득 그룹이 완전히 독립이라는 "
            "귀무가설을 기각합니다."
        )
    else:
        significance = (
            "p >= 0.05이므로 교육수준과 소득 그룹의 연관성을 판단할 "
            "근거가 부족합니다."
        )

    # Cramér's V의 효과크기 기준은 표의 자유도 구조 k = min(행-1, 열-1)에 따라 달라짐
    # (Cohen w 기준 0.1/0.3/0.5를 sqrt(k)로 나눠 보정)
    min_dimension = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    small_threshold = 0.1 / float(np.sqrt(min_dimension))
    medium_threshold = 0.3 / float(np.sqrt(min_dimension))
    large_threshold = 0.5 / float(np.sqrt(min_dimension))
    if cramers_v < small_threshold:
        effect = "매우 작은 편"
    elif cramers_v < medium_threshold:
        effect = "작은 편"
    elif cramers_v < large_threshold:
        effect = "중간 정도"
    else:
        effect = "큰 편"

    print("\n[10. 카이제곱 독립성 검정: education × income]")
    print(f"카이제곱 통계량: {chi2:.6f}")
    print(f"자유도: {dof}")
    print(f"p-value: {p_value_text}")
    print(f"Cramér's V: {cramers_v:.6f}")
    print(
        f"기대빈도 점검: 최소 {min_expected:.1f}, "
        f"5 미만 셀 {small_cell_count}/{total_cell_count}개"
    )
    print(f"통계적 해석: {significance}")
    return {
        "chi2": chi2,
        "p_value": p_value,
        "p_value_text": p_value_text,
        "cramers_v": cramers_v,
        "dof": dof,
        "min_expected": min_expected,
        "small_cell_count": small_cell_count,
        "total_cell_count": total_cell_count,
        "min_dimension": min_dimension,
        "small_threshold": small_threshold,
        "medium_threshold": medium_threshold,
        "large_threshold": large_threshold,
        "significance": significance,
        "effect": effect,
    }


def build_model_pipeline(model: object) -> Pipeline:
    """공통 수치형·범주형 전처리와 전달받은 분류 모델을 Pipeline으로 구성"""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_COLUMNS),
            ("category", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def get_model_candidates() -> dict[str, object]:
    """서로 다른 학습 원리를 가진 비교용 분류 모델을 생성"""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def train_evaluate_models(
    frame: pd.DataFrame,
    model_path: Path,
    roc_curve_path: Path,
    metrics_chart_path: Path,
) -> tuple[dict[str, dict[str, float]], str, dict[str, float]]:
    """여러 분류 Pipeline을 같은 조건에서 비교하고 최고 모델을 저장"""
    features = frame[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    target = (frame[TARGET_COLUMN] == POSITIVE_CLASS).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model_results: dict[str, dict[str, float]] = {}
    trained_pipelines: dict[str, Pipeline] = {}
    predictions_by_model: dict[str, np.ndarray] = {}
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for model_name, estimator in get_model_candidates().items():
        pipeline = build_model_pipeline(estimator)
        started_at = time.perf_counter()
        pipeline.fit(x_train, y_train)
        elapsed = time.perf_counter() - started_at
        predictions = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)

        model_results[model_name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1": float(f1_score(y_test, predictions)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "pr_auc": float(average_precision_score(y_test, probabilities)),
            "training_seconds": elapsed,
        }
        trained_pipelines[model_name] = pipeline
        predictions_by_model[model_name] = predictions
        roc_curves[model_name] = (false_positive_rate, true_positive_rate)

    best_model_name = max(
        model_results,
        key=lambda name: model_results[name]["roc_auc"],
    )
    best_pipeline = trained_pipelines[best_model_name]

    configure_korean_font()
    fig, ax = plt.subplots(figsize=(8, 7))
    model_colors = {
        "Logistic Regression": "#457B9D",
        "Random Forest": "#E07A5F",
        "HistGradientBoosting": "#1D3557",
    }
    roc_legend_order = sorted(
        roc_curves,
        key=lambda name: model_results[name]["roc_auc"],
        reverse=True,
    )
    for model_name in roc_legend_order:
        false_positive_rate, true_positive_rate = roc_curves[model_name]
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            color=model_colors[model_name],
            linewidth=2.3,
            label=f"{model_name} (AUC = {model_results[model_name]['roc_auc']:.4f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9AA5B1", label="무작위 분류 (AUC = 0.5)")
    best_false_positive_rate, best_true_positive_rate = roc_curves[best_model_name]
    ax.fill_between(
        best_false_positive_rate,
        best_true_positive_rate,
        alpha=0.08,
        color=model_colors[best_model_name],
    )
    ax.set_title("Adult Income 모델별 ROC Curve", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(roc_curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    metric_names = ["Accuracy", "F1", "ROC-AUC", "PR-AUC"]
    metric_keys = ["accuracy", "f1", "roc_auc", "pr_auc"]
    x_positions = np.arange(len(metric_names))
    bar_width = 0.24
    fig, ax = plt.subplots(figsize=(11, 7))
    for index, model_name in enumerate(model_results):
        positions = x_positions + (index - 1) * bar_width
        values = [model_results[model_name][key] for key in metric_keys]
        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            color=model_colors[model_name],
            label=model_name,
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.3f}" for value in values],
            padding=4,
            fontsize=9,
            rotation=90,
        )
    ax.set_title("분류 모델 성능 비교", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("평가 지표")
    ax.set_ylabel("점수")
    ax.set_xticks(x_positions, metric_names)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(metrics_chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    joblib.dump(best_pipeline, model_path)
    loaded_pipeline = joblib.load(model_path)
    if not np.array_equal(
        predictions_by_model[best_model_name],
        loaded_pipeline.predict(x_test),
    ):
        raise ValueError("저장 전후 모델 예측 결과가 일치하지 않습니다.")

    # 불균형 데이터 해석 기준: 양성 비율과 다수 클래스 기준선
    test_positive_rate = float(y_test.mean())
    evaluation_info = {
        "train_rows": float(len(x_train)),
        "test_rows": float(len(x_test)),
        "test_positive_rate": test_positive_rate,
        "majority_accuracy": 1 - test_positive_rate,
    }

    print("\n[11. sklearn 분류 Pipeline 비교]")
    print(f"훈련 데이터: {len(x_train):,}행 / 평가 데이터: {len(x_test):,}행")
    print(
        f"평가 데이터 양성(>50K) 비율: {test_positive_rate:.1%} "
        f"(다수 클래스 기준 정확도: {1 - test_positive_rate:.4f})"
    )
    for model_name, metrics in model_results.items():
        print(
            f"{model_name}: 정확도={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, "
            f"PR-AUC={metrics['pr_auc']:.4f}, "
            f"학습={metrics['training_seconds']:.3f}초"
        )
    print(f"최종 선택 모델: {best_model_name} (ROC-AUC 기준)")
    print(f"ROC Curve 저장: {roc_curve_path}")
    print(f"평가 지표 막대그래프 저장: {metrics_chart_path}")
    print(f"모델 저장 및 재로딩 확인: {model_path}")
    return model_results, best_model_name, evaluation_info


def format_table_number(value: float, decimals: int | None = None) -> str:
    """표 셀 숫자를 천 단위 구분 기호와 함께 표기"""
    if decimals is not None:
        return f"{value:,.{decimals}f}"
    text = f"{float(value):,.3f}".rstrip("0").rstrip(".")
    return text or "0"


def format_report_number(value: float | int, decimals: int = 2) -> str:
    """보고서 본문 숫자를 지정한 소수 자릿수와 천 단위 구분으로 표기"""
    return f"{float(value):,.{decimals}f}"


def format_report_percent(value: float, decimals: int = 1) -> str:
    """0~1 비율을 지정한 소수 자릿수의 백분율로 표기"""
    return f"{float(value):.{decimals}%}"


def create_report_environment() -> Environment:
    """누락 변수를 즉시 검출하는 Jinja2 보고서 렌더링 환경 구성"""
    environment = Environment(
        loader=FileSystemLoader(str(REPORT_TEMPLATE_DIR)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    environment.filters["number"] = format_report_number
    environment.filters["percent"] = format_report_percent
    return environment


def dataframe_to_markdown(
    frame: pd.DataFrame,
    index_label: str,
    decimals: int | None = None,
) -> str:
    """수치형 DataFrame을 수치 우측 정렬 Markdown 표로 변환"""
    header = (
        f"| {index_label} | "
        + " | ".join(str(column) for column in frame.columns)
        + " |"
    )
    divider = "|---|" + "---:|" * len(frame.columns)
    rows = [
        f"| `{index}` | "
        + " | ".join(format_table_number(value, decimals) for value in row)
        + " |"
        for index, row in frame.iterrows()
    ]
    return "\n".join([header, divider, *rows])


def write_report(
    output_path: Path,
    frame: pd.DataFrame,
    load_times: dict[str, float],
    description: pd.DataFrame,
    correlation: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    exclusion: dict[str, float | int],
    normality: dict[str, dict[str, float | str]],
    rank_sum: dict[str, float | str],
    t_test: dict[str, float | str],
    chi_square: dict[str, float | str],
    model_results: dict[str, dict[str, float]],
    best_model_name: str,
    evaluation: dict[str, float],
) -> None:
    """분석 결과와 본인 의견을 Markdown 보고서로 자동 생성"""
    missing = frame.isna().sum()
    missing_counts = missing[missing > 0].sort_values(ascending=False)
    if missing_counts.empty:
        missing_table = "이상치 제외 후 데이터에 결측치가 없습니다."
    else:
        missing_table = "\n".join(
            [
                "| 컬럼 | 결측 수 | 결측 비율 |",
                "|---|---:|---:|",
                *(
                    f"| `{column}` | {int(count):,} | {count / len(frame) * 100:.2f}% |"
                    for column, count in missing_counts.items()
                ),
            ]
        )
    load_time_rows = "\n".join(
        f"| {library} | {seconds:.6f}초 |" for library, seconds in load_times.items()
    )
    description_table = dataframe_to_markdown(description, "변수")
    correlation_labels = {
        "age": "나이",
        "education_num": "교육수준",
        "capital_gain": "자본이득",
        "capital_loss": "자본손실",
        "hours_per_week": "주당 근무시간",
    }
    correlation_pairs = sorted(
        (
            (
                abs(float(correlation.loc[left, right])),
                float(correlation.loc[left, right]),
                correlation_labels[left],
                correlation_labels[right],
            )
            for left_index, left in enumerate(correlation.columns)
            for right in correlation.columns[left_index + 1 :]
        ),
        reverse=True,
    )
    strongest_correlation = correlation_pairs[0]
    correlation_insight_rows = "\n".join(
        f"- **{left} ↔ {right}**: `{value:.2f}` — "
        f"{'양의' if value > 0 else '음의'} 선형 관계"
        for _, value, left, right in correlation_pairs[:3]
    )
    best_metrics = model_results[best_model_name]
    model_rows = "\n".join(
        "| "
        f"{model_name} | {metrics['accuracy']:.4f} | {metrics['f1']:.4f} | "
        f"{metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} | "
        f"{metrics['training_seconds']:.3f}초 |"
        for model_name, metrics in model_results.items()
    )
    outlier_rows = "\n".join(
        "| "
        f"`{row.column}` | {row.q1:,.2f} | {row.q3:,.2f} | {row.iqr:,.2f} | "
        f"{row.lower:,.2f} | {row.upper:,.2f} | {int(row.outlier_count):,} | "
        f"{row.outlier_rate:.2f}% |"
        for row in outlier_summary.itertuples(index=False)
    )
    normality_rows = "\n".join(
        "| "
        f"`{group_name}` | {int(result['sample_size']):,} | "
        f"{result['statistic']:.6f} | {result['p_value_text']} | "
        f"{result['interpretation']} |"
        for group_name, result in normality.items()
    )

    # 데이터에 따라 달라지는 문장은 하드코딩하지 않고 결과값으로부터 생성
    if "excluded_high_income_rate" in exclusion:
        exclusion_bias_text = (
            f"- 제외된 행의 `>50K` 비율: {exclusion['excluded_high_income_rate']:.1f}% "
            f"(유지 데이터의 `>50K` 비율: {exclusion['kept_high_income_rate']:.1f}%)\n\n"
            "극단 자본손익은 입력 오류가 아니라 실제 고소득자에게서 흔히 나타나는 값이므로, "
            "이 제외는 무작위가 아니라 특정 소득 그룹을 편중 제거하는 선택입니다. "
            "위 비율 차이가 클수록 분석 데이터는 원본 모집단보다 고소득 극단값이 적은 쪽으로 "
            "치우칠 수 있습니다. 이후의 기술통계·검정·모델 평가는 이 점을 감안해 해석해야 합니다."
        )
    else:
        exclusion_bias_text = ""

    if rank_sum["high_median"] == rank_sum["low_median"]:
        median_note = (
            f"두 그룹의 중앙값은 {rank_sum['high_median']:.0f}시간으로 같지만 "
            "순위합 검정은 유의하며"
        )
    else:
        median_note = (
            f"두 그룹의 중앙값은 {rank_sum['high_median']:.0f}시간과 "
            f"{rank_sum['low_median']:.0f}시간으로 다르며"
        )

    rejected_groups = [
        group_name
        for group_name, result in normality.items()
        if str(result["interpretation"]).endswith("(비정규)")
    ]
    if len(rejected_groups) == len(normality):
        normality_context = "두 그룹 모두 정규성 가설이 기각되었지만 표본 수가 크므로"
        rank_sum_transition = (
            "앞의 정규성 진단에서 두 그룹 모두 정규성 가설이 기각되었으므로, "
            "분포의 정규성 가정이 필요 없는 비모수 검정을 먼저 적용했습니다."
        )
    elif rejected_groups:
        normality_context = (
            f"{', '.join(rejected_groups)} 그룹에서 정규성 가설이 기각되었지만 표본 수가 크므로"
        )
        rank_sum_transition = (
            f"앞의 정규성 진단에서 {', '.join(rejected_groups)} 그룹의 정규성 가설이 "
            "기각되었으므로, 분포의 정규성 가정이 필요 없는 비모수 검정을 먼저 적용했습니다."
        )
    else:
        normality_context = "두 그룹 모두 정규성 가설이 기각되지 않았으므로"
        rank_sum_transition = (
            "앞의 정규성 진단에서 정규성 가설이 기각되지 않았지만, "
            "분포 가정에 의존하지 않는 비모수 검정 결과도 함께 확인했습니다."
        )

    environment = create_report_environment()
    template = environment.get_template("report.md.j2")
    report = template.render(
        frame_rows=len(frame),
        frame_columns=frame.shape[1],
        random_state=RANDOM_STATE,
        load_time_rows=load_time_rows,
        exclusion=exclusion,
        outlier_rows=outlier_rows,
        exclusion_bias_text=exclusion_bias_text,
        missing_table=missing_table,
        description_table=description_table,
        correlation_insight_rows=correlation_insight_rows,
        strongest_correlation=strongest_correlation[0],
        normality_rows=normality_rows,
        rank_sum=rank_sum,
        rank_sum_transition=rank_sum_transition,
        median_note=median_note,
        t_test=t_test,
        normality_context=normality_context,
        chi_square=chi_square,
        evaluation=evaluation,
        model_rows=model_rows,
        best_model_name=best_model_name,
        best_metrics=best_metrics,
    )
    output_path.write_text(report, encoding="utf-8")
    print(f"\n[12. Markdown 보고서 자동 생성]\n{output_path}")


def run_pipeline(data_path: Path, output_dir: Path) -> None:
    """데이터 준비부터 보고서 생성까지 End-to-End 분석을 실행"""
    data_path = validate_input(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, load_times = compare_loaders(data_path)
    print_eda(frame)
    outlier_summary = create_data_quality_visuals(
        frame,
        output_dir / "missing_values.png",
        output_dir / "numeric_boxplots.png",
    )
    analysis_frame, exclusion = exclude_capital_outliers(frame)
    create_capital_outlier_comparison(
        frame,
        analysis_frame,
        output_dir / "capital_outlier_comparison.png",
    )
    description = analysis_frame[NUMERIC_COLUMNS].describe().T
    correlation = analysis_frame[NUMERIC_COLUMNS].corr()
    analysis_frame.to_parquet(output_dir / "adult_clean.parquet", index=False)
    create_categorical_countplots(
        analysis_frame,
        output_dir / "categorical_countplots.png",
    )
    create_correlation_heatmap(correlation, output_dir / "correlation_heatmap.png")
    create_plotly_chart(analysis_frame, output_dir / "plotly_occupation_income.html")
    normality = run_normality_analysis(
        analysis_frame,
        output_dir / "normality_diagnostics.png",
    )
    rank_sum = run_rank_sum_test(analysis_frame)
    t_test = run_t_test(analysis_frame)
    chi_square = run_chi_square_test(analysis_frame)
    model_results, best_model_name, evaluation = train_evaluate_models(
        analysis_frame,
        output_dir / "adult_income_pipeline.joblib",
        output_dir / "roc_curve.png",
        output_dir / "model_metrics.png",
    )
    write_report(
        output_dir / "report.md",
        analysis_frame,
        load_times,
        description,
        correlation,
        outlier_summary,
        exclusion,
        normality,
        rank_sum,
        t_test,
        chi_square,
        model_results,
        best_model_name,
        evaluation,
    )
    print("\nDay 2 종합실습 Pipeline을 완료했습니다.")
