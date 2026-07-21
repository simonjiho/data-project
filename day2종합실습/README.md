# Day 2 종합실습 - Adult Census Income

강의자료 107~124페이지의 자동화·프로젝트 구조화·End-to-End 종합실습 요구사항을 구현한 프로젝트입니다.

## 데이터

- 데이터셋: UCI Adult Census Income
- 입력 경로: `../data/adult.data`
- 출처: <https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data>

최초 실행 시 입력 파일이 없으면 UCI 공식 URL에서 `data/adult.data`로 자동 다운로드합니다. 이후 실행에서는 저장된 원본 파일을 다시 사용하며 직접 수정하지 않습니다. 정제 결과는 이 프로젝트의 `output/`에 별도로 저장합니다.

## 환경 설정

프로젝트 루트에서 다음 명령을 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```


## 실행

```bash
cd day2종합실습
../.venv/bin/python 광주_1반_김지호.py
```

## 분석 단계

1. Pandas·Polars 로딩 및 정제 결과·시간 비교
2. 결측치 현황과 수치형 변수 Boxplot·IQR 이상치 후보 분석
3. 중복 처리와 기본 EDA
4. Seaborn 정적 차트·범주형 Count bar plot·Plotly 인터랙티브 차트 생성
5. `capital_gain`·`capital_loss`의 0을 보존하고 양수값 내부 IQR 극단값 제외
6. Shapiro-Wilk 정규성 검정·소득 그룹별 Boxplot·Q-Q plot 생성
7. Mann-Whitney U(Wilcoxon rank-sum)·Welch t-test·카이제곱 검정·Cramér's V 해석
8. sklearn Pipeline으로 Logistic Regression·Random Forest·Histogram Gradient Boosting 학습
9. 정확도·F1·ROC-AUC·PR-AUC 비교 및 최고 모델 joblib 저장
10. Jinja2 템플릿으로 용어 정리를 포함한 `report.md` 자동 생성

## 주요 폴더 구조

```text
day2종합실습/
├── src/analysis_pipeline.py       # 데이터 처리·분석·모델링·렌더링 데이터 준비
├── templates/report.md.j2         # 자동 생성 보고서의 Markdown 본문
├── tests/test_analysis_pipeline.py
├── 광주_1반_김지호_day2종합실습.py  # 실행 진입점
└── output/                        # 실행 결과(자동 생성, Git 제외)
```

보고서 문장과 레이아웃은 `templates/report.md.j2`에서 관리하고, Python 코드는 계산 결과를 템플릿에 전달합니다. 템플릿에 필요한 값이 누락되면 `StrictUndefined` 설정으로 즉시 오류가 발생합니다.

## 학습·평가 데이터 분리

자본손익 이상치를 제외한 최종 분석 데이터 32,264행을 `train_test_split`으로 무작위 분리합니다.

- 학습 데이터: 80%, 25,811행
- 평가 데이터: 20%, 6,453행
- `random_state=42`: 다시 실행해도 동일한 행이 선택되도록 난수 고정
- `stratify=target`: 학습·평가 데이터의 `>50K`와 `<=50K` 비율 유지
- 모델 비교: 세 모델 모두 동일한 학습·평가 데이터 사용
- 평가 데이터 출처: 별도 데이터를 내려받지 않고 `adult.data` 내부에서 분리한 hold-out 데이터이며 UCI `adult.test`는 사용하지 않음

현재 구현은 자본손익 IQR 경계를 전체 데이터에서 계산한 후 학습·평가 데이터를 나누므로, 평가 데이터의 정보가 이상치 기준에 미세하게 반영될 수 있습니다. 더 엄격한 평가에서는 데이터를 먼저 분리하고 학습 데이터만으로 이상치 기준을 계산하거나, UCI의 별도 `adult.test`를 최종 평가에 사용해야 합니다.

## 생성 결과

실행 후 `output/` 폴더에 다음 파일이 생성됩니다.

- `adult_clean.parquet`
- `missing_values.png`
- `numeric_boxplots.png`
- `capital_outlier_comparison.png`
- `categorical_countplots.png`
- `normality_diagnostics.png`
- `correlation_heatmap.png`
- `plotly_occupation_income.html`
- `adult_income_pipeline.joblib`
- `roc_curve.png`
- `model_metrics.png`
- `report.md`
