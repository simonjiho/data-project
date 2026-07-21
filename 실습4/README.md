# 실습 4 - 시각화·통계 검정·sklearn Pipeline

PDF 88~106페이지의 시각화, 통계 검정 및 머신러닝 Pipeline 내용을 실습 3의 정제 과정과 연결한 코드입니다.

## 준비

`sales_100k.csv`를 프로젝트의 `data` 폴더에 넣습니다. 다음 컬럼이 필요합니다.

- `region`
- `category`
- `amount`

`date`, `order_date`, `sales_date`, `transaction_date` 등의 날짜 컬럼이 있으면 2×2 차트에 월별 총매출 라인을 표시합니다. 날짜 컬럼이 없으면 카테고리별 총매출 막대 차트로 대체합니다.

```bash
python -m pip install -r ../requirements.txt
```

## 실행

```bash
python 광주_1반_김지호.py
```

CSV가 다른 위치에 있다면 경로를 전달합니다.

```bash
python 광주_1반_김지호.py /경로/sales_100k.csv
```

## 생성 결과

`output` 폴더에 다음 파일을 생성합니다.

- `eda_2x2.png`: 히스토그램·박스플롯·라인/막대·상관 히트맵
- `sales_pipeline.joblib`: 전처리와 Ridge 모델을 묶은 Pipeline
- `sales_by_region_category.html`: Plotly 인터랙티브 막대 차트

카이제곱 검정 결과에는 p-value뿐 아니라 범주형 변수 간 관계의 실질적인 크기를 보여주는 Cramér's V와 종합 해석도 함께 출력합니다.
