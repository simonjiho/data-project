# 실습 3 - Pandas·Polars·DuckDB 비교

PDF 70~87페이지의 Pandas EDA, IQR 이상치 처리, Polars Lazy API 및 DuckDB 내용을 반영한 코드입니다. DuckDB도 Relation API로 실행 계획을 연결한 뒤 마지막 `df()`에서 실행하며, 같은 집계의 SQL 버전은 Python 파일 안에 학습용 주석으로 남겨 두었습니다.

## 준비

`sales_100k.csv`를 프로젝트의 `data` 폴더에 넣습니다. 데이터에는 다음 컬럼이 필요합니다.

- `region`
- `category`
- `amount`

필요한 패키지를 설치합니다.

```bash
python -m pip install -r ../requirements.txt
```

## 실행

기본 데이터 경로를 사용할 때:

```bash
python 광주_1반_김지호.py
```

CSV가 다른 위치에 있을 때:

```bash
python 광주_1반_김지호.py /경로/sales_100k.csv --number 5
```

`--number`는 각 도구의 성능 측정 반복 횟수입니다. 세 도구 모두 같은 값으로 실행됩니다.
