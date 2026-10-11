# data-project

데이터 분석용 Python 실습 코드 모음이다. 실습 4개와 종합실습 2개가 들어 있다. 코드에 적힌 작성일은 2026-07-20과 07-21이다.

| 폴더 | 한 일 |
|---|---|
| `실습1` | json 판매 기록을 컴프리헨션, `Counter`, `defaultdict`로 집계한다. 제너레이터와 리스트의 메모리 크기를 `sys.getsizeof`로 비교하고 `assert`로 결과를 검증한다 |
| `실습2` | 같은 json을 Pydantic으로 검증한다. 정상 레코드는 CSV, 오류는 JSON으로 저장하고 CSV를 다시 읽어 건수를 맞춘다. 파일 읽기 예외 처리를 넣었다 |
| `실습3` | `sales_100k.csv`로 Pandas EDA와 IQR 이상치 제거를 한 뒤, 지역·카테고리별 합계·평균·건수를 Pandas, Polars(Lazy), DuckDB로 집계한다. 결과가 같은지 확인하고 `timeit`으로 시간을 잰다 |
| `실습4` | 같은 CSV로 2×2 EDA 차트, Welch t-test, 카이제곱 검정과 Cramér's V, `ColumnTransformer`+Ridge `Pipeline`(joblib 저장·재로딩), Plotly HTML 차트를 만든다 |
| `day1종합실습` | Open-Meteo, Countries.dev, ip-api를 `asyncio.gather`로 동시에 호출하고 Pydantic으로 검증해 한 표로 합친다. CSV와 Parquet의 저장·읽기 시간과 파일 크기를 비교한다. 재시도 데코레이터와 pytest 테스트가 있다 |
| `day2종합실습` | UCI Adult Census Income으로 EDA, 정규성·순위합·t·카이제곱 검정을 하고 분류 모델 3종(Logistic Regression, Random Forest, HistGradientBoosting)을 비교한다. 결과는 Jinja2 템플릿으로 `report.md`에 쓴다 |

폴더마다 실행 파일은 `.py` 하나다. `day1종합실습`, `day2종합실습`, `실습3`, `실습4`에는 폴더별 README가 따로 있다.

## 실행

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

폴더에 들어가 그 폴더의 `.py` 파일을 `python`으로 실행한다. 입력 파일은 이렇다.

- 실습1·2: `data/Python_Practice2_Data.json`
- 실습3·4: `data/sales_100k.csv` (`region`, `category`, `amount` 컬럼 필요)
- day2: `data/adult.data`. 없으면 UCI에서 내려받는다
- day1: 파일은 필요 없고 인터넷 연결이 필요하다

`data/`와 `output/`은 `.gitignore`에 들어 있어 저장소에 없다. 결과는 각 폴더의 `output/`에 생긴다.
