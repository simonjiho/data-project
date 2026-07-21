# Day 1 종합실습 - 비동기 API 데이터 파이프라인

Open-Meteo, Countries.dev 및 ip-api의 데이터를 비동기로 수집하고 Pydantic으로 검증한 뒤 CSV와 Parquet으로 저장·비교하는 프로젝트입니다.

## 환경 설정

프로젝트 루트에서 공용 환경과 `requirements.txt`를 사용합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 실행

```bash
cd day1종합실습
../.venv/bin/python 광주_1반_김지호.py
```

외부 API를 호출하므로 실행 시 인터넷 연결이 필요합니다.

## 테스트

```bash
../.venv/bin/pytest -q test_day1.py
```

## 산출물

`output/` 폴더에 수집 데이터와 실행결과 보고서가 저장됩니다.

- `collected_data.csv`
- `collected_data.parquet`
- `documents/광주_1반_김지호_Day1_종합실습_실행결과보고서.docx`
- `pdf/광주_1반_김지호_Day1_종합실습_실행결과보고서.pdf`
