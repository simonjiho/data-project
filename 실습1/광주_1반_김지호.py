"""
Practice 1: 컴프리헨션, Counter, defaultdict, 제너레이터 실습.

작성자: 김지호
작성일자: 26.07.20

본 코드는 데이터 분석을 위한 python 실습입니다.
코드의 가독성을 위해 항목별 문항만 코드 중간에 주석으로 표기하였습니다.


주 학습내용은 다음과 같습니다.
- json 파일 로드
- 컴프리헨션 문법을 통해 딕셔너리, 리스트 등 컬렉션 생성
- defaultdict를 활용하여 dictionary 만들기
- generator 생성 및 sys.getsizeof() 함수를 이용해 generator의 효율성 확인
- assert를 활용한 validation



새로 학습한 내용을 정리한 바는 다음과 같습니다.

- defaultdict()
    함수를 인자로 받아, default value 생성시 호출한다.
    존재하지 않는 key value 쌍에 접근 시 자동으로 default value를 부여하여 key value 쌍을 생성한다.
    주의:
        defaultdict(함수명()) -> x
        defaultdict(함수명) -> o
    lambda도 활용 가능하다
        defaultdict(lambda: [0])

- generator
    두가지 사용법이 존재한다.
        1. 소괄호 () 안에 컴프리헨드로 작성
        2. yield를 포함한 함수를 작성
            - 호출 시 본문이 곧바로 실행되지 않는다.
            - 호출 시 값 대신 generator 객체를 반환한다.
                - yield가 하나라도 포함 된 함수는 무조건 generator를 반환한다.
            - return은 generator를 중간에 끝낼 때 유용하다.
                - 그러나 return하는 값은 일반적인 반복 결과에는 포함되지 않고, StopIteration 예외에 저장된다.

- dict.fromkeys() (ai gen code에서 학습)
    인자로 받은 key들을 바탕으로 dictionary를 생성하는 함수
    dictionary의 key들은 순서가 있다. (python3.7+) 이를 이용해서 순서를 유지하면서 중복을 제거한 value가 none인 dictionary를 만들 수 있다.

- lambda
    사용법: "lambda 변수1, 변수2, ...: 반환값"


"""

import sys
import json
from collections import Counter, defaultdict


with open("Python_Practice2_Data.json", encoding="utf-8") as file:
    sales = json.load(file)


# 1) list/dictionary comprehension
filtered_sales = [row for row in sales if row["amount"] >= 1000]
regions = dict.fromkeys(row["region"] for row in filtered_sales)

region_total = {
    region: sum(row["amount"] for row in filtered_sales if row["region"] == region)
    for region in regions
}


# 2) Counter + defaultdict
region_count = Counter(row["region"] for row in sales)
region_most_common = region_count.most_common()

category_amounts = defaultdict(list)

for row in sales:
    category_amounts[row["category"]].append(row["amount"])

top3 = sorted(sales, key=lambda row: row["amount"], reverse=True)[:3]


# 3) generator - memory comparison
def sales_over_1000(rows):
    """amount가 1,000을 초과하는 거래를 한 건씩 생성한다."""
    for row in rows:
        if row["amount"] > 1000:
            yield row


sales_list = [row for row in sales if row["amount"] > 1000]
sales_generator = sales_over_1000(sales)
list_size = sys.getsizeof(sales_list)
generator_size = sys.getsizeof(sales_generator)


# 4) total sales grouped by monthly-category
monthly_category_group = defaultdict(list)
for row in sales:
    monthly_category_group[(row["month"], row["category"])].append(row["amount"])

monthly_category_total = {
    key: sum(amounts) for key, amounts in monthly_category_group.items()
}


# checkpoint
assert region_total == {
    "서울": 17670,
    "부산": 4550,
    "대구": 8320,
    "인천": 11950,
    "광주": 4830,
    "대전": 6300,
    "울산": 7270,
    "세종": 5750,
}
assert region_most_common == [
    ("서울", 14),
    ("부산", 13),
    ("대구", 13),
    ("인천", 12),
    ("광주", 12),
    ("대전", 12),
    ("울산", 12),
    ("세종", 12),
]
assert generator_size < list_size
assert [row["amount"] for row in top3] == [2500, 2200, 2200]


# execution
if __name__ == "__main__":
    print("1. amount >= 1000 거래 수:", len(filtered_sales))
    print("   지역별 총매출:", region_total)
    print("2. 지역별 거래 건수:", region_most_common)
    print("   카테고리별 amount:", dict(category_amounts))
    print("   top3:", top3)
    print("3. list 메모리 크기:", list_size, "bytes")
    print("   generator 메모리 크기:", generator_size, "bytes")
    print("4. 월별 카테고리 총매출:", monthly_category_total)
