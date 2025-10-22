############################
# 동적 크롤링
# 네이버 증권 사이트는 동적 크롤링으로 해야 다운 가능!
############################

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager # 동적 크롤링
import pandas as pd
from bs4 import BeautifulSoup
import time
from pathlib import Path

def get_kospi_top100_selenium():
    options = Options()
    options.add_argument("--headless")  # 창을 띄우지 않음(백그라운드로 진행)
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    stocks = [] # 데이터를 담을 list 생성

    for page in range(1, 3):
        url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok=0&page={page}"
        driver.get(url)
        time.sleep(1)  # 페이지 로딩 대기
        soup = BeautifulSoup(driver.page_source, "html.parser")
        rows = soup.select("table.type_2 tbody tr")

        for row in rows:
            cols = [col.get_text(strip=True) for col in row.select("td")]
            if len(cols) < 10:
                continue
            stocks.append({
                "종목명": cols[1],
                "현재가": cols[2],
                "시가총액": cols[6],
                "PER": cols[10],
                "ROE": cols[11],
            })

    driver.quit()
    df = pd.DataFrame(stocks)
    print(f"✅ 코스피 상위 {len(df)}개 기업 불러오기 완료!")
    return df

# 실행
top100 = get_kospi_top100_selenium()
print(top100.head(100))

# 저장하기
output_path = Path(__file__).resolve().parents[2] / "data" / "kospi_top100.txt"
top100.to_csv(output_path, sep="\t", index=False, encoding="utf-8-sig")