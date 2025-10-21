from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import pandas as pd
from datetime import datetime
from webdriver_manager.chrome import ChromeDriverManager

def crawl_shinhan_research_after_2025():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"
    )

    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    driver.get("https://m.shinhansec.com/mweb/invt/shrh/ishrh1001?tabIdx=1")
    time.sleep(2)

    # 🌀 무한 스크롤 (데이터 다 불러오기)
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # 🧩 카드 선택자 (필요시 수정)
    cards = driver.find_elements(By.CSS_SELECTOR, "div.card")  
    print(f"총 {len(cards)}개 카드 감지됨")

    results = []
    filter_date = datetime(2025, 1, 1)  # ✅ 기준 날짜 (2025년 1월 1일)

    for i, card in enumerate(cards):
        try:
            cards = driver.find_elements(By.CSS_SELECTOR, "div.card")
            card = cards[i]
            driver.execute_script("arguments[0].scrollIntoView(true);", card)
            time.sleep(1)
            card.click()
            time.sleep(2)

            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")

            title = soup.select_one(".title").get_text(strip=True)
            subtitle = soup.select_one(".sub-title").get_text(strip=True)
            date_str = soup.select_one(".date").get_text(strip=True)
            author = soup.select_one(".author").get_text(strip=True)
            content = "\n".join([p.get_text(strip=True) for p in soup.select(".content p")])

            # ✅ 날짜 파싱 (예: 2025.10.20)
            date_obj = datetime.strptime(date_str, "%Y.%m.%d")

            # ✅ 날짜 필터 적용
            if date_obj >= filter_date:
                results.append({
                    "제목": title,
                    "부제": subtitle,
                    "날짜": date_str,
                    "작성자": author,
                    "본문": content,
                })
                print(f"✅ {i+1}번째 ({date_str}) 카드 수집됨: {title}")
            else:
                print(f"⏩ {i+1}번째 ({date_str})는 2025년 1월 1일 이전 — 건너뜀")

            driver.back()
            time.sleep(2)

        except Exception as e:
            print(f"⚠️ {i+1}번째 카드 에러: {e}")
            driver.back()
            time.sleep(2)
            continue

    driver.quit()

    df = pd.DataFrame(results)
    df.to_csv("shinhan_research_2025.csv", index=False, encoding="utf-8-sig")
    print(f"✅ 총 {len(df)}개 데이터 저장 완료!")

    return df

# 실행
df = crawl_shinhan_research_after_2025()
