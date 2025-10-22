############################
# 코드 실행한 유의사항!!
'''
requirements_crawling.txt에 있는 패키지 설치
playwright install -> 브라우저 엔진 설치

위 두 단계를 먼저 해주셔야합니다!
'''
############################
import asyncio
from playwright.async_api import async_playwright
import pandas as pd
from datetime import datetime

async def crawl_shinhan_reports():
    base_url = "https://m.shinhansec.com/mweb/invt/shrh/ishrh1001?tabIdx=1"
    results = []
    filter_date = datetime(2025, 1, 1)
    MAX_CARDS = 1000  # 🚧 테스트 시 10개만 (완료되면 1000으로 변경 가능)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={"width": 430, "height": 932},
            user_agent=(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"
            ),
        )

        page = await context.new_page()
        await page.goto(base_url, timeout=60000)
        await page.wait_for_selector("li.list__card-items")
        print(f"📜 무한 스크롤 시작... (최대 {MAX_CARDS}개까지)")

        # 🔽 무한 스크롤
        prev_count, same_count = 0, 0
        while True:
            await page.evaluate("""
                const el = document.querySelector('.content');
                if (el) el.scrollTo(0, el.scrollHeight);
                else window.scrollBy(0, document.body.scrollHeight);
            """)
            await page.wait_for_timeout(1500)

            cards = await page.query_selector_all("li.list__card-items")
            new_count = len(cards)
            print(f"📈 감지된 카드 수: {new_count}")

            if new_count >= MAX_CARDS:
                print(f"✅ {MAX_CARDS}개 도달 — 스크롤 종료")
                break

            if new_count == prev_count:
                same_count += 1
            else:
                same_count = 0
                prev_count = new_count

            if same_count >= 3:
                print("✅ 더 이상 새로운 카드 없음 — 스크롤 종료")
                break

        cards = await page.query_selector_all("li.list__card-items")
        print(f"총 {len(cards)}개 카드 감지 완료!\n")

        # 🧩 각 카드 본문 크롤링
        for i, card in enumerate(cards[:MAX_CARDS]):
            try:
                data = await card.query_selector("div.list_data_area")

                title = await data.get_attribute("data-title")
                subtitle = await data.get_attribute("data-subtitle")
                date_str = await data.get_attribute("data-date")
                author = await data.get_attribute("data-nickname")
                category = await data.get_attribute("data-gubun")
                link = await data.get_attribute("data-message_url")

                if not link or not date_str:
                    continue

                date_obj = datetime.strptime(date_str, "%Y.%m.%d")
                if date_obj < filter_date:
                    continue

                # 새 탭 열기
                detail_page = await context.new_page()
                await detail_page.goto(link, timeout=90000)
                await detail_page.wait_for_load_state("domcontentloaded")
                await detail_page.wait_for_timeout(2000)

                content = ""

                try:
                    # ✅ iframe 내부 접근
                    frames = detail_page.frames
                    target_frame = None
                    for f in frames:
                        if "bbs2.shinhaninvest.com" in (f.url or ""):
                            target_frame = f
                            break

                    if target_frame:
                        await target_frame.wait_for_selector("#contents", timeout=20000)

                        # ✅ 요약(span) 제거 (본문 첫 span 전체 삭제)
                        await target_frame.evaluate("""
                            const container = document.querySelector('#contents');
                            if (container) {
                                const firstSpan = container.querySelector('span');
                                if (firstSpan) firstSpan.remove();
                            }
                        """)

                        # ✅ 본문 추출
                        content_el = await target_frame.query_selector("#contents")
                        if content_el:
                            content = await content_el.inner_text()
                        else:
                            print(f"⚠️ 본문 요소 없음: {title}")
                    else:
                        print(f"⚠️ iframe을 찾을 수 없음: {title}")

                except Exception as e:
                    print(f"⚠️ 본문 추출 오류: {e}")

                await detail_page.close()

                results.append({
                    "제목": title.strip() if title else "",
                    "부제": subtitle.strip() if subtitle else "",
                    "날짜": date_str,
                    "작성자": author.strip() if author else "",
                    "카테고리": category.strip() if category else "",
                    "링크": link,
                    "본문": content.strip() if content else "",
                })

                print(f"✅ {i+1}번째 완료: {title} ({date_str})")

            except Exception as e:
                print(f"⚠️ {i+1}번째 카드 에러: {e}")
                continue

        await browser.close()

    # 💾 CSV 저장
    df = pd.DataFrame(results)
    df.to_csv("./data/shinhan_research_2025_playwright.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 총 {len(df)}개 데이터 저장 완료! → shinhan_research_2025_playwright.csv")


if __name__ == "__main__":
    asyncio.run(crawl_shinhan_reports())
