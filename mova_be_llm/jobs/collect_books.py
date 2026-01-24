import asyncio

from mova_be_llm.scraper import Scraper

URL = "https://knihi.com/updates.html"


def main():
    cd = Scraper(
        URL,
        out_dir="./data/raw/",
        max_depth=2,
        ext_filters=("zip", "pdf", "epub", "djvu"),
        concurrency=4,
        obey_robots=True,
        rate_min=0.1,
        rate_max=0.6,
        timeout=30,
        max_retries=3,
        user_agent=None,
    )
    cd.prepare()
    try:
        asyncio.run(cd.run())
    except KeyboardInterrupt:
        print("Interrupted by user")


if __name__ == "__main__":
    main()
