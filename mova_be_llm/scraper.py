import os
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp
from tqdm.asyncio import tqdm
from aiohttp_retry import RetryClient, ExponentialRetry


BOOKS_PATH = "data/raw/knihi_books" 
os.makedirs(BOOKS_PATH, exist_ok=True)

URL = "https://knihi.com/updates.html"
EXTS = ('.pdf', '.zip', '.epub', '.djvu', '.pdf.zip')


def get_links(url: str = URL) -> set[str]:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    download_links = set()
    for link in soup.find_all('a'):
        href = link.get('href')
        if (
            href and isinstance(href, str) 
            and not href.lower().endswith(".mp3.zip")
            and any(href.lower().endswith(ext) for ext in EXTS)
        ):
            if href.startswith('http'):
                download_links.add(href)
            else:
                download_links.add(f"https://knihi.com/{href}")
    return download_links

async def fetch(sem: asyncio.Semaphore, session: aiohttp.ClientSession, url: str, timeout_sec: int = 30, max_retries: int = 3):
    timeout = aiohttp.ClientTimeout(
        total=timeout_sec,
        connect=10,       
        sock_read=20,   
    )
    for i in range(max_retries):
        try:
            async with sem:
                async with session.get(url, timeout=timeout) as response:
                    filename = os.path.join(BOOKS_PATH, url.split('/')[-1])
                    total_size = int(response.headers.get("content-length", 0))
                    with open(filename, "wb") as f:
                        progress = tqdm(
                            total=total_size,
                            unit="B",
                            unit_scale=True,
                            desc=filename,
                            ascii=True,
                        )
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                            progress.update(len(chunk))
                        progress.close()
                    return await response.text()
        except aiohttp.ClientError as e:
            print(f"Attempt {i + 1}: Request failed for {url}: {e}")
        except asyncio.TimeoutError as e:
            print(f"Attempt {i + 1}: Request timed out for {url}... {e}")
            await asyncio.sleep(2 ** i)
        except Exception as e:
            print(f"Attempt {i + 1}: Unexpected error for {url}: {e}")
            await asyncio.sleep(2 ** i)
    raise Exception("Max retries exceeded")

async def main():
    download_links = get_links()
    already_downloaded = set(os.listdir("data/raw/knihi_books/"))
    to_download = download_links - already_downloaded
    print(
        f"Get {len(download_links)} book links. "
        f"Number of already downloaded files: {len(already_downloaded)}. "
        f"To download: {len(to_download)}."
    )

    sem = asyncio.Semaphore(8)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(sem, session, url) for url in to_download]
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching"):
            result = await coro
            results.append(result)
        return results


if __name__ == "__main__":
    results = asyncio.run(main())
