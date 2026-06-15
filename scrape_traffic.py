import asyncio
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path

from playwright.async_api import async_playwright

BASE_URL = "https://mainedottrafficdata.drakewell.com/tfdaysreport.asp"
SITE_ID = "133119702600"
NODE = "MAINE_DOT_CCS"
DIR = "-3"
CHUNK_DAYS = 35
DELAY_SECS = 0.5
MAX_RETRIES = 3

START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2026, 6, 14)

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)


def build_url(report_date: datetime, end_date: datetime) -> str:
    return (
        f"{BASE_URL}?node={NODE}&cosit={SITE_ID}&dir={DIR}"
        f"&reportdate={report_date:%Y-%m-%d}"
        f"&enddate={end_date:%Y-%m-%d}&sidebar=1"
    )


def generate_chunks(start: datetime, end: datetime, chunk_days: int):
    chunks = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


async def wait_for_table_data(page, timeout_secs: int = 30) -> bool:
    """Poll every 0.5s until the main table has hourly data rows with numbers."""
    deadline = asyncio.get_event_loop().time() + timeout_secs
    while asyncio.get_event_loop().time() < deadline:
        has_data = await page.evaluate("""() => {
            const table = document.querySelectorAll('table')[0];
            if (!table) return false;
            const rows = table.querySelectorAll('tr');
            // Look for a row that starts with an hour label and has numeric data
            for (let i = 0; i < rows.length; i++) {
                const cells = rows[i].querySelectorAll('th, td');
                if (cells.length < 2) continue;
                const text = cells[0].textContent.trim();
                const val = cells[1].textContent.trim();
                if ((text.endsWith('am') || text.endsWith('pm')) && !isNaN(parseInt(val))) {
                    return true;
                }
            }
            return false;
        }""")
        if has_data:
            return True
        await asyncio.sleep(0.5)
    return False


async def extract_table_data(page, url: str) -> dict:
    await page.goto(url, wait_until="domcontentloaded")

    if not await wait_for_table_data(page):
        raise RuntimeError("Table data never loaded")

    return await page.evaluate("""() => {
        const table = document.querySelectorAll('table')[0];
        const rows = table.querySelectorAll('tr');

        // Extract site info
        const siteNameEl = document.evaluate(
            '//div[contains(text(), "Site Name")]/following-sibling::div',
            document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null
        ).singleNodeValue;

        const result = {
            site_name: siteNameEl ? siteNameEl.textContent.trim() : '',
            sections: []
        };

        const sectionNames = ['All directions', 'All Northbound', 'All Southbound',
                              'Ln 1 NB', 'Center Turn Lane', 'Ln 1 SB'];
        const dowNames = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

        let currentSection = null;

        rows.forEach((row) => {
            const cells = row.querySelectorAll('th, td');
            const values = Array.from(cells).map(c => c.textContent.trim());
            const label = values[0];

            // Section header row
            if (sectionNames.includes(label)) {
                currentSection = {
                    direction: label,
                    dates: [],
                    rows: []
                };
                result.sections.push(currentSection);
                return;
            }

            if (!currentSection) return;

            // Day-of-week header row (empty label, first data cell is a day name)
            if (label === '' && currentSection.dates.length === 0 &&
                currentSection.rows.length === 0 &&
                dowNames.includes(values[1])) {
                return; // skip, we don't need days of week
            }

            // Date header row (empty label, first data cell matches "Mon 01" pattern)
            if (label === '' && currentSection.dates.length === 0 &&
                values.length > 1 && values[1].match(/^[A-Z][a-z]{2} \\d{1,2}$/)) {
                // Skip last 3 columns: Workday, 7 Day, Count (or Average, Total, Count)
                currentSection.dates = values.slice(1, -3);
                return;
            }

            // Skip blank separator rows
            if (label === '' && values.every(v => v === '')) return;

            // Data row
            currentSection.rows.push({
                label: label,
                values: values.slice(1, -3)  // skip summary columns
            });
        });

        return result;
    }""")


def resolve_dates(short_dates: list[str], chunk_start: datetime) -> list[str]:
    """Convert ['Aug 21', 'Aug 22', ...] to ['2025-08-21', '2025-08-22', ...]"""
    from datetime import datetime as dt

    # Build a mapping from "Mon DD" -> actual date by walking the chunk range
    resolved = []
    current = chunk_start
    for short in short_dates:
        resolved.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return resolved


def flatten_data(raw: dict, chunk_start: datetime) -> list[dict]:
    records = []
    site_name = raw.get("site_name", "")

    for section in raw["sections"]:
        direction = section["direction"]
        short_dates = section["dates"]

        if not short_dates:
            continue

        dates = resolve_dates(short_dates, chunk_start)

        for row in section["rows"]:
            label = row["label"]
            values = row["values"]

            # Hourly rows (e.g. "12:00 am", "01:00 pm") and summary rows
            if label and (
                ":00" in label or
                label in ("7am-7pm", "6am-10pm", "6am-12am", "12am-12am")
            ):
                for i, date_str in enumerate(dates):
                    if i < len(values):
                        records.append({
                            "site_id": SITE_ID,
                            "site_name": site_name,
                            "direction": direction,
                            "date": date_str,
                            "hour": label,
                            "count": values[i],
                        })

            elif label in ("am Peak", "pm Peak"):
                for i, date_str in enumerate(dates):
                    if i < len(values):
                        records.append({
                            "site_id": SITE_ID,
                            "site_name": site_name,
                            "direction": direction,
                            "date": date_str,
                            "peak_type": label,
                            "peak_hour": values[i],
                        })

            elif label == "Peak Volume":
                for i, date_str in enumerate(dates):
                    if i < len(values):
                        records.append({
                            "site_id": SITE_ID,
                            "site_name": site_name,
                            "direction": direction,
                            "date": date_str,
                            "peak_volume": values[i],
                        })

    return records


async def fetch_chunk(page, start: datetime, end: datetime) -> list[dict]:
    url = build_url(start, end)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = await extract_table_data(page, url)
            return flatten_data(raw, start)
        except Exception as e:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1)
            else:
                raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {e}")


async def main():
    chunks = generate_chunks(START_DATE, END_DATE, CHUNK_DAYS)
    total_chunks = len(chunks)
    print(f"Collecting {total_chunks} chunks from {START_DATE:%Y-%m-%d} to {END_DATE:%Y-%m-%d}")

    all_records = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        for i, (start, end) in enumerate(chunks, 1):
            url = build_url(start, end)
            print(f"[{i}/{total_chunks}] {start:%Y-%m-%d} to {end:%Y-%m-%d} ... ", end="", flush=True)

            try:
                records = await fetch_chunk(page, start, end)
                all_records.extend(records)
                print(f"got {len(records)} records")
            except Exception as e:
                print(f"ERROR: {e}")

            if i < total_chunks:
                await asyncio.sleep(DELAY_SECS)

        await browser.close()

    # Save to JSON
    json_path = OUTPUT_DIR / "raw_traffic.json"
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=2)

    # Save hourly counts as CSV, deduplicated
    csv_path = OUTPUT_DIR / "hourly_counts.csv"
    hourly = [r for r in all_records if "count" in r]
    seen = set()
    unique = []
    for r in hourly:
        key = (r["date"], r["hour"], r["direction"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    if unique:
        fieldnames = ["site_id", "site_name", "direction", "date", "hour", "count"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(unique)

    print(f"\nDone! {len(all_records)} total records, {len(unique)} unique hourly records")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
