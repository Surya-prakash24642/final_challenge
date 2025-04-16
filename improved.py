import pandas as pd
import os
import time
import re
import json
import logging
import openai
from tqdm import tqdm
from dotenv import load_dotenv

# Configuration
INPUT_EXCEL_FILE = "company_information_results.xlsx"
OUTPUT_EXCEL_FILE = "enriched_companies.xlsx"
LIMIT_COMPANIES = None
REQUEST_DELAY_SECONDS = 1.2
CACHE_FILE = "openai_cache.json"

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenAI key
OPENAI_API_KEY = ""
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
openai.api_key = OPENAI_API_KEY

# Fields to enrich
IMPORTANT_FIELDS = ["Description", "Industry", "Geography", "Employee Count"]
ALL_FIELDS = [
    "Company Name", "Website", "Description", "Software Classification",
    "Enterprise Grade Classification", "Industry", "Geography", "Street Address",
    "City", "Postal Code", "Country", "Phone", "Email", "Employee Count",
    "Customers", "Investors", "Parent Company", "Financial Info"
]

# Load or initialize cache
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
else:
    cache = {}

def fill_missing_fields(company_data):
    company_name = company_data["Company Name"]
    cache_key = company_name.strip().lower()

    # Return from cache if available
    if cache_key in cache:
        logger.info(f"Using cached data for {company_name}")
        return {**company_data, **cache[cache_key]}

    missing_fields = [
        field for field in IMPORTANT_FIELDS
        if field in company_data and (company_data[field] == "Not found" or pd.isna(company_data[field]))
    ]

    if not missing_fields:
        return company_data

    known_info = {
        "Company Name": company_name,
        "Website": company_data.get("Website", "Unknown"),
        "Description": company_data.get("Description", "Unknown"),
        "Industry": company_data.get("Industry", "Unknown"),
    }

    prompt = f"""
For the company '{company_name}', fill in these fields: {', '.join(missing_fields)}.
Known:
- Website: {known_info['Website']}
- Description: {known_info['Description']}
- Industry: {known_info['Industry']}

Respond ONLY with valid JSON like:
{{"Field1": "Value1", "Field2": "Value2"}}. Use \"Not found\" if unknown.
"""

    for attempt in range(3):
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a business analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            time.sleep(REQUEST_DELAY_SECONDS)

            content = response.choices[0].message.content
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                raise ValueError("No valid JSON in response")

            result = json.loads(json_match.group(0))

            for field, value in result.items():
                if field in company_data and (company_data[field] == "Not found" or pd.isna(company_data[field])):
                    company_data[field] = value

            # Update cache
            cache[cache_key] = {field: company_data[field] for field in IMPORTANT_FIELDS}
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)

            logger.info(f"Filled fields for {company_name}: {list(result.keys())}")
            return company_data

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {company_name}: {e}")
            time.sleep(2 ** attempt)

    return company_data

def enrich_company_data():
    logger.info(f"Reading data from {INPUT_EXCEL_FILE}")
    df = pd.read_excel(INPUT_EXCEL_FILE)
    if "Company Name" not in df.columns:
        raise ValueError("Missing required column: Company Name")

    if LIMIT_COMPANIES:
        df = df.head(LIMIT_COMPANIES)

    enriched_data = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        company_data = row.to_dict()
        enriched = fill_missing_fields(company_data)
        enriched_data.append(enriched)

        if (idx + 1) % 5 == 0:
            pd.DataFrame(enriched_data).to_excel(OUTPUT_EXCEL_FILE.replace('.xlsx', '_intermediate.xlsx'), index=False)

    pd.DataFrame(enriched_data).to_excel(OUTPUT_EXCEL_FILE, index=False)
    logger.info(f"Done! Final output saved to {OUTPUT_EXCEL_FILE}")

if __name__ == "__main__":
    print("\n🔍 Optimized Company Data Enrichment Tool Running...")
    enrich_company_data()
