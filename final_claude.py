import requests
from bs4 import BeautifulSoup
import pandas as pd
import xml.etree.ElementTree as ET
import re
import os
from urllib.parse import urlparse, urljoin, quote
import time
from supabase import create_client
import google.generativeai as genai
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
import json
from fake_useragent import UserAgent  # For rotating user agents

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration - REPLACE WITH ENVIRONMENT VARIABLES IN PRODUCTION
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAu5ft39F5BQ4-zDG1PhqYjM9yDNfSbYAk")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://edekginkagyjntqetfej.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVkZWtnaW5rYWd5am50cWV0ZmVqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ3MDk2NTMsImV4cCI6MjA2MDI4NTY1M30.4mQBtWI4QnPSiclCjCWQBlI69Mb98AbxxYrdNo01cSc")
# New keys for Google Custom Search Engine
GOOGLE_CSE_API_KEY = os.environ.get("GOOGLE_CSE_API_KEY", "AIzaSyCT0Kp6bj4goWnBKgi3XJjqUTvkaIlamEc")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID", "467392a6f2ed14c71")

# Initialize rotating user agent
ua = UserAgent()

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Rate limiting parameters
REQUEST_DELAY = 1.0  # Minimum seconds between requests to the same domain
domain_last_request = {}  # Track last request time per domain

# Create tables in Supabase if they don't exist
def setup_supabase():
    try:
        # Test connection by getting count from tables
        supabase.table("company_information").select("count", count="exact").limit(1).execute()
        
        logger.info("Supabase tables exist and connection successful")
    except Exception as e:
        logger.error(f"Supabase tables may not exist: {e}")
        logger.info("Please create the company_information table in your Supabase dashboard")
        raise Exception("Required Supabase tables not found. Please create them first.")

def get_with_rate_limit(url, timeout=20, max_retries=3):
    """Make a GET request with rate limiting and exponential backoff"""
    domain = urlparse(url).netloc
    
    # Respect rate limits
    current_time = time.time()
    if domain in domain_last_request:
        time_since_last = current_time - domain_last_request[domain]
        if time_since_last < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - time_since_last)
    
    headers = {
        'User-Agent': ua.random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    retry_delay = 1
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            # Update last request time
            domain_last_request[domain] = time.time()
            
            return response
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                return None
            
            # Exponential backoff
            sleep_time = retry_delay * (2 ** attempt)
            logger.warning(f"Request to {url} failed. Retrying in {sleep_time}s. Error: {e}")
            time.sleep(sleep_time)
    
    return None

# Function to get important pages
def get_important_pages(company_name, website_url):
    """Get only the most important pages instead of scraping everything"""
    try:
        # Ensure URL has proper format
        if not website_url.startswith('http'):
            website_url = 'https://' + website_url
        
        logger.info(f"Extracting key pages for {company_name}...")
        
        # Define key page patterns we're interested in
        key_paths = [
            "",  # Homepage
            "about", "about-us", "company", "team", "our-team",
            "contact", "contact-us",
            "customers", "clients", "case-studies", "testimonials",
            "investors", "investor-relations", "funding",
            "products", "services", "solutions",
            "partners", "pricing", "careers"
        ]
        
        # Create list of potential important URLs
        base_domain = urlparse(website_url).netloc
        target_urls = [
            urljoin(website_url, path) for path in key_paths
        ]
        
        # Verify which URLs actually exist
        valid_urls = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            def check_url(url):
                try:
                    response = get_with_rate_limit(url, timeout=5)
                    if response and response.status_code == 200:
                        return url
                    return None
                except Exception:
                    return None
            
            checked_urls = list(executor.map(check_url, target_urls))
            valid_urls = [url for url in checked_urls if url]
        
        # Get homepage content to extract additional important links
        response = get_with_rate_limit(website_url, timeout=15)
        if response and response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for important links based on text
            important_keywords = ['about', 'contact', 'team', 'investor', 'partner', 
                                 'product', 'service', 'customer', 'career']
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                link_text = link.get_text().strip().lower()
                
                # Check if link text contains important keywords
                if any(keyword in link_text for keyword in important_keywords):
                    # Convert to absolute URL
                    full_url = urljoin(website_url, href)
                    # Only include if it's from the same domain
                    if urlparse(full_url).netloc == base_domain and full_url not in valid_urls:
                        valid_urls.append(full_url)
        
        logger.info(f"Found {len(valid_urls)} important URLs for {company_name}")
        
        # Limit to a reasonable number
        if len(valid_urls) > 15:
            logger.info(f"Limiting to 15 most important URLs")
            # Prioritize the homepage and pages with important keywords in the path
            homepage = [url for url in valid_urls if url == website_url]
            important_pages = [url for url in valid_urls if any(f"/{keyword}" in url.lower() for keyword in important_keywords)]
            other_pages = [url for url in valid_urls if url not in homepage and url not in important_pages]
            
            valid_urls = homepage + important_pages + other_pages
            valid_urls = valid_urls[:15]  # Limit to first 15
        
        return valid_urls
    
    except Exception as e:
        logger.error(f"Error getting important pages for {company_name}: {e}")
        # Return at least the homepage if everything fails
        return [website_url]

# Function to scrape content from a URL
def scrape_url(url):
    try:
        response = get_with_rate_limit(url, timeout=15)
        if not response or response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text
        text = soup.get_text(separator=' ')
        
        # Clean text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Limit text size to prevent memory errors
        if len(text) > 100000:  # ~100KB limit per page
            logger.warning(f"Text from {url} is too large, truncating to first 100KB")
            text = text[:100000]
            
        return {
            'url': url,
            'content': text
        }
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return None

# NEW FUNCTION: Google Custom Search Engine API
def google_cse_search(query, specific_fields=None):
    """Use Google Custom Search Engine to find specific company information
    
    Args:
        query (str): The search query, typically company name + info type
        specific_fields (list, optional): List of specific fields to search for
    
    Returns:
        dict: Search results organized by field
    """
    try:
        # Base CSE API URL
        base_url = "https://www.googleapis.com/customsearch/v1"
        
        # If specific fields provided, create targeted queries
        if specific_fields:
            results = {}
            
            for field in specific_fields:
                # Create field-specific query
                field_query = f"{query} {field}"
                logger.info(f"Performing CSE search for: {field_query}")
                
                # Make API request with rate limiting
                params = {
                    'key': GOOGLE_CSE_API_KEY,
                    'cx': GOOGLE_CSE_ID,
                    'q': field_query,
                    'num': 3  # Get top 3 results per field
                }
                
                # Respect rate limits for Google API (100 queries per day free tier)
                time.sleep(2)  # Basic rate limiting
                
                response = requests.get(base_url, params=params)
                
                if response.status_code == 200:
                    search_results = response.json()
                    
                    # Extract and store results for this field
                    field_results = []
                    if 'items' in search_results:
                        for item in search_results['items']:
                            result_data = {
                                'title': item.get('title', ''),
                                'link': item.get('link', ''),
                                'snippet': item.get('snippet', '')
                            }
                            field_results.append(result_data)
                    
                    results[field] = field_results
                else:
                    logger.error(f"CSE API error: {response.status_code}")
                    results[field] = []
            
            return results
        else:
            # Simple search without field specificity
            logger.info(f"Performing general CSE search for: {query}")
            
            params = {
                'key': GOOGLE_CSE_API_KEY,
                'cx': GOOGLE_CSE_ID,
                'q': query,
                'num': 5  # Get top 5 results
            }
            
            response = requests.get(base_url, params=params)
            
            if response.status_code == 200:
                search_results = response.json()
                results = []
                
                if 'items' in search_results:
                    for item in search_results['items']:
                        result_data = {
                            'title': item.get('title', ''),
                            'link': item.get('link', ''),
                            'snippet': item.get('snippet', '')
                        }
                        results.append(result_data)
                
                return {'general': results}
            else:
                logger.error(f"CSE API error: {response.status_code}")
                return {'general': []}
                
    except Exception as e:
        logger.error(f"Error in Google CSE search: {e}")
        return {}

# Function to extract content from search results
def extract_from_cse_results(results, max_urls=3):
    """Extract content from top URLs in CSE results"""
    extracted_content = []
    
    # Get unique URLs from all fields
    all_urls = set()
    for field, field_results in results.items():
        for item in field_results:
            all_urls.add(item['link'])
    
    # Limit number of URLs to scrape
    urls_to_scrape = list(all_urls)[:max_urls]
    
    # Scrape content from URLs
    with ThreadPoolExecutor(max_workers=3) as executor:
        scraped_contents = list(executor.map(scrape_url, urls_to_scrape))
    
    # Add snippets and content to results
    for field, field_results in results.items():
        field_content = f"Information about {field}:\n"
        
        # Add snippets from search results
        for item in field_results:
            field_content += f"- {item['title']}: {item['snippet']}\n"
        
        # Add full content from scraped pages
        for content_dict in scraped_contents:
            if content_dict and content_dict['content']:
                field_content += f"\nContent from {content_dict['url']}:\n"
                field_content += content_dict['content'][:5000]  # Limit size per URL
        
        extracted_content.append({
            'field': field,
            'content': field_content
        })
    
    return extracted_content

# EXISTING THIRD-PARTY SOURCE FUNCTIONS
def get_crunchbase_info(company_name):
    """Scrape company information from Crunchbase"""
    try:
        # Format company name for URL
        formatted_name = company_name.lower().replace(' ', '-')
        url = f"https://www.crunchbase.com/organization/{formatted_name}"
        
        logger.info(f"Attempting to get Crunchbase data for {company_name}")
        
        response = get_with_rate_limit(url, timeout=20)
        if not response or response.status_code != 200:
            logger.warning(f"Couldn't access Crunchbase for {company_name}")
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text content
        text_content = soup.get_text(separator=' ')
        
        # Clean text
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return {
            'source': 'crunchbase',
            'url': url,
            'content': text[:150000] if text else ""  # Limit content size
        }
        
    except Exception as e:
        logger.error(f"Error getting Crunchbase info for {company_name}: {e}")
        return None

def get_linkedin_info(company_name):
    """Scrape company information from LinkedIn"""
    try:
        # Format company name for search URL
        search_term = quote(company_name)
        url = f"https://www.linkedin.com/company/{search_term.lower().replace(' ', '-')}"
        
        logger.info(f"Attempting to get LinkedIn data for {company_name}")
        
        response = get_with_rate_limit(url, timeout=20)
        if not response or response.status_code != 200:
            logger.warning(f"Couldn't access LinkedIn for {company_name}")
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text content
        text_content = soup.get_text(separator=' ')
        
        # Clean text
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return {
            'source': 'linkedin',
            'url': url,
            'content': text[:150000] if text else ""  # Limit content size
        }
        
    except Exception as e:
        logger.error(f"Error getting LinkedIn info for {company_name}: {e}")
        return None

def get_indeed_info(company_name):
    """Get company information from Indeed"""
    try:
        search_term = quote(company_name)
        url = f"https://www.indeed.com/cmp/{search_term.lower().replace(' ', '-')}"
        
        logger.info(f"Attempting to get Indeed data for {company_name}")
        
        response = get_with_rate_limit(url, timeout=20)
        if not response or response.status_code != 200:
            logger.warning(f"Couldn't access Indeed for {company_name}")
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text content
        text_content = soup.get_text(separator=' ')
        
        # Clean text
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return {
            'source': 'indeed',
            'url': url,
            'content': text[:100000] if text else ""  # Limit content size
        }
        
    except Exception as e:
        logger.error(f"Error getting Indeed info for {company_name}: {e}")
        return None

def search_google_for_company(company_name):
    """Search Google for company information"""
    try:
        search_term = quote(f"{company_name} company information")
        url = f"https://www.google.com/search?q={search_term}"
        
        logger.info(f"Performing Google search for {company_name}")
        
        response = get_with_rate_limit(url, timeout=15)
        if not response or response.status_code != 200:
            logger.warning(f"Couldn't perform Google search for {company_name}")
            return []
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract search result links
        result_links = []
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if href.startswith('/url?') and 'google.com' not in href:
                # Extract actual URL from Google's redirect URL
                actual_url = href.split('?q=')[1].split('&')[0] if '?q=' in href else None
                if actual_url and actual_url not in result_links:
                    result_links.append(actual_url)
        
        # Return top 3 search results
        return result_links[:3]
        
    except Exception as e:
        logger.error(f"Error searching Google for {company_name}: {e}")
        return []

def get_third_party_info(company_name):
    """Get information from third-party sources"""
    third_party_data = []
    
    # Try specialized business information sites
    sources = [
        get_crunchbase_info(company_name),
        get_linkedin_info(company_name),
        get_indeed_info(company_name)
    ]
    
    # Add any successful results
    third_party_data.extend([source for source in sources if source])
    
    # If we didn't get enough data, try general Google search
    if len(third_party_data) < 2:
        google_results = search_google_for_company(company_name)
        
        # Scrape content from Google search results
        with ThreadPoolExecutor(max_workers=3) as executor:
            for url in google_results:
                content = scrape_url(url)
                if content:
                    third_party_data.append({
                        'source': 'google_search',
                        'url': url,
                        'content': content['content']
                    })
    
    logger.info(f"Found {len(third_party_data)} third-party sources for {company_name}")
    return third_party_data

# NEW FUNCTION: Fill missing information using CSE
def fill_missing_info_with_cse(company_name, company_info):
    """Use Google CSE to fill in 'Not found' fields"""
    missing_fields = []
    
    # Check which fields are missing
    for field, value in company_info.items():
        if value == "Not found" and field != "Company Name" and field != "Website":
            missing_fields.append(field)
    
    if not missing_fields:
        logger.info(f"No missing fields for {company_name}")
        return company_info
    
    logger.info(f"Attempting to fill {len(missing_fields)} missing fields for {company_name}")
    
    # Use CSE to search for missing fields
    cse_results = google_cse_search(company_name, missing_fields)
    
    if not cse_results:
        logger.warning(f"No CSE results found for {company_name}")
        return company_info
    
    # Extract content from CSE results
    extracted_content = extract_from_cse_results(cse_results)
    
    # Use Gemini to extract specific missing fields
    for field_data in extracted_content:
        field = field_data['field']
        content = field_data['content']
        
        # Skip if content is too short
        if len(content) < 100:
            continue
        
        prompt = f"""
        Based on the following information about {company_name}, extract the value for: {field}
        
        If you cannot find a specific value for {field}, respond with "Not found".
        Be concise and extract only the relevant information.
        
        Content:
        {content[:30000]}  # Limit prompt size
        """
        
        # Call Gemini with exponential backoff
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                extracted_value = response.text.strip()
                
                # Clean up response
                if "not found" in extracted_value.lower():
                    extracted_value = "Not found"
                
                # Update the field if we found something
                if extracted_value != "Not found":
                    company_info[field] = extracted_value
                    logger.info(f"Updated {field} for {company_name} using CSE")
                
                break
                
            except Exception as e:
                logger.error(f"Error extracting {field} using Gemini: {e}")
                if attempt < max_retries - 1:
                    logger.warning(f"Retrying extraction in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
    
    return company_info

def extract_company_info_combined(company_name, content, urls, third_party_data=None):
    """Extract all company information in a single LLM call to reduce API usage"""
    try:
        # Truncate content if too large
        max_content_length = 40000  # Adjusted to leave room for third-party data
        if len(content) > max_content_length:
            logger.warning(f"Content for {company_name} exceeds limit, truncating")
            content = content[:max_content_length]
        
        urls_text = "\n".join(urls)
        
        # Prepare third-party information
        third_party_text = ""
        if third_party_data:
            third_party_text = "THIRD-PARTY INFORMATION:\n\n"
            
            for i, source in enumerate(third_party_data):
                # Limit each third-party source content
                max_source_length = 10000
                source_content = source['content'][:max_source_length] if len(source['content']) > max_source_length else source['content']
                
                third_party_text += f"Source {i+1} ({source['source']}): {source['url']}\n"
                third_party_text += f"{source_content}\n\n"
        
        # Truncate third-party text if needed
        max_third_party_length = 30000
        if len(third_party_text) > max_third_party_length:
            logger.warning(f"Third-party content for {company_name} exceeds limit, truncating")
            third_party_text = third_party_text[:max_third_party_length]
        
        prompt = f"""
        I need you to extract structured business information about the company {company_name}.
        
        Based on the following content scraped from their website AND additional third-party sources, extract these specific pieces of information:
        
        - Website URL
        - Description (short summary of what they do)
        - Software Classification
        - Enterprise Grade Classification (enterprise or SMB)
        - Industry
        - Customers (list notable ones)
        - Employee Count
        - Investors
        - Geography (regions served)
        - Parent Company (if any)
        - Address Information (street, postal code, city, country)
        - Contact Information (email, phone)
        - Financial Information (revenue/funding if available)
        
        For each item, if the information is not found in ANY of the provided content, respond with "Not found".
        Format your response as a JSON object with these exact field names matching the database:
        {{
          "Company Name": "{company_name}",
          "Website": "",
          "Description": "",
          "Software Classification": "",
          "Enterprise Grade Classification": "",
          "Industry": "",
          "Geography": "",
          "Street Address": "",
          "City": "",
          "Postal Code": "",
          "Country": "",
          "Phone": "",
          "Email": "",
          "Employee Count": "",
          "Customers": "",
          "Investors": "",
          "Parent Company": "",
          "Financial Info": ""
        }}
        
        Content from {company_name}'s website:
        {content}
        
        Website URLs scraped:
        {urls_text}
        
        {third_party_text}
        """
        
        # Exponential backoff for Gemini API
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                answer = response.text.strip()
                
                # Extract JSON from response
                pattern = r'\{[\s\S]*\}'
                match = re.search(pattern, answer)
                if match:
                    json_str = match.group(0)
                    try:
                        result = json.loads(json_str)
                        # Make sure company name is included
                        result["Company Name"] = company_name
                        return result
                    except json.JSONDecodeError:
                        logger.error(f"Error parsing JSON response for {company_name}")
                        
                # Retry or fallback in case of failure
                if attempt < max_retries - 1:
                    logger.warning(f"Retrying extraction for {company_name} in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                        
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                if attempt < max_retries - 1:
                    logger.warning(f"Retrying extraction for {company_name} in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
        
        # Fallback in case all attempts fail
        result = {
            "Company Name": company_name, 
            "Website": urls[0] if urls else "Not found",
            "Description": "Not found",
            "Software Classification": "Not found",
            "Enterprise Grade Classification": "Not found",
            "Industry": "Not found",
            "Geography": "Not found",
            "Street Address": "Not found",
            "City": "Not found",
            "Postal Code": "Not found",
            "Country": "Not found",
            "Phone": "Not found",
            "Email": "Not found",
            "Employee Count": "Not found",
            "Customers": "Not found",
            "Investors": "Not found",
            "Parent Company": "Not found",
            "Financial Info": "Not found"
        }
            
        return result
        
    except Exception as e:
        logger.error(f"Error in extract_company_info_combined: {e}")
        return {"Company Name": company_name, "Error": str(e)}

# Modified process_company function with CSE integration
def process_company(company_name, website_url):
    logger.info(f"Processing company: {company_name}")
    
    try:
        # Ensure URL has proper format
        if not website_url.startswith('http'):
            website_url = 'https://' + website_url
        
        # Step 1: Get only important pages instead of all links
        relevant_urls = get_important_pages(company_name, website_url)
        
        # Step 2: Scrape content from relevant URLs (with limited concurrency to avoid overloading)
        logger.info(f"Scraping {len(relevant_urls)} important URLs for {company_name}...")
        scraped_contents = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            scraped_contents = list(executor.map(scrape_url, relevant_urls))
        
        # Step 3: Process content more efficiently
        logger.info(f"Processing content for {company_name}...")
        all_content = ""
        url_content_map = {}
        
        for content_dict in scraped_contents:
            if content_dict and content_dict['content']:
                url = content_dict['url']
                content = content_dict['content']
                
                # Store mapping of URL to content for reference
                url_content_map[url] = content
                
                # Append to combined content with URL reference
                page_info = f"\n\n--- Content from {url} ---\n\n"
                all_content += page_info + content
        
        # Step 4: Get third-party information
        logger.info(f"Getting third-party information for {company_name}...")
        third_party_data = get_third_party_info(company_name)
        
        # Step 5: Use a single combined prompt with both website and third-party data
        if all_content or third_party_data:
            company_info = extract_company_info_combined(company_name, all_content, relevant_urls, third_party_data)
            logger.info(f"Checking for missing information for {company_name}...")
            missing_count = sum(1 for value in company_info.values() if value == "Not found")
            
            if missing_count > 0 and GOOGLE_CSE_API_KEY != "YOUR_GOOGLE_API_KEY" and GOOGLE_CSE_ID != "YOUR_CSE_ID":
                logger.info(f"Found {missing_count} missing fields, attempting to fill with CSE")
                company_info = fill_missing_info_with_cse(company_name, company_info)
            
            # Step 6: Save company information to Supabase
            # Add source information
            # company_info["Data Sources"] = ["Company Website"]
            # if third_party_data:
            #     company_info["Data Sources"].extend([source['source'] for source in third_party_data])
            # if missing_count > 0:
            #     company_info["Data Sources"].append("Google CSE")
            # company_info["Data Sources"] = ", ".join(set(company_info["Data Sources"]))
            
            supabase.table("company_information").insert(company_info).execute()
            
            return company_info
        else:
            logger.warning(f"No content found for {company_name}")
            
            # Try using CSE directly if no website content was found
            if GOOGLE_CSE_API_KEY != "YOUR_GOOGLE_API_KEY" and GOOGLE_CSE_ID != "YOUR_CSE_ID":
                logger.info(f"Attempting to get information for {company_name} using CSE only")
                
                # Create a basic company info structure
                company_info = {
                    "Company Name": company_name,
                    "Website": website_url,
                    "Description": "Not found",
                    "Software Classification": "Not found",
                    "Enterprise Grade Classification": "Not found",
                    "Industry": "Not found",
                    "Geography": "Not found",
                    "Street Address": "Not found",
                    "City": "Not found",
                    "Postal Code": "Not found",
                    "Country": "Not found",
                    "Phone": "Not found",
                    "Email": "Not found",
                    "Employee Count": "Not found",
                    "Customers": "Not found",
                    "Investors": "Not found",
                    "Parent Company": "Not found",
                    "Financial Info": "Not found"
                }
                
                # Use CSE to fill in all fields
                fields = [
                    "Description", "Industry", "Employees", "Revenue", 
                    "Customers", "Location", "Contact", "Investors"
                ]
                
                # Get CSE results
                cse_results = google_cse_search(company_name, fields)
                extracted_content = extract_from_cse_results(cse_results)
                
                # Combine all content for a single Gemini call
                combined_content = f"Information about {company_name}:\n\n"
                for field_data in extracted_content:
                    combined_content += f"--- {field_data['field']} ---\n"
                    combined_content += field_data['content'][:10000] + "\n\n"
                
                # Use Gemini to extract structured information
                prompt = f"""
                I need to extract structured business information about {company_name} from search results.
                
                Based on the following content, extract these specific pieces of information:
                
                - Description (short summary of what they do)
                - Software Classification
                - Enterprise Grade Classification (enterprise or SMB)
                - Industry
                - Customers (list notable ones)
                - Employee Count
                - Investors
                - Geography (regions served)
                - Parent Company (if any)
                - Address Information (street, postal code, city, country)
                - Contact Information (email, phone)
                - Financial Information (revenue/funding if available)
                
                For each item, if the information is not found, respond with "Not found".
                Format your response as a JSON object with these exact field names:
                {{
                  "Description": "",
                  "Software Classification": "",
                  "Enterprise Grade Classification": "",
                  "Industry": "",
                  "Geography": "",
                  "Street Address": "",
                  "City": "",
                  "Postal Code": "",
                  "Country": "",
                  "Phone": "",
                  "Email": "",
                  "Employee Count": "",
                  "Customers": "",
                  "Investors": "",
                  "Parent Company": "",
                  "Financial Info": ""
                }}
                
                Content from search results:
                {combined_content[:50000]}
                """
                
                try:
                    response = model.generate_content(prompt)
                    answer = response.text.strip()
                    
                    # Extract JSON from response
                    pattern = r'\{[\s\S]*\}'
                    match = re.search(pattern, answer)
                    if match:
                        json_str = match.group(0)
                        try:
                            cse_result = json.loads(json_str)
                            # Update company_info with CSE results
                            for key, value in cse_result.items():
                                if key in company_info and value != "Not found":
                                    company_info[key] = value
                                    
                            # Add data source
                            # company_info["Data Sources"] = "Google CSE"
                            
                            # Save to Supabase
                            supabase.table("company_information").insert(company_info).execute()
                            
                            return company_info
                        except json.JSONDecodeError:
                            logger.error(f"Error parsing JSON from CSE response for {company_name}")
                except Exception as e:
                    logger.error(f"Error extracting info from CSE using Gemini: {e}")
            
            return {'Company Name': company_name, 'Website': website_url, 'Error': 'No content found'}
            
    except Exception as e:
        logger.error(f"Error processing company {company_name}: {str(e)}")
        return {'Company Name': company_name, 'Website': website_url, 'Error': str(e)}

# Function to check if company exists in database
def company_exists_in_db(company_name):
    try:
        result = supabase.table("company_information").select("*").eq("Company Name", company_name).execute()
        return len(result.data) > 0
    except Exception as e:
        logger.error(f"Error checking if company exists: {e}")
        return False

# Main function to process input Excel and create output Excel
def main(input_file, resume_processing=False):
    # Setup Supabase
    setup_supabase()
    
    # Check Google CSE configuration
    if (GOOGLE_CSE_API_KEY == "YOUR_GOOGLE_API_KEY" or GOOGLE_CSE_ID == "YOUR_CSE_ID"):
        logger.warning("Google CSE API key or CSE ID not configured. Some functionality will be limited.")
        print("""
        ⚠️ Google Custom Search Engine not configured.
        For improved results, set these environment variables:
        - GOOGLE_CSE_API_KEY: Your Google API key with CSE enabled
        - GOOGLE_CSE_ID: Your Custom Search Engine ID
        
        Visit https://developers.google.com/custom-search/v1/overview to set up CSE.
        """)
    else:
        logger.info("Google CSE configured. Will use for enhanced data collection.")
    
    # Read input Excel
    df = pd.read_excel(input_file)
    
    if 'Company Name' not in df.columns or 'Website' not in df.columns:
        raise ValueError("Input Excel must contain 'Company Name' and 'Website' columns")
    
    # Create a backup of previous results if resuming
    if resume_processing:
        output_file = 'company_information_results.xlsx'
        if os.path.exists(output_file):
            backup_file = f'company_information_results_backup_{int(time.time())}.xlsx'
            os.rename(output_file, backup_file)
            logger.info(f"Created backup of previous results at {backup_file}")
            
            # Load previous results
            previous_results = pd.read_excel(backup_file)
            processed_companies = set(previous_results['Company Name'])
            logger.info(f"Found {len(processed_companies)} previously processed companies")
        else:
            processed_companies = set()
    else:
        processed_companies = set()
    
    # Process each company
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        company_name = row['Company Name']
        website = row['Website']
        
        # Skip already processed companies if resuming
        if company_name in processed_companies:
            logger.info(f"Skipping already processed company: {company_name}")
            # Get previous result
            if resume_processing:
                previous_result = previous_results[previous_results['Company Name'] == company_name].to_dict('records')[0]
                results.append(previous_result)
            continue
            
        # Skip if already in database
        if company_exists_in_db(company_name):
            logger.info(f"Company already exists in database: {company_name}")
            # Get from database
            db_result = supabase.table("company_information").select("*").eq("Company Name", company_name).execute()
            if db_result.data:
                results.append(db_result.data[0])
            continue
        
        company_info = process_company(company_name, website)
        results.append(company_info)
        
        # Save intermediate results after each company
        intermediate_df = pd.DataFrame(results)
        intermediate_file = 'company_information_results_intermediate.xlsx'
        intermediate_df.to_excel(intermediate_file, index=False)
        
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    # Save to Excel
    output_file = 'company_information_results.xlsx'
    output_df.to_excel(output_file, index=False)
    
    logger.info(f"Processing complete. Results saved to {output_file}")
    return output_file

if __name__ == "__main__":
    # Display welcome message with instructions for CSE setup
    print("""
    🔍 Company Information Scraper with Google CSE Integration
    
    This tool extracts detailed company information from company websites and third-party sources.
    For optimal results, configure a Google Custom Search Engine with the following domains:
    - linkedin.com
    - crunchbase.com
    - glassdoor.com
    - indeed.com
    - bloomberg.com
    - zoominfo.com
    - dnb.com
    - hoovers.com
    
    Then set these environment variables:
    export GOOGLE_CSE_API_KEY="your_api_key"
    export GOOGLE_CSE_ID="your_cse_id"
    """)
    
    input_file = input("Enter the path to the input Excel file: ")
    resume = input("Resume processing previous results? (y/n): ").lower() == 'y'
    main(input_file, resume_processing=resume)