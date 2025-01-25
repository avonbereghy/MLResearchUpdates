import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from accelerate.test_utils.testing import get_backend
from pdf2image import convert_from_path
from PIL import Image
from dotenv import load_dotenv


def media_path_to_pil_images(pdf_path, dpi=300, transforms=None, num_pages=None):
    if pdf_path.endswith('.pdf'):
        if num_pages is not None:
            images = convert_from_path(pdf_path, dpi=dpi)[0:num_pages]  # Convert pages to PIL Images
        else:
            images = convert_from_path(pdf_path, dpi=dpi)
    else:
        images = [Image.open(pdf_path)]

    return images

def get_paper_links(papers_url, base_url):
    print(f"Retrieving papers page...{papers_url}")
    response = requests.get(papers_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        paper_data = {}
        for a in soup.find_all('a', href=True, class_="line-clamp-3 cursor-pointer text-balance"):
            title = a.get_text(strip=True)  # title
            paper_url = f"{base_url}{a['href']}"  # paper link
            paper_data[paper_url] = title
    
        return paper_data
    else:
        print(f"Failed to retrieve papers page, status code: {response.status_code}")
        return {}

def get_pdf_link(paper_url):
    response = requests.get(paper_url, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        # Find the first <a> tag with an href containing "arxiv.org/pdf/"
        pdf_link_tag = soup.find('a', href=lambda href: href and "arxiv.org/pdf/" in href)
        if pdf_link_tag:
            return pdf_link_tag['href']
        else:
            print(f"No arXiv PDF link found on {paper_url}")
    else:
        print(f"Failed to retrieve paper page: {paper_url}, status code: {response.status_code}")
    return None


# Function to download a PDF
def download_pdf(pdf_url, title, pdf_folder):
    pdf_path = os.path.join(pdf_folder, title+'.pdf')
    if os.path.exists(pdf_path):
        print(f"Skipping, already downloaded: {title}")
        return
    response = requests.get(pdf_url, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
    if response.status_code == 200:
        with open(pdf_path, "wb") as pdf_file:
            for chunk in response.iter_content(1024):
                pdf_file.write(chunk)
        print(f"Downloaded: {title}")
    else:
        print(f"Failed to download: {pdf_url}")

def download_pdfs(date:str = None):
    "https://huggingface.co/papers"
    
    if date == None:
        # current date in YYYY-MM-DD format
        date = datetime.today().strftime('%Y-%m-%d')
    
    # Define the Hugging Face papers URL
    base_url = "https://huggingface.co"
    
    # papers_url = f"{base_url}/papers?date={date}"
    papers_url = f"{base_url}/papers"

    # Create a new folder to save PDFs
    pdf_folder = f"{os.getenv('PAPER_PATH')}/{date}"
    print(pdf_folder)
    os.makedirs(pdf_folder, exist_ok=True)

    # Get paper links
    paper_links = get_paper_links(papers_url, base_url)
    print(f"Found {len(paper_links)} papers")

    for paper_link, title in paper_links.items():
        # print(paper_link)
        pdf_link = get_pdf_link(paper_link)
        print('\t', pdf_link)
        if pdf_link:
            download_pdf(pdf_link, title, pdf_folder)
            

if __name__ == "__main__":
    ask_user = input("Download papers? (y/n): ")
    if ask_user.lower() == 'y':
        ask_user_again = input("Are you sure? (y/n): ")
        if ask_user_again.lower() == 'y':
            # Load .env file
            load_dotenv()
            device, _, _ = get_backend()

            download_pdfs()
    else:
        print("You may have ran the wrong file")