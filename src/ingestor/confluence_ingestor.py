from atlassian import Confluence
from ingestor import Ingestor
from bs4 import BeautifulSoup
import os


class ConfluenceIngester(Ingestor):
    def __init__(self, data_path='advendio_pages'):
        # data_path can be local or remote
        self.data_path = data_path
        self.documents = []

    def ingest_confluence_data(self, seed_url='https://advendio.atlassian.net', page_name="SO"):
        # Set up Confluence API connection
        confluence = Confluence(url=seed_url)
        pages = confluence.get_all_pages_from_space(page_name)
        # Create a directory to store the downloaded pages
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        # Download each page
        for page in pages:
            page_id = page['id']
            page_title = page['title']
            page_filename = page_title.replace(' ', '_') + '.html'
            page_content = confluence.get_page_by_id(page_id, expand='body.storage')[
                'body']['storage']['value']
            try:
                with open(f'{self.data_path}/{page_filename}', 'w') as f:
                    f.write(page_content)
            except:
                pass

    def format_documents(self):
        for filename in os.listdir('advendio_pages'):
            f = os.path.join('advendio_pages', filename)
            with open(f, 'r', encoding='utf-8') as file:
                html_content = file.read()
                soup = BeautifulSoup(html_content, "lxml")
                text_content = soup.get_text(separator=" ", strip=True)
                self.documents.append(text_content)
