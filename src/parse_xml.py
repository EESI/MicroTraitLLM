import os
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
import re
import subprocess
from nltk.tokenize import sent_tokenize
import json

class PMCArticleExtractor:
    def __init__(self, ftp_downloads_dir: str):
        """
        Initialize the extractor with the FTP_Downloads directory path.
        
        Args:
            ftp_downloads_dir: Path to the FTP_Downloads folder containing XML files
        """
        self.ftp_downloads_dir = Path(ftp_downloads_dir)
    
    def find_article_type(self, pmcid: str) -> Optional[str]:
        """
        Search CSV files to determine if an article is in oa_comm or oa_noncomm.
        
        Args:
            pmcid: PMC ID to search for
            
        Returns:
            'oa_comm' or 'oa_noncomm' if found, None otherwise
        """
        csv_files = [
            (self.ftp_downloads_dir / "oa_comm.filelist.csv", "oa_comm"),
            (self.ftp_downloads_dir / "oa_noncomm.filelist.csv", "oa_noncomm"),
            (self.ftp_downloads_dir / "phe_timebound.filelist.csv", "phe_timebound")
        ]
        
        for csv_path, article_type in csv_files:
            if not csv_path.exists():
                continue
                
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('AccessionID', '').strip() == pmcid:
                            return article_type
            except Exception as e:
                print(f"    Error reading {csv_path.name}: {e}")
        
        return None
    
    def download_article_from_s3(self, pmcid: str, article_type: str) -> bool:
        """
        Download an article from AWS S3 using the aws CLI.
        
        Args:
            pmcid: PMC ID to download
            article_type: Either 'oa_comm' or 'oa_noncomm'
            
        Returns:
            True if download successful, False otherwise
        """
        s3_path = f"s3://pmc-oa-opendata/{article_type}/xml/all/{pmcid}.xml"
        local_path = self.ftp_downloads_dir / f"{pmcid}.xml"
        
        print(f"    Downloading from S3: {s3_path}")
        
        try:
            # Run aws s3 cp command
            result = subprocess.run(
                ["aws", "s3", "cp", s3_path, str(local_path), "--no-sign-request"],
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            if result.returncode == 0:
                print(f"    ✓ Downloaded successfully")
                return True
            else:
                print(f"    ✗ Download failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"    ✗ Download timed out")
            return False
        except FileNotFoundError:
            print(f"    ✗ AWS CLI not found. Please install it: pip install awscli")
            return False
        except Exception as e:
            print(f"    ✗ Download error: {e}")
            return False
    
    def extract_metadata_from_xml(self, xml_content: str) -> Dict:
        """
        Extract metadata from PMC XML content.
        
        Args:
            xml_content: The XML content as a string
            
        Returns:
            Dictionary with title, authors, journal, publication_date, volume, issue, first_page, pages, doi
        """
        metadata = {
            'title': '',
            'authors': [],
            'journal': '',
            'publication_date': '',
            'volume': '',
            'issue': '',
            'first_page': '',
            'pages': '',
            'doi': '',
        }
        
        try:
            root = ET.fromstring(xml_content)
            
            # Extract title
            title_elem = root.find(".//article-title")
            if title_elem is not None:
                metadata['title'] = ''.join(title_elem.itertext()).strip()
            
            # Extract authors
            authors = []
            for contrib in root.findall(".//contrib[@contrib-type='author']"):
                surname = contrib.find(".//surname")
                given_names = contrib.find(".//given-names")
                if surname is not None:
                    author_name = surname.text or ''
                    if given_names is not None and given_names.text:
                        author_name = f"{given_names.text} {author_name}"
                    authors.append(author_name.strip())
            metadata['authors'] = authors
            
            # Extract journal name
            journal_title = root.find(".//journal-title")
            if journal_title is not None:
                metadata['journal'] = journal_title.text or ''
            
            # Extract publication date
            pub_date = root.find(".//pub-date[@pub-type='epub']") or root.find(".//pub-date[@pub-type='ppub']") or root.find(".//pub-date")
            if pub_date is not None:
                year = pub_date.find("year")
                month = pub_date.find("month")
                day = pub_date.find("day")
                date_parts = []
                if year is not None and year.text:
                    date_parts.append(year.text)
                if month is not None and month.text:
                    date_parts.append(month.text.zfill(2))
                if day is not None and day.text:
                    date_parts.append(day.text.zfill(2))
                metadata['publication_date'] = '-'.join(date_parts)
            
            # Extract volume
            volume = root.find(".//volume")
            if volume is not None:
                metadata['volume'] = volume.text or ''
            
            # Extract issue
            issue = root.find(".//issue")
            if issue is not None:
                metadata['issue'] = issue.text or ''
            
            # Extract first page and page range
            fpage = root.find(".//fpage")
            lpage = root.find(".//lpage")
            if fpage is not None:
                metadata['first_page'] = fpage.text or ''
                if lpage is not None and lpage.text:
                    metadata['pages'] = f"{fpage.text}-{lpage.text}"
                else:
                    metadata['pages'] = fpage.text or ''
            
            # Extract DOI
            article_id_doi = root.find(".//article-id[@pub-id-type='doi']")
            if article_id_doi is not None:
                metadata['doi'] = article_id_doi.text or ''
            
        except ET.ParseError as e:
            print(f"XML parsing error: {e}")
        
        return metadata
    
    def extract_text_from_xml(self, xml_content: str) -> str:
        """
        Extract article text from PMC XML content, excluding supplementary materials.
        
        Args:
            xml_content: The XML content as a string
            
        Returns:
            Extracted article text
        """
        try:
            root = ET.fromstring(xml_content)
            
            # Tags to exclude (supplementary materials, supporting info, etc.)
            exclude_tags = {
                'supplementary-material',
                'media',
                'fig',
                'table-wrap',
                'disp-formula',
                'caption',
                'label',
                'sec-meta',
                'ack',  # acknowledgments
                'ref-list',  # references
                'fn-group',  # footnotes
            }
            
            def get_text_recursive(element, exclude_tags_set):
                """Recursively extract text, skipping excluded tags."""
                text_parts = []
                
                # Skip if this is an excluded tag
                if element.tag in exclude_tags_set:
                    return []
                
                # Add this element's text
                if element.text:
                    text_parts.append(element.text.strip())
                
                # Process children
                for child in element:
                    text_parts.extend(get_text_recursive(child, exclude_tags_set))
                    # Add tail text (text after the child tag)
                    if child.tail:
                        text_parts.append(child.tail.strip())
                
                return text_parts
            
            text_parts = []
            
            # Title
            title_elements = root.findall(".//article-title")
            for elem in title_elements:
                title_text = ' '.join(get_text_recursive(elem, exclude_tags)).strip()
                if title_text:
                    text_parts.append(title_text)
            
            # Abstract
            abstract_elements = root.findall(".//abstract")
            for abstract in abstract_elements:
                abstract_text = ' '.join(get_text_recursive(abstract, exclude_tags)).strip()
                if abstract_text:
                    text_parts.append(abstract_text)
            
            # Body text (excluding back matter)
            body_elements = root.findall(".//body")
            for body in body_elements:
                body_text = ' '.join(get_text_recursive(body, exclude_tags)).strip()
                if body_text:
                    text_parts.append(body_text)
            
            # Join and clean up the text
            full_text = ' '.join(text_parts)
            
            # Remove common supplementary material phrases
            cleanup_patterns = [
                r'Supporting Information.*?(?=\n|$)',
                r'Dataset S\d+.*?(?=\n|$)',
                r'Figure S\d+.*?(?=\n|$)',
                r'Table S\d+.*?(?=\n|$)',
                r'Click here for additional data file\.',
                r'\(\d+\.?\d*\s*(?:MB|KB|GB)\s*(?:ZIP|PDF|TXT|DOC|XLS)\)',
                r'Supplementary (?:Material|Data|Information|File).*?(?=\n|$)',
            ]
            
            for pattern in cleanup_patterns:
                full_text = re.sub(pattern, '', full_text, flags=re.IGNORECASE)
            
            # Clean up extra whitespace
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            return full_text
        
        except ET.ParseError as e:
            print(f"XML parsing error: {e}")
            return ""
    
    def extract_articles(self, id_list: List[str]) -> List[Dict]:
        """
        Extract articles for the given list of PMC IDs directly from the FTP_Downloads folder.
        
        Args:
            id_list: List of PMC IDs (e.g., "PMC176545") to extract
            
        Returns:
            List of dictionaries containing text and metadata
        """
        results = []
        
        print(f"Extracting {len(id_list)} articles from {self.ftp_downloads_dir}...")
        
        for id_val in id_list:
            # Construct the XML filename
            xml_filename = f"{id_val}.xml"
            xml_path = self.ftp_downloads_dir / xml_filename
            
            print(f"  Looking for {xml_filename}...", end=' ')
            
            # If file doesn't exist, try to download it
            if not xml_path.exists():
                print(f"NOT FOUND locally")
                
                # Search CSV files to determine article type
                article_type = self.find_article_type(id_val)
                
                if article_type:
                    print(f"    Found in {article_type} catalog")
                    # Try to download from S3
                    if self.download_article_from_s3(id_val, article_type):
                        # File should now exist, continue to extraction
                        pass
                    else:
                        print(f"    Skipping {id_val}")
                        continue
                else:
                    print(f"    Not found in any catalog, skipping")
                    continue
            else:
                print(f"Found locally")
            
            # Read and parse the XML file
            try:
                with open(xml_path, 'r', encoding='utf-8') as f:
                    xml_content = f.read()
                
                text = self.extract_text_from_xml(xml_content)
                metadata = self.extract_metadata_from_xml(xml_content)
                
                results.append({
                    'text': text,
                    'meta': metadata
                })
                
                print(f"    ✓ Extracted successfully")
                
            except Exception as e:
                print(f"    ✗ Error reading/parsing XML: {e}")
        
        print(f"\n{'='*80}")
        print(f"Successfully extracted {len(results)} articles out of {len(id_list)} requested")
        with open('all_articles.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
