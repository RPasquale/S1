"""
Data Extraction API for Agent Architecture Components

This API continuously extracts and monitors data from:
1. Stanford DSPy (https://github.com/stanfordnlp/dspy)
2. Hugging Face TRL (https://huggingface.co/docs/trl/en/index)
3. DeepSeek AI Models (https://huggingface.co/deepseek-ai)
4. Reason-ModernColBERT (https://huggingface.co/lightonai/Reason-ModernColBERT)
"""

import os
import json
import time
import requests
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import schedule
from pathlib import Path

import git
from bs4 import BeautifulSoup
from huggingface_hub import HfApi, list_models, model_info, list_datasets


@dataclass
class DataSource:
    """Configuration for a data source"""
    name: str
    url: str
    source_type: str  # 'github', 'huggingface_model', 'huggingface_docs'
    extraction_method: str
    update_frequency: int  # hours
    last_updated: Optional[datetime] = None
    enabled: bool = True


@dataclass
class ExtractedData:
    """Container for extracted data"""
    source_name: str
    data_type: str
    content: str
    metadata: Dict[str, Any]
    hash: str
    timestamp: datetime
    url: str


class DatabaseManager:
    """Manages the SQLite database for storing extracted data"""
    
    def __init__(self, db_path: str = "data_extraction.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                url TEXT NOT NULL,
                source_type TEXT NOT NULL,
                extraction_method TEXT NOT NULL,
                update_frequency INTEGER NOT NULL,
                last_updated TIMESTAMP,
                enabled BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extracted_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                data_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                hash TEXT UNIQUE NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                url TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_name) REFERENCES data_sources (name)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extraction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash ON extracted_data (hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_timestamp ON extracted_data (source_name, timestamp)")
        
        conn.commit()
        conn.close()
    
    def add_data_source(self, source: DataSource):
        """Add or update a data source"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO data_sources 
            (name, url, source_type, extraction_method, update_frequency, last_updated, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (source.name, source.url, source.source_type, source.extraction_method, 
              source.update_frequency, source.last_updated, source.enabled))
        
        conn.commit()
        conn.close()
    
    def get_data_sources(self) -> List[DataSource]:
        """Get all enabled data sources"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name, url, source_type, extraction_method, update_frequency, last_updated, enabled
            FROM data_sources WHERE enabled = TRUE
        """)
        
        sources = []
        for row in cursor.fetchall():
            last_updated = datetime.fromisoformat(row[5]) if row[5] else None
            sources.append(DataSource(
                name=row[0], url=row[1], source_type=row[2], 
                extraction_method=row[3], update_frequency=row[4], 
                last_updated=last_updated, enabled=row[6]
            ))
        
        conn.close()
        return sources
    
    def store_extracted_data(self, data: ExtractedData) -> bool:
        """Store extracted data, return True if new data was stored"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO extracted_data 
                (source_name, data_type, content, metadata, hash, timestamp, url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (data.source_name, data.data_type, data.content, 
                  json.dumps(data.metadata), data.hash, data.timestamp, data.url))
            
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Data already exists (hash collision)
            return False
        finally:
            conn.close()
    
    def log_extraction(self, source_name: str, status: str, message: str = None):
        """Log extraction attempt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO extraction_logs (source_name, status, message)
            VALUES (?, ?, ?)
        """, (source_name, status, message))
        
        conn.commit()
        conn.close()
    
    def update_source_timestamp(self, source_name: str):
        """Update the last_updated timestamp for a source"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE data_sources SET last_updated = ? WHERE name = ?
        """, (datetime.now(), source_name))
        
        conn.commit()
        conn.close()


class GitHubExtractor:
    """Extracts data from GitHub repositories"""
    
    def __init__(self, data_dir: str = "extracted_data/github"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_repository_data(self, repo_url: str, source_name: str) -> List[ExtractedData]:
        """Extract comprehensive data from a GitHub repository"""
        extracted_data = []
        
        try:
            # Clone or update repository
            repo_path = self.data_dir / source_name
            if repo_path.exists():
                repo = git.Repo(repo_path)
                repo.remotes.origin.pull()
                print(f"Updated repository: {source_name}")
            else:
                repo = git.Repo.clone_from(repo_url, repo_path)
                print(f"Cloned repository: {source_name}")
            
            # Extract different types of data
            extracted_data.extend(self._extract_code_files(repo_path, source_name, repo_url))
            extracted_data.extend(self._extract_documentation(repo_path, source_name, repo_url))
            extracted_data.extend(self._extract_examples(repo_path, source_name, repo_url))
            extracted_data.extend(self._extract_tests(repo_path, source_name, repo_url))
            extracted_data.extend(self._extract_configs(repo_path, source_name, repo_url))
            
        except Exception as e:
            print(f"Error extracting from {repo_url}: {e}")
        
        return extracted_data
    
    def _extract_code_files(self, repo_path: Path, source_name: str, repo_url: str) -> List[ExtractedData]:
        """Extract Python source code files"""
        extracted_data = []
        
        for py_file in repo_path.rglob("*.py"):
            if any(skip in str(py_file) for skip in ['.git', '__pycache__', '.pytest_cache']):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                # Skip empty or very small files
                if len(content.strip()) < 50:
                    continue
                
                metadata = {
                    "file_path": str(py_file.relative_to(repo_path)),
                    "file_size": len(content),
                    "lines": len(content.splitlines()),
                    "language": "python"
                }
                
                hash_obj = hashlib.sha256(content.encode()).hexdigest()
                
                extracted_data.append(ExtractedData(
                    source_name=source_name,
                    data_type="source_code",
                    content=content,
                    metadata=metadata,
                    hash=hash_obj,
                    timestamp=datetime.now(),
                    url=f"{repo_url}/blob/main/{metadata['file_path']}"
                ))
            except Exception as e:
                print(f"Error reading {py_file}: {e}")
        
        return extracted_data
    
    def _extract_documentation(self, repo_path: Path, source_name: str, repo_url: str) -> List[ExtractedData]:
        """Extract documentation files"""
        extracted_data = []
        doc_patterns = ["*.md", "*.rst", "*.txt"]
        
        for pattern in doc_patterns:
            for doc_file in repo_path.rglob(pattern):
                if any(skip in str(doc_file) for skip in ['.git', 'node_modules']):
                    continue
                
                try:
                    content = doc_file.read_text(encoding='utf-8', errors='ignore')
                    
                    if len(content.strip()) < 100:
                        continue
                    
                    metadata = {
                        "file_path": str(doc_file.relative_to(repo_path)),
                        "file_size": len(content),
                        "file_type": doc_file.suffix[1:],
                        "is_readme": doc_file.name.lower().startswith('readme')
                    }
                    
                    hash_obj = hashlib.sha256(content.encode()).hexdigest()
                    
                    extracted_data.append(ExtractedData(
                        source_name=source_name,
                        data_type="documentation",
                        content=content,
                        metadata=metadata,
                        hash=hash_obj,
                        timestamp=datetime.now(),
                        url=f"{repo_url}/blob/main/{metadata['file_path']}"
                    ))
                except Exception as e:
                    print(f"Error reading {doc_file}: {e}")
        
        return extracted_data
    
    def _extract_examples(self, repo_path: Path, source_name: str, repo_url: str) -> List[ExtractedData]:
        """Extract example files and tutorials"""
        extracted_data = []
        example_dirs = ["examples", "tutorials", "demos", "samples"]
        
        for example_dir in example_dirs:
            example_path = repo_path / example_dir
            if example_path.exists():
                for py_file in example_path.rglob("*.py"):
                    try:
                        content = py_file.read_text(encoding='utf-8', errors='ignore')
                        
                        if len(content.strip()) < 50:
                            continue
                        
                        metadata = {
                            "file_path": str(py_file.relative_to(repo_path)),
                            "file_size": len(content),
                            "example_type": example_dir,
                            "language": "python"
                        }
                        
                        hash_obj = hashlib.sha256(content.encode()).hexdigest()
                        
                        extracted_data.append(ExtractedData(
                            source_name=source_name,
                            data_type="example",
                            content=content,
                            metadata=metadata,
                            hash=hash_obj,
                            timestamp=datetime.now(),
                            url=f"{repo_url}/blob/main/{metadata['file_path']}"
                        ))
                    except Exception as e:
                        print(f"Error reading {py_file}: {e}")
        
        return extracted_data
    
    def _extract_tests(self, repo_path: Path, source_name: str, repo_url: str) -> List[ExtractedData]:
        """Extract test files"""
        extracted_data = []
        test_patterns = ["test_*.py", "*_test.py", "tests/*.py"]
        
        for pattern in test_patterns:
            for test_file in repo_path.rglob(pattern):
                try:
                    content = test_file.read_text(encoding='utf-8', errors='ignore')
                    
                    if len(content.strip()) < 50:
                        continue
                    
                    metadata = {
                        "file_path": str(test_file.relative_to(repo_path)),
                        "file_size": len(content),
                        "language": "python",
                        "test_type": "unit_test"
                    }
                    
                    hash_obj = hashlib.sha256(content.encode()).hexdigest()
                    
                    extracted_data.append(ExtractedData(
                        source_name=source_name,
                        data_type="test",
                        content=content,
                        metadata=metadata,
                        hash=hash_obj,
                        timestamp=datetime.now(),
                        url=f"{repo_url}/blob/main/{metadata['file_path']}"
                    ))
                except Exception as e:
                    print(f"Error reading {test_file}: {e}")
        
        return extracted_data
    
    def _extract_configs(self, repo_path: Path, source_name: str, repo_url: str) -> List[ExtractedData]:
        """Extract configuration files"""
        extracted_data = []
        config_files = ["pyproject.toml", "setup.py", "requirements.txt", "environment.yml", "Dockerfile"]
        
        for config_file in config_files:
            config_path = repo_path / config_file
            if config_path.exists():
                try:
                    content = config_path.read_text(encoding='utf-8', errors='ignore')
                    
                    metadata = {
                        "file_path": config_file,
                        "file_size": len(content),
                        "config_type": config_file.split('.')[-1] if '.' in config_file else config_file
                    }
                    
                    hash_obj = hashlib.sha256(content.encode()).hexdigest()
                    
                    extracted_data.append(ExtractedData(
                        source_name=source_name,
                        data_type="configuration",
                        content=content,
                        metadata=metadata,
                        hash=hash_obj,
                        timestamp=datetime.now(),
                        url=f"{repo_url}/blob/main/{config_file}"
                    ))
                except Exception as e:
                    print(f"Error reading {config_path}: {e}")
        
        return extracted_data


class HuggingFaceExtractor:
    """Extracts data from Hugging Face models and documentation"""
    
    def __init__(self):
        self.api = HfApi()
    
    def extract_model_data(self, model_name: str, source_name: str) -> List[ExtractedData]:
        """Extract comprehensive model data from Hugging Face"""
        extracted_data = []
        
        try:
            # Get model info
            info = model_info(model_name)
            
            # Extract model card
            if hasattr(info, 'cardData') and info.cardData:
                card_content = str(info.cardData)
                
                metadata = {
                    "model_name": model_name,
                    "model_type": getattr(info, 'pipeline_tag', 'unknown'),
                    "downloads": getattr(info, 'downloads', 0),
                    "likes": getattr(info, 'likes', 0),
                    "tags": getattr(info, 'tags', []),
                    "library_name": getattr(info, 'library_name', 'unknown')
                }
                
                hash_obj = hashlib.sha256(card_content.encode()).hexdigest()
                
                extracted_data.append(ExtractedData(
                    source_name=source_name,
                    data_type="model_card",
                    content=card_content,
                    metadata=metadata,
                    hash=hash_obj,
                    timestamp=datetime.now(),
                    url=f"https://huggingface.co/{model_name}"
                ))
            
            # Extract README if available
            try:
                readme_content = self.api.hf_hub_download(
                    repo_id=model_name,
                    filename="README.md",
                    repo_type="model"
                )
                
                with open(readme_content, 'r', encoding='utf-8') as f:
                    readme_text = f.read()
                
                metadata = {
                    "model_name": model_name,
                    "file_type": "readme",
                    "file_size": len(readme_text)
                }
                
                hash_obj = hashlib.sha256(readme_text.encode()).hexdigest()
                
                extracted_data.append(ExtractedData(
                    source_name=source_name,
                    data_type="model_readme",
                    content=readme_text,
                    metadata=metadata,
                    hash=hash_obj,
                    timestamp=datetime.now(),
                    url=f"https://huggingface.co/{model_name}/blob/main/README.md"
                ))
            except:
                pass  # README might not exist
            
            # Extract configuration files
            config_files = ["config.json", "tokenizer_config.json", "generation_config.json"]
            for config_file in config_files:
                try:
                    config_path = self.api.hf_hub_download(
                        repo_id=model_name,
                        filename=config_file,
                        repo_type="model"
                    )
                    
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_content = f.read()
                    
                    metadata = {
                        "model_name": model_name,
                        "file_type": config_file,
                        "file_size": len(config_content)
                    }
                    
                    hash_obj = hashlib.sha256(config_content.encode()).hexdigest()
                    
                    extracted_data.append(ExtractedData(
                        source_name=source_name,
                        data_type="model_config",
                        content=config_content,
                        metadata=metadata,
                        hash=hash_obj,
                        timestamp=datetime.now(),
                        url=f"https://huggingface.co/{model_name}/blob/main/{config_file}"
                    ))
                except:
                    pass  # Config file might not exist
        
        except Exception as e:
            print(f"Error extracting model data for {model_name}: {e}")
        
        return extracted_data
    
    def extract_documentation_data(self, docs_url: str, source_name: str) -> List[ExtractedData]:
        """Extract documentation from Hugging Face docs"""
        extracted_data = []
        
        try:
            response = requests.get(docs_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract main content
            main_content = soup.find('main') or soup.find('div', class_='content')
            if main_content:
                text_content = main_content.get_text(strip=True, separator='\n')
                
                metadata = {
                    "url": docs_url,
                    "content_length": len(text_content),
                    "extraction_method": "html_parsing"
                }
                
                hash_obj = hashlib.sha256(text_content.encode()).hexdigest()
                
                extracted_data.append(ExtractedData(
                    source_name=source_name,
                    data_type="documentation",
                    content=text_content,
                    metadata=metadata,
                    hash=hash_obj,
                    timestamp=datetime.now(),
                    url=docs_url
                ))
            
            # Extract code examples
            code_blocks = soup.find_all('code') + soup.find_all('pre')
            for i, code_block in enumerate(code_blocks):
                code_content = code_block.get_text(strip=True)
                
                if len(code_content) > 50:  # Filter out small code snippets
                    metadata = {
                        "url": docs_url,
                        "code_block_index": i,
                        "content_length": len(code_content)
                    }
                    
                    hash_obj = hashlib.sha256(code_content.encode()).hexdigest()
                    
                    extracted_data.append(ExtractedData(
                        source_name=source_name,
                        data_type="code_example",
                        content=code_content,
                        metadata=metadata,
                        hash=hash_obj,
                        timestamp=datetime.now(),
                        url=docs_url
                    ))
        
        except Exception as e:
            print(f"Error extracting documentation from {docs_url}: {e}")
        
        return extracted_data
    
    def extract_organization_models(self, org_name: str, source_name: str) -> List[ExtractedData]:
        """Extract all models from a Hugging Face organization"""
        extracted_data = []
        
        try:
            models = list_models(author=org_name)
            
            for model in models:
                model_data = self.extract_model_data(model.modelId, source_name)
                extracted_data.extend(model_data)
                
                # Add delay to respect rate limits
                time.sleep(0.5)
        
        except Exception as e:
            print(f"Error extracting models from organization {org_name}: {e}")
        
        return extracted_data


class DataExtractionAPI:
    """Main API for managing data extraction"""
    
    def __init__(self, db_path: str = "data_extraction.db"):
        self.db = DatabaseManager(db_path)
        self.github_extractor = GitHubExtractor()
        self.hf_extractor = HuggingFaceExtractor()
        self.setup_default_sources()
    
    def setup_default_sources(self):
        """Setup the four main data sources"""
        sources = [
            DataSource(
                name="dspy",
                url="https://github.com/stanfordnlp/dspy",
                source_type="github",
                extraction_method="repository_full",
                update_frequency=12  # 12 hours
            ),
            DataSource(
                name="trl_docs",
                url="https://huggingface.co/docs/trl/en/index",
                source_type="huggingface_docs",
                extraction_method="documentation_crawl",
                update_frequency=24  # 24 hours
            ),
            DataSource(
                name="deepseek_models",
                url="https://huggingface.co/deepseek-ai",
                source_type="huggingface_org",
                extraction_method="organization_models",
                update_frequency=6  # 6 hours
            ),
            DataSource(
                name="reason_moderncolbert",
                url="https://huggingface.co/lightonai/Reason-ModernColBERT",
                source_type="huggingface_model",
                extraction_method="model_full",
                update_frequency=12  # 12 hours
            )
        ]
        
        for source in sources:
            self.db.add_data_source(source)
    
    def extract_from_source(self, source: DataSource) -> int:
        """Extract data from a single source"""
        print(f"Extracting data from {source.name}...")
        extracted_count = 0
        
        try:
            if source.source_type == "github":
                extracted_data = self.github_extractor.extract_repository_data(
                    source.url, source.name
                )
            elif source.source_type == "huggingface_model":
                model_name = source.url.split('/')[-2] + '/' + source.url.split('/')[-1]
                extracted_data = self.hf_extractor.extract_model_data(
                    model_name, source.name
                )
            elif source.source_type == "huggingface_org":
                org_name = source.url.split('/')[-1]
                extracted_data = self.hf_extractor.extract_organization_models(
                    org_name, source.name
                )
            elif source.source_type == "huggingface_docs":
                extracted_data = self.hf_extractor.extract_documentation_data(
                    source.url, source.name
                )
            else:
                print(f"Unknown source type: {source.source_type}")
                return 0
            
            # Store extracted data
            for data in extracted_data:
                if self.db.store_extracted_data(data):
                    extracted_count += 1
            
            # Update source timestamp
            self.db.update_source_timestamp(source.name)
            self.db.log_extraction(source.name, "SUCCESS", f"Extracted {extracted_count} items")
            
        except Exception as e:
            error_msg = f"Error extracting from {source.name}: {e}"
            print(error_msg)
            self.db.log_extraction(source.name, "ERROR", error_msg)
        
        print(f"Extracted {extracted_count} new items from {source.name}")
        return extracted_count
    
    def run_extraction_cycle(self):
        """Run a full extraction cycle for all sources"""
        print("Starting data extraction cycle...")
        total_extracted = 0
        
        sources = self.db.get_data_sources()
        
        # Use ThreadPoolExecutor for parallel extraction
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_source = {}
            
            for source in sources:
                # Check if source needs updating
                if self.should_update_source(source):
                    future = executor.submit(self.extract_from_source, source)
                    future_to_source[future] = source
            
            # Collect results
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    count = future.result()
                    total_extracted += count
                except Exception as e:
                    print(f"Error in extraction for {source.name}: {e}")
        
        print(f"Extraction cycle complete. Total new items: {total_extracted}")
        return total_extracted
    
    def should_update_source(self, source: DataSource) -> bool:
        """Check if a source should be updated based on frequency"""
        if not source.last_updated:
            return True
        
        hours_since_update = (datetime.now() - source.last_updated).total_seconds() / 3600
        return hours_since_update >= source.update_frequency
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about extracted data"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Overall stats
        cursor.execute("SELECT COUNT(*) FROM extracted_data")
        total_items = cursor.fetchone()[0]
        
        # Stats by source
        cursor.execute("""
            SELECT source_name, COUNT(*) as count, MAX(timestamp) as last_update
            FROM extracted_data 
            GROUP BY source_name
        """)
        source_stats = {}
        for row in cursor.fetchall():
            source_stats[row[0]] = {
                "count": row[1],
                "last_update": row[2]
            }
        
        # Stats by data type
        cursor.execute("""
            SELECT data_type, COUNT(*) as count
            FROM extracted_data 
            GROUP BY data_type
        """)
        type_stats = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            "total_items": total_items,
            "source_stats": source_stats,
            "type_stats": type_stats,
            "last_generated": datetime.now().isoformat()
        }
    
    def search_extracted_data(self, query: str, source_name: str = None, 
                            data_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search through extracted data"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        sql = """
            SELECT source_name, data_type, content, metadata, timestamp, url
            FROM extracted_data 
            WHERE content LIKE ?
        """
        params = [f"%{query}%"]
        
        if source_name:
            sql += " AND source_name = ?"
            params.append(source_name)
        
        if data_type:
            sql += " AND data_type = ?"
            params.append(data_type)
        
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        results = []
        
        for row in cursor.fetchall():
            results.append({
                "source_name": row[0],
                "data_type": row[1],
                "content": row[2][:500] + "..." if len(row[2]) > 500 else row[2],
                "metadata": json.loads(row[3]),
                "timestamp": row[4],
                "url": row[5]
            })
        
        conn.close()
        return results
    
    def export_data(self, output_dir: str = "exported_data"):
        """Export all extracted data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Export by source
        cursor.execute("SELECT DISTINCT source_name FROM extracted_data")
        sources = [row[0] for row in cursor.fetchall()]
        
        for source in sources:
            source_dir = output_path / source
            source_dir.mkdir(exist_ok=True)
            
            cursor.execute("""
                SELECT data_type, content, metadata, timestamp, url
                FROM extracted_data 
                WHERE source_name = ?
                ORDER BY timestamp DESC
            """, (source,))
            
            for i, row in enumerate(cursor.fetchall()):
                data_type, content, metadata, timestamp, url = row
                
                filename = f"{data_type}_{i:04d}_{timestamp.replace(':', '-').replace(' ', '_')}.txt"
                file_path = source_dir / filename
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {data_type.upper()}\n")
                    f.write(f"# Source: {source}\n")
                    f.write(f"# URL: {url}\n")
                    f.write(f"# Timestamp: {timestamp}\n")
                    f.write(f"# Metadata: {metadata}\n\n")
                    f.write(content)
        
        conn.close()
        print(f"Data exported to {output_path}")


def start_scheduled_extraction():
    """Start the scheduled extraction service"""
    api = DataExtractionAPI()
    
    # Schedule regular extractions
    schedule.every(1).hours.do(api.run_extraction_cycle)
    
    print("Data extraction scheduler started...")
    print("Manual extraction commands:")
    print("- api.run_extraction_cycle() - Run full cycle")
    print("- api.get_extraction_stats() - Get statistics")
    print("- api.search_extracted_data('query') - Search data")
    print("- api.export_data() - Export all data")
    
    # Run initial extraction
    api.run_extraction_cycle()
    
    # Keep the scheduler running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "schedule":
        start_scheduled_extraction()
    else:
        # Interactive mode
        api = DataExtractionAPI()
        print("Data Extraction API loaded.")
        print("Available methods:")
        print("- api.run_extraction_cycle()")
        print("- api.get_extraction_stats()")
        print("- api.search_extracted_data('query')")
        print("- api.export_data()")
        
        # Run one extraction cycle
        api.run_extraction_cycle()
        
        # Show stats
        stats = api.get_extraction_stats()
        print("\nExtraction Statistics:")
        print(f"Total items: {stats['total_items']}")
        for source, data in stats['source_stats'].items():
            print(f"  {source}: {data['count']} items")
