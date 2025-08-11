"""
File utility functions for AI Fitness Coach.

This module provides common file operations, directory management,
and file handling utilities used throughout the application.
"""

import os
import shutil
import json
import csv
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from datetime import datetime
import zipfile
import tempfile

from ai_fitness.config.settings import get_settings

def ensure_directory(directory_path: Union[str, Path]) -> bool:
    """Ensure a directory exists, create if it doesn't"""
    try:
        directory_path = Path(directory_path)
        directory_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {str(e)}")
        return False

def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes"""
    try:
        file_path = Path(file_path)
        if file_path.exists():
            return file_path.stat().st_size
        return 0
    except Exception as e:
        print(f"Error getting file size for {file_path}: {str(e)}")
        return 0

def get_file_extension(file_path: Union[str, Path]) -> str:
    """Get file extension from file path"""
    try:
        file_path = Path(file_path)
        return file_path.suffix.lower()
    except Exception:
        return ""

def is_valid_file(file_path: Union[str, Path], allowed_extensions: List[str] = None) -> bool:
    """Check if file is valid based on extension and existence"""
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False
        
        if allowed_extensions:
            file_ext = get_file_extension(file_path)
            return file_ext in allowed_extensions
        
        return True
    except Exception:
        return False

def safe_filename(filename: str) -> str:
    """Convert filename to safe version by removing/replacing invalid characters"""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_file"
    
    return filename

def create_backup_file(file_path: Union[str, Path], backup_dir: Union[str, Path] = None) -> Optional[Path]:
    """Create a backup of a file"""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        
        if backup_dir is None:
            backup_dir = Path(file_path).parent / "backups"
        
        ensure_directory(backup_dir)
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_filename
        
        # Copy file
        shutil.copy2(file_path, backup_path)
        print(f"Backup created: {backup_path}")
        return backup_path
        
    except Exception as e:
        print(f"Error creating backup: {str(e)}")
        return None

def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> bool:
    """Save data to JSON file"""
    try:
        file_path = Path(file_path)
        ensure_directory(file_path.parent)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        
        print(f"Data saved to JSON: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error saving JSON: {str(e)}")
        return False

def load_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load data from JSON file"""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
        
    except Exception as e:
        print(f"Error loading JSON: {str(e)}")
        return None

def save_csv(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> bool:
    """Save data to CSV file"""
    try:
        file_path = Path(file_path)
        ensure_directory(file_path.parent)
        
        if not data:
            print("No data to save")
            return False
        
        # Get fieldnames from first row
        fieldnames = list(data[0].keys())
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"Data saved to CSV: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error saving CSV: {str(e)}")
        return False

def load_csv(file_path: Union[str, Path]) -> Optional[List[Dict[str, Any]]]:
    """Load data from CSV file"""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        
        data = []
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        
        return data
        
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return None

def find_files(directory: Union[str, Path], pattern: str = "*", recursive: bool = True) -> List[Path]:
    """Find files matching pattern in directory"""
    try:
        directory = Path(directory)
        if not directory.exists():
            return []
        
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
        
        # Return only files, not directories
        return [f for f in files if f.is_file()]
        
    except Exception as e:
        print(f"Error finding files: {str(e)}")
        return []

def get_directory_size(directory: Union[str, Path]) -> int:
    """Get total size of directory in bytes"""
    try:
        directory = Path(directory)
        if not directory.exists():
            return 0
        
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
        
    except Exception as e:
        print(f"Error getting directory size: {str(e)}")
        return 0

def clean_old_files(directory: Union[str, Path], days_old: int = 30) -> int:
    """Clean old files from directory"""
    try:
        directory = Path(directory)
        if not directory.exists():
            return 0
        
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        deleted_count = 0
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                file_age = file_path.stat().st_mtime
                if file_age < cutoff_date:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        print(f"Deleted old file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting file {file_path}: {str(e)}")
        
        print(f"Cleaned {deleted_count} old files")
        return deleted_count
        
    except Exception as e:
        print(f"Error cleaning old files: {str(e)}")
        return 0

def create_zip_archive(source_dir: Union[str, Path], output_path: Union[str, Path]) -> bool:
    """Create ZIP archive from directory"""
    try:
        source_dir = Path(source_dir)
        output_path = Path(output_path)
        
        if not source_dir.exists():
            print(f"Source directory does not exist: {source_dir}")
            return False
        
        ensure_directory(output_path.parent)
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir)
                    zipf.write(file_path, arcname)
        
        print(f"ZIP archive created: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating ZIP archive: {str(e)}")
        return False

def extract_zip_archive(zip_path: Union[str, Path], extract_dir: Union[str, Path]) -> bool:
    """Extract ZIP archive to directory"""
    try:
        zip_path = Path(zip_path)
        extract_dir = Path(extract_dir)
        
        if not zip_path.exists():
            print(f"ZIP file does not exist: {zip_path}")
            return False
        
        ensure_directory(extract_dir)
        
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_dir)
        
        print(f"ZIP archive extracted to: {extract_dir}")
        return True
        
    except Exception as e:
        print(f"Error extracting ZIP archive: {str(e)}")
        return False

def get_temp_directory() -> Path:
    """Get temporary directory for the application"""
    settings = get_settings()
    temp_dir = settings.data_dir / "temp"
    ensure_directory(temp_dir)
    return temp_dir

def cleanup_temp_files() -> int:
    """Clean up temporary files"""
    try:
        temp_dir = get_temp_directory()
        deleted_count = 0
        
        for file_path in temp_dir.glob('*'):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception:
                    pass
        
        return deleted_count
        
    except Exception:
        return 0

if __name__ == "__main__":
    # Example usage
    settings = get_settings()
    
    # Test directory creation
    test_dir = settings.data_dir / "test_utils"
    ensure_directory(test_dir)
    
    # Test file operations
    test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
    test_file = test_dir / "test.json"
    save_json(test_data, test_file)
    
    # Test file finding
    files = find_files(test_dir, "*.json")
    print(f"Found {len(files)} JSON files")
    
    # Test cleanup
    cleanup_temp_files()
