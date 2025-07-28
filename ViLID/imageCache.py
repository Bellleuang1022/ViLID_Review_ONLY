import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms as transforms
import logging

import hashlib
import pickle
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageCache:
    """Cache for storing downloaded or processed images."""
    
    def __init__(self, cache_dir=".image_cache", max_size_gb=2):
        """
        Initialize image cache.
        
        Args:
            cache_dir: Directory to store cached images
            max_size_gb: Maximum cache size in GB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.current_size_bytes = 0
        self.index_file = self.cache_dir / "cache_index.pkl"
        
        # Load cache index if exists
        self.cache_index = {}
        if self.index_file.exists():
            try:
                with open(self.index_file, "rb") as f:
                    self.cache_index = pickle.load(f)
                    
                # Calculate current cache size
                self.calculate_current_size()
                    
                logger.info(f"Loaded image cache with {len(self.cache_index)} entries "
                           f"({self.current_size_bytes / (1024*1024):.2f} MB)")
            except Exception as e:
                logger.warning(f"Failed to load cache index: {str(e)}. Creating new cache.")
                self.cache_index = {}
                self.current_size_bytes = 0
    
    def calculate_current_size(self):
        """Calculate the current size of the cache."""
        self.current_size_bytes = 0
        for cache_path in self.cache_dir.glob("*.pt"):
            if cache_path.name != "cache_index.pkl":
                self.current_size_bytes += cache_path.stat().st_size
    
    def url_to_key(self, url):
        """Convert URL or file path to a unique cache key."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def get(self, key):
        """Get image from cache if it exists."""
        if key in self.cache_index:
            cache_path = self.cache_dir / f"{key}.pt"
            if cache_path.exists():
                try:
                    return torch.load(cache_path)
                except Exception as e:
                    logger.warning(f"Failed to load cached image {key}: {str(e)}")
                    # Remove from index if loading fails
                    self.remove(key)
        return None
    
    def put(self, key, tensor):
        """Store image tensor in cache."""
        try:
            cache_path = self.cache_dir / f"{key}.pt"
            torch.save(tensor, cache_path)
            
            file_size = cache_path.stat().st_size
            self.cache_index[key] = {"path": str(cache_path), "size": file_size}
            self.current_size_bytes += file_size
            
            # Enforce size limit
            self._enforce_size_limit()
            
            # Update index file
            self._save_index()
            
            return True
        except Exception as e:
            logger.warning(f"Failed to cache image {key}: {str(e)}")
            return False
    
    def remove(self, key):
        """Remove image from cache."""
        if key in self.cache_index:
            cache_path = Path(self.cache_index[key]["path"])
            if cache_path.exists():
                try:
                    file_size = cache_path.stat().st_size
                    cache_path.unlink()
                    self.current_size_bytes -= file_size
                except Exception as e:
                    logger.warning(f"Failed to remove cached file {key}: {str(e)}")
            
            del self.cache_index[key]
            self._save_index()
    
    def _enforce_size_limit(self):
        """Remove oldest entries when cache exceeds size limit."""
        if self.current_size_bytes <= self.max_size_bytes:
            return
            
        # Sort by access time (would require tracking, for now just use keys)
        keys_to_remove = sorted(self.cache_index.keys())
        
        # Remove oldest entries until under the limit
        for key in keys_to_remove:
            if self.current_size_bytes <= self.max_size_bytes:
                break
                
            file_size = self.cache_index[key]["size"]
            self.remove(key)
            self.current_size_bytes -= file_size
    
    def _save_index(self):
        """Save cache index to disk."""
        try:
            with open(self.index_file, "wb") as f:
                pickle.dump(self.cache_index, f)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {str(e)}")
    
    def clear(self):
        """Clear the entire cache."""
        for key in list(self.cache_index.keys()):
            self.remove(key)
        
        # Also remove any stray files
        for cache_path in self.cache_dir.glob("*.pt"):
            try:
                cache_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove file {cache_path}: {str(e)}")
        
        self.current_size_bytes = 0
        self.cache_index = {}
        self._save_index()