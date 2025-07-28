import os
import json
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms as transforms
import logging
from pathlib import Path
from typing import Optional

from imageCache import ImageCache  

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_imgur_url(url: str) -> str:
    """
    Convert Imgur page URLs to direct image links on i.imgur.com ending in .jpg
    """
    lower = url.lower()
    if "imgur.com" in lower and not lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
        return url.replace("imgur.com", "i.imgur.com") + ".jpg"
    return url

class ViLIDDataset(Dataset):
    """
    Dataset for Vision-Language Inconsistency Detection.
    Returns tokenized text, processed image tensor, label, and rationale encodings, 
    plus a flag 'image_present'.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        image_processor,
        max_text_length: int = 512,
        max_rationale_length: int = 512,
        text_col: str = 'text',
        image_col: str = 'image_url',
        label_col: str = 'label',
        text_rationale_col: str = 'text_rationale',
        image_rationale_col: str = 'image_rationale',
        id_col: str = 'id',
        image_store_way: str = 'URL',    # 'URL' or 'LOCAL'
        image_dir: str = None,
        image_transform=None,
        use_image_cache: bool = True,
        image_cache_dir: str = ".image_cache",
        image_cache_size_gb: float = 2.0
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.processor = image_processor      # Expect a CLIPProcessor
        self.max_text_length = max_text_length
        self.max_rationale_length = max_rationale_length

        # Column names
        self.text_col = text_col
        self.image_col = image_col
        self.label_col = label_col
        self.text_rationale_col = text_rationale_col
        self.image_rationale_col = image_rationale_col
        self.id_col = id_col

        self.image_store_way = image_store_way
        self.image_dir = image_dir

        # Fallback image transform
        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.image_transform = image_transform

        # Image cache
        self.use_image_cache = use_image_cache
        if use_image_cache:
            self.image_cache = ImageCache(
                cache_dir=image_cache_dir,
                max_size_gb=image_cache_size_gb
            )
            logger.info(f"Image caching enabled at {image_cache_dir}")
        else:
            self.image_cache = None

        if self.image_store_way == 'LOCAL' and not self.image_dir:
            logger.warning("LOCAL image_store_way but no image_dir provided.")

        logger.info(f"ViLIDDataset initialized: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def load_image(self, source: str) -> Optional[Image.Image]:
        """
        Fetch PIL image from URL or local path, return None on failure.
        """
        if source.startswith(('http://', 'https://')):
            source = fix_imgur_url(source)
            headers = {"User-Agent": "Mozilla/5.0 (compatible)"}
            for attempt in range(3):
                try:
                    resp = requests.get(source, timeout=10, headers=headers)
                    resp.raise_for_status()
                    content_type = resp.headers.get("Content-Type", "")
                    if not content_type.startswith("image/"):
                        raise ValueError(f"URL did not return an image (Content-Type: {content_type})")
                    return Image.open(BytesIO(resp.content)).convert("RGB")
                except Exception as e:
                    status = getattr(e, 'response', None)
                    logger.error(f"Error loading image [{source}], attempt {attempt}: {e}")
                    if hasattr(e, 'response') and getattr(e.response, 'status_code', None) == 429 and attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    break
            return None
        else:
            try:
                return Image.open(source).convert('RGB')
            except Exception as e:
                logger.error(f"Error loading local image [{source}]: {e}")
                return None

    def process_image(self, img: Image.Image, key: str) -> torch.Tensor:
        """Cacheable image processing via CLIPProcessor or torchvision transforms."""
        if self.use_image_cache:
            cache_key = self.image_cache.url_to_key(key)
            cached = self.image_cache.get(cache_key)
            if cached is not None:
                return cached

        if self.processor is not None:
            inputs = self.processor(images=img, return_tensors="pt")
            tensor = inputs['pixel_values'].squeeze(0)
        else:
            tensor = self.image_transform(img)

        if self.use_image_cache:
            self.image_cache.put(cache_key, tensor)

        return tensor

    def tokenize(self, text: str, length: int) -> dict:
        """Tokenize text or rationale to fixed-length dict of tensors."""
        enc = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=length,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        # Text fields
        text = str(row.get(self.text_col, ''))
        txt_rat = str(row.get(self.text_rationale_col, ''))
        img_rat = str(row.get(self.image_rationale_col, ''))

        # Image path or URL
        if self.image_store_way == 'LOCAL':
            filename = str(row.get(self.image_col, ''))
            source = filename if os.path.isabs(filename) else os.path.join(self.image_dir, filename)
        else:
            source = str(row.get(self.image_col, ''))

        # Load image and set flag
        pil = self.load_image(source)
        image_present = pil is not None
        if not image_present:
            pil = Image.new('RGB', (224, 224), (0, 0, 0))

        img_tensor = self.process_image(pil, source)

        # Label
        label = int(row.get(self.label_col, 0))

        # Tokenize all text
        text_enc = self.tokenize(text, self.max_text_length)
        text_rat_enc = self.tokenize(txt_rat, self.max_rationale_length)
        img_rat_enc = self.tokenize(img_rat, self.max_rationale_length)

        return {
            'text_encoding': text_enc,
            'text_rationale_encoding': text_rat_enc,
            'image_rationale_encoding': img_rat_enc,
            'images': img_tensor,
            'image_present': torch.tensor(image_present, dtype=torch.bool),
            'labels': torch.tensor(label, dtype=torch.long),
            'metadata': {
                'id': row.get(self.id_col, str(idx)),
                'text_raw': text,
                'image_source': source,
                'text_rationale_raw': txt_rat,
                'image_rationale_raw': img_rat
            }
        }

def vilid_collate_fn(batch):
    """Batch together a list of samples into tensors."""
    collated = {
        'text_encoding': {},
        'text_rationale_encoding': {},
        'image_rationale_encoding': {},
        'images': [],
        'image_present': [],
        'labels': [],
        'metadata': [item['metadata'] for item in batch]
    }

    # keys for token dicts
    for key in batch[0]['text_encoding']:
        collated['text_encoding'][key] = torch.stack([b['text_encoding'][key] for b in batch])
        collated['text_rationale_encoding'][key] = torch.stack([b['text_rationale_encoding'][key] for b in batch])
        collated['image_rationale_encoding'][key] = torch.stack([b['image_rationale_encoding'][key] for b in batch])

    collated['images'] = torch.stack([b['images'] for b in batch])
    collated['image_present'] = torch.stack([b['image_present'] for b in batch])
    collated['labels'] = torch.stack([b['labels'] for b in batch])

    return collated


def read_data_file(path: str) -> pd.DataFrame:
    """Load TSV, CSV, JSON, or JSONL into a DataFrame."""
    if path.endswith('.tsv'):
        df = pd.read_csv(path, sep='\t')
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith(('.jsonl', '.jsonl.gz')):
        df = pd.read_json(path, lines=True)
    elif path.endswith('.json'):
        try:
            with open(path, 'r') as f:
                obj = json.load(f)
            records = obj if isinstance(obj, list) else [obj]
            df = pd.DataFrame(records)
        except json.JSONDecodeError:
            df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported extension: {path}")
    return df.fillna('')

def get_dataloader(
    data_path: str,
    tokenizer,
    image_processor,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    max_text_length: int = 512,
    max_rationale_length: int = 512,
    text_col: str = 'text',
    image_col: str = 'image_url',
    label_col: str = 'label',
    text_rationale_col: str = 'text_rationale',
    image_rationale_col: str = 'image_rationale',
    id_col: str = 'id',
    image_store_way: str = 'URL',
    image_dir: str = None,
    filter_empty: bool = True,
    use_image_cache: bool = True,
    image_cache_dir: str = ".image_cache",
    image_cache_size_gb: float = 2.0
):
    df = read_data_file(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")

    if filter_empty:
        before = len(df)
        df = df[df[text_col].astype(bool)]
        if image_store_way == 'URL':
            df = df[df[image_col].astype(bool)]
        logger.info(f"Filtered empty → {len(df)} rows (removed {before - len(df)})")

    ds = ViLIDDataset(
        df, tokenizer, image_processor,
        max_text_length=max_text_length,
        max_rationale_length=max_rationale_length,
        text_col=text_col, image_col=image_col,
        label_col=label_col,
        text_rationale_col=text_rationale_col,
        image_rationale_col=image_rationale_col,
        id_col=id_col,
        image_store_way=image_store_way,
        image_dir=image_dir,
        use_image_cache=use_image_cache,
        image_cache_dir=image_cache_dir,
        image_cache_size_gb=image_cache_size_gb
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=vilid_collate_fn
    )
    return loader, ds


