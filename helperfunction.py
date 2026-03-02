import pymongo
import pymysql
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote

from dotenv import dotenv_values
from time import time
import logging
import aiohttp

import redis
import cv2
import requests
import numpy as np
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Tuple, Any
from PIL import Image
from io import BytesIO


from dotenv import load_dotenv
from pathlib import Path
import os
async def read_image_from_url(url):
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                content = await response.read()
                image_array = np.asarray(bytearray(content), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                return image, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logging.error(f"Error reading image from URL: {url}, {repr(e)}")
        return None
def read_image_from_bytes(image_bytes: bytes):
    try:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            logging.error("cv2.imdecode returned None from bytes")
            return None, None

        # return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image , cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    except Exception as e:
        logging.error(f"Error decoding image from bytes: {repr(e)}")
        return None, None