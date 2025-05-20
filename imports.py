# 의존성 라이브러리

import torch
import torchvision
from torchvision.transforms import v2, functional as F
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torch.utils.data import Dataset, DataLoader

import cv2
from PIL import Image

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

# XML 파싱 관련
import xml.etree.ElementTree as ET

# 시각화 관련
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import matplotlib.font_manager as fm
import matplotlib as mpl

# 기타 유틸리티
import os
import time
from tqdm import tqdm
import platform

# 로깅 관련
import logging
import sys
from datetime import datetime

