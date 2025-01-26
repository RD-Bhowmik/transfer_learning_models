import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from collections import Counter
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to the dataset
positive_folder = "D:\\categorization\\new\\hpv\\positive"
negative_folder = "D:\\categorization\\new\\hpv\\negative"
xlsx_file_path = "D:\\categorization\\IARCImageBankColpo\\Cases Meta data.xlsx"

# Parameters
image_size = (224, 224)  # Resize dimensions