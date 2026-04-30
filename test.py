



##### input_size подгружать сюда ############




# app.py
import streamlit as st
import pandas as pd
import joblib
import torch
import torch.nn as nn
import numpy as np

preprocessor = joblib.load("preprocessor.joblib")

cat_encoder = preprocessor.named_transformers_['cat']

AVAILABLE_BRANDS = sorted(cat_encoder.categories_[0])

AVAILABLE_FUELTYPES = sorted(cat_encoder.categories_[1])

AVAILABLE_CARBODY = sorted(cat_encoder.categories_[2])



print(AVAILABLE_BRANDS)
print('-----------')
print(AVAILABLE_FUELTYPES)
print('-----------')
print(AVAILABLE_CARBODY)
print('-----------')