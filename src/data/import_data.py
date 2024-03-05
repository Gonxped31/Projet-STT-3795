import pandas as pd
import numpy as np
import datasets
from datasets import load_dataset

data = load_dataset("common_language")

command = 'curl -X GET \
     "https://datasets-server.huggingface.co/rows?dataset=common_language&config=full&split=train&offset=0&length=100"'

