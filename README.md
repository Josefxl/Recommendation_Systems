# Recommendation Systems — Movies (Collaborative & Content-Based)

## 📌 Overview
This repository demonstrates two classic approaches to movie recommendation:
- **Collaborative Filtering (CF):** learns user/item latent vectors from the ratings matrix to predict unseen ratings.
- **Content-Based Filtering (CBF) with a Neural Network:** learns user/movie embeddings from attributes (genres, year, engineered stats) and scores via a dot product.

Both are implemented in separate notebooks with clear, reproducible workflows.

---

## 🛠️ Tools & Libraries
- Python, Jupyter Notebook
- NumPy, Pandas
- TensorFlow / Keras
- scikit-learn (StandardScaler, MinMaxScaler, train_test_split)
- tabulate (for readable tables)

---

## 📂 Dataset
- **MovieLens ml-latest-small (reduced)** — ratings from **0.5 to 5** in 0.5 steps; focused on post-2000 films and popular genres for faster iteration.
- Citation: Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context.* **ACM TiiS, 5(4)**, 19:1–19:19. https://doi.org/10.1145/2827872

**Typical CF shapes (example):**  
`Y (ratings) = (4778, 443)`, `R (indicator) = (4778, 443)`, `X = (4778, 10)`, `W = (443, 10)`, `b = (1, 443)`.

---

## 🧠 Approach 1 — Collaborative Filtering (Matrix Factorization)
Minimize:
\[
J=\tfrac{1}{2}\|(XW^\top+b-Y)\odot R\|_F^2+\tfrac{\lambda}{2}(\|X\|_F^2+\|W\|_F^2)
\]
- Vectorized TF implementation for custom training loops.
- Supports adding a **new user**’s ratings, normalization, and **Top-N** recommendations.

---

## 🧠 Approach 2 — Content-Based Filtering (Neural Network)
- Two Keras subnets (UserNN & ItemNN): `Dense(256,relu) → Dense(128,relu) → Dense(32,linear)`, then **L2-normalize** and **Dot** for similarity.
- Data scaled with `StandardScaler`; targets scaled to `[-1,1]` with `MinMaxScaler`.
- Examples: Train loss ≈ **0.0704**, Test loss ≈ **0.087** after 30 epochs.

---

## 🚀 How to Run (ONE single block: clone + install + imports + open notebooks)
```bash
# Clone the repository
git clone https://github.com/Josefxl/Recommendation_Systems.git
cd Recommendation_Systems

# (Optional) Create and activate a virtual environment
# python -m venv .venv && source .venv/bin/activate    # macOS/Linux
# py -m venv .venv && .venv\Scripts\activate           # Windows

# Install dependencies
python -m pip install --upgrade pip
pip install numpy pandas tensorflow scikit-learn tabulate jupyter

# --- Reference Imports (paste into notebooks as first cell) ---

# Collaborative Filtering notebook imports
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from recsys_utils import *

# Content-Based NN notebook imports
# import numpy as np
# import numpy.ma as ma
# import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.layers import Layer, Input, Dense, Dot
# from tensorflow.keras.models import Model, Sequential
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import train_test_split
# import tabulate
# from recsysNN_utils import *
# pd.set_option("display.precision", 1)
# import warnings; warnings.filterwarnings("ignore")

# Launch Jupyter (then open the notebooks inside /notebooks)
jupyter notebook
