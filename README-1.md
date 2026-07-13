# 💻 Laptop Price Prediction

Predicting laptop prices from hardware specifications using Linear Regression, Decision Tree, and Random
Forest — with a clean walkthrough of data cleaning, feature engineering, and two experiments that show why a
log transform matters.

**Dataset:** 1,303 laptops · 12 raw features → 17 engineered features
**Models:** Linear Regression · Decision Tree · Random Forest
**Best result:** Random Forest, R² = 0.716, MAE ≈ ₹12,082 (raw price scale)

---

## 📌 Why this project

Laptop pricing looks arbitrary from the outside — a spec sheet full of screen resolutions, CPU codenames, and
storage combos that don't obviously map to a number. This project turns that mess into a numeric prediction
problem: given a laptop's specs, how close can a model get to its real price?

Beyond the model results, this repo is written to be a **teaching notebook** — every cell has a short markdown
explanation of what it does and why, so it's readable by anyone learning the standard ML tabular-data workflow
(clean → engineer → encode → model → evaluate → improve).

## 🧹 What the pipeline does

1. **Load** the [Kaggle laptop specs dataset](https://www.kaggle.com/datasets/muhammadmusharraf444/laptop-specifications-and-price-prediction-dataset) (no missing values, no duplicates).
2. **Explore** the data — the key finding: `Price` is right-skewed (skewness = 1.521), which matters a lot for
   Linear Regression's assumptions.
3. **Clean** text-encoded columns: `Ram` (`'8GB'` → `8`), `Weight` (`'1.5kg'` → `1.5`).
4. **Engineer features** out of messy text columns:
   - `ScreenResolution` → `Touchscreen`, `IPS`, `width_res`, `height_res`, and a derived `ppi` (pixels per inch)
   - `Cpu` → simplified `cpu_brand` (Intel i7 / i5 / i3 / AMD / other)
   - `Memory` → `SSD`, `HDD`, `flash_storage`, `Hybrid` (handles laptops with two drives)
   - `Gpu` → `Gpu_brand`
5. **Label-encode** remaining categorical columns (`Company`, `TypeName`, `OpSys`, `cpu_brand`, `Gpu_brand`).
6. **Experiment 1** — train all three models on raw price. Random Forest wins (R² = 0.716).
7. **Feature importance** — RAM alone explains ~69% of the Random Forest's splits; `Hybrid`, `flash_storage`,
   and `HDD` are near-zero and get dropped.
8. **Experiment 2** — log-transform price (skewness drops from 1.521 → -0.174) and retrain on the pruned
   feature set, with a clean comparison of why R² isn't directly comparable across the raw-price and
   log-price scales.
9. **Experiment 3** — swap label encoding for **one-hot encoding** (`pd.get_dummies`), keeping the same log
   price and pruned feature set as Experiment 2, isolating encoding method as the only variable. Removes the
   arbitrary numeric ordering label encoding imposes on categorical columns — a known limitation for Linear
   Regression specifically.

## 📊 Results

| Experiment | Encoding | Target | Model | R² | MAE |
|---|---|---|---|---|---|
| 1 | Label | Raw price | Linear Regression | 0.7026 | ₹15,129.71 |
| 1 | Label | Raw price | Decision Tree | 0.6878 | ₹13,039.63 |
| 1 | Label | Raw price | **Random Forest** | **0.7159** | **₹12,082.17** |
| 2 | Label | Log price, pruned features | Linear Regression / Decision Tree / Random Forest | see notebook | log-scale |
| 3 | One-hot | Log price, pruned features | Linear Regression / Decision Tree / Random Forest | see notebook | log-scale |

*(Experiment 1's raw-price R² isn't directly comparable to Experiments 2/3's log-price R² — see the notebook's
final section for why. Experiment 2 vs. 3, however, is apples-to-apples: same target, same features, only the
encoding method changes — the cleanest read on what one-hot encoding buys you, and it improves Linear
Regression's R² the most, as expected.)*

## 🗂️ Repo contents

- `Laptop_Price_Prediction_Annotated.ipynb` — the full, cell-by-cell annotated notebook
- `README.md` — this file

## 🚀 Running it yourself

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Laptop_Price_Prediction_Annotated.ipynb
```

The notebook downloads the dataset automatically via `kagglehub` on first run.

## 🔍 Known limitations (documented honestly)

- One-hot encoding widens the feature space considerably (a column per category, minus one baseline per
  original column) — worth watching for overfitting risk on a dataset this size, especially for the Decision
  Tree.
- Decision Tree and Random Forest hyperparameters (`max_depth=10`, `min_samples_leaf=10`) are fixed, not tuned
  via cross-validation.
- Results come from a single train/test split (`random_state=42`), not k-fold cross-validation.

## 🧠 What I'd improve next

- Hyperparameter search (grid/random search + cross-validation) for the tree models
- Try gradient boosting (XGBoost/LightGBM) as a stronger baseline
- Residual analysis to see where the model is most wrong (very high-end laptops, likely)
- Try target/mean encoding as a middle ground between label and one-hot encoding for high-cardinality columns
  like `Company`

---

*Built as a hands-on walkthrough of the standard tabular ML workflow: clean → engineer → encode → model →
evaluate → improve.*
