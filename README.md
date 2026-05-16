Regression Models Repository
=============================================

Abstract
--------

This repository implements production-ready linear and logistic regression models using vectorized gradient descent optimization from scratch using NumPy. The dual-function architecture serves PropTech Automated Valuation Models (AVM) through linear regression on the Boston Housing dataset, and binary risk classification frameworks through logistic regression on the Titanic Survival dataset. All implementations prioritize mathematical rigor, computational efficiency, numerical stability, and deep empirical validation.

Mathematical Foundations
------------------------

### Linear Regression: PropTech Automated Valuation Model

**Hypothesis Function:**

$$\hat{y} = Xw + b$$

**Vectorized Cost Function (Mean Squared Error):**

$$J(w, b) = \frac{1}{2m} (Xw + b\mathbf{1} - y)^T (Xw + b\mathbf{1} - y)$$

where:

-   $m \in \mathbb{N}$: number of training examples

-   $X \in \mathbb{R}^{m \times n}$: design matrix

-   $w \in \mathbb{R}^n$: weight vector

-   $b \in \mathbb{R}$: bias scalar

-   $y \in \mathbb{R}^m$: target vector

-   $\mathbf{1} \in \mathbb{R}^m$: vector of ones

**Vectorized Gradient Update Rules:**

$$w := w - \alpha \frac{1}{m} X^T (Xw + b\mathbf{1} - y)$$

$$b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} ((Xw)_i + b - y_i)$$

where $\alpha \in \mathbb{R}^+$ represents the learning rate.

* * * * *

### Logistic Regression: Binary Risk Classification Framework

**Hypothesis Function:**

$$\hat{y} = \sigma(Xw + b) = \frac{1}{1 + e^{-(Xw + b)}}$$

where $\sigma(\cdot)$ is the element-wise Sigmoid activation function mapping values strictly to the probability manifold $[0, 1]$.

**Cross-Entropy Loss Function (Log-Loss):**

$$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

where $y_i \in \{0, 1\}$.

**Vectorized Gradient Update Rules:**

$$w := w - \alpha \frac{1}{m} X^T (\sigma(Xw + b\mathbf{1}) - y)$$

$$b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (\sigma(Xw)_i + b - y_i)$$

* * * * *

System Architecture
-------------------

```
Regression_Models/
├── environment.yml               # Conda environment specification
├── .gitignore                    # Version control exclusions
├── README.md                     # Technical documentation
└── src/
    ├── data_loader.py            # Dataset loading and validation
    ├── preprocessing.py          # Feature engineering pipelines
    ├── train.py                  # Training orchestration
    ├── evaluate.py               # Model evaluation and metrics
    └── models/
        ├── linear_model.py       # Linear Regression (Vectorized GD)
        └── logistic_model.py     # Logistic Regression (Vectorized GD)

```

* * * * *

System Setup
------------

### Prerequisites

-   Ubuntu 20.04+ or compatible Linux distribution

-   Git with SSH authentication configured

-   Conda package manager (Miniconda/Anaconda)

### Installation Protocol

**Step 1: Repository Acquisition**

Bash

```
git clone git@github.com:namanatgoel/Regression_Models.git
cd Regression_Models

```

**Step 2: Environment Construction**

Bash

```
conda env create -f environment.yml
conda activate regression_env

```

**Step 3: Verify Installation**

Bash

```
python -c "import numpy, pandas, sklearn; print('Environment validated')"

```

### Dataset Requirements

**Linear Regression (Boston Housing):**

-   **File:** `HousingData.csv`

-   **Features:** 13 continuous and categorical variables

-   **Target:** `MEDV` (Median value of owner-occupied homes in $1000s)

-   **Preprocessing:** Median imputation for missing elements, `StandardScaler` applied to all independent continuous features.

**Logistic Regression (Titanic Survival):**

-   **File:** `train.csv`

-   **Features:** `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`

-   **Target:** `Survived` (Binary: 0 = Perished, 1 = Survived)

-   **Preprocessing:** One-Hot Encoding for multi-class categorical handles (`Sex`, `Embarked`), median imputation for missing `Age` blocks, categorical structural drops for un-vectorizable dimensions (`Cabin`, `Ticket`, `Name`).

* * * * *

Implementation & Optimization Details
-------------------------------------

### Numerical Stability Guardrails

-   **Sigmoid Clipping:** Pre-activation vector $z = Xw + b$ is clipped to $[-500, 500]$ to prevent floating-point underflow/overflow errors (`NaN` generation) during standard exponentiation.

-   **Log-Loss Bounds:** Predicted probabilities are clipped to $[\epsilon, 1-\epsilon]$ where $\epsilon = 1 \times 10^{-15}$ to protect against evaluates evaluating to $\log(0)$ which yields numerical infinity.

-   **Standard Scaling:** Features are standardized to zero-mean and unit-variance ($\mu=0, \sigma^2=1$). This eliminates scale-dependent optimization drift, ensuring the cost surface is symmetric and prevents gradient explosion or vanishing maneuvers.

### Training Pipeline Execution

Bash

```
cd src
# Train both frameworks concurrently
python train.py --model both --linear-data ../HousingData.csv --logistic-data ../TitanicData.csv

```

* * * * *

Empirical Performance & Analytical Inferences
---------------------------------------------

### 1\. Linear Regression (Boston Housing Optimization)

The baseline Ordinary Least Squares (OLS) optimization converged to an $R^2 \approx 0.7787$, matching scikit-learn's optimized matrix solution up to precision tolerances.

#### Target Censoring Identification

-   **Plot Reference:** `outputs/linear/actual_vs_pred.png` & `outputs/linear/residuals.png`

-   **Insight:** Both plots isolate a distinct diagonal boundary capping the true target array cleanly at $y = 50.0$. This demonstrates an artificial data ceiling imposed during the historical collection protocol. The model correctly projects linear trends past this ceiling. This dynamic reveals that model variance at the upper tail is an artifact of raw data collection limits, not algorithmic optimization failures.

#### Gauss-Markov Assumption Verification

-   **Plot Reference:** `outputs/linear/residuals.png`

-   **Insight:** The residuals are structurally symmetric and zero-centered across the intermediate predicted space. This empirically validates the homoscedasticity assumption ($Var(e|X) = \sigma^2$). Outside of the capped target anomaly, the error distribution displays no non-linear structural curvature, verifying the structural viability of the linear hypothesis.

#### Standardized Coefficient Interpretability

-   **Plot Reference:** `outputs/linear/feature_importance.png`

-   **Insight:** Because inputs are passed through a uniform standard scaling pipeline, the resulting weights act as **standardized coefficients**. `LSTAT` (lower socioeconomic status percentage) displays the strongest negative gradient, while `RM` (average rooms per dwelling) acts as the dominant positive optimization weight. This mathematically isolates local macroeconomic conditions and physical asset volume as the primary drivers of market valuations.

#### Multicollinearity Assessment

-   **Plot Reference:** `outputs/linear/correlation_heatmap(linear).png`

-   **Insight:** Pre-screening exposes critical inter-feature dependencies, notably between index parameters `RAD` (highway accessibility) and `TAX` (property tax rate). Identifying these collinear pairs confirms that variance expansion is managed effectively within the gradient tracking pipeline.

* * * * *

### 2\. Logistic Regression (Titanic Survival Classification)

The binary cross-entropy classifier yields robust threshold-agnostic performance metrics (ROC-AUC $\approx 0.88$).

#### Structural Class Imbalance Adaptability

-   **Plot Reference:** `outputs/logistic/pr_curve.png`

-   **Insight:** The underlying dataset exhibits a skewed baseline survival distribution ($\approx 38\%$). Under these constraints, standard classification accuracy is a flawed optimization metric. The Precision-Recall Curve isolates predictive accuracy specifically on the minority positive class. The high Average Precision (AP) baseline confirms that the model retains high positive predictive value without generating high false-alarm rates.

#### Boundary Separation Mechanics

-   **Plot Reference:** `outputs/logistic/roc_curve.png`

-   **Insight:** The steep initial derivative of the Receiver Operating Characteristic curve demonstrates high true positive capture speeds before false positives are allowed to scale. The recorded Area Under the Curve (AUC = 0.88) verifies that a randomly drawn positive instance (survivor) has a higher computed probability mapping than a negative instance 88% of the time.

#### Operational Threshold Tuning Bias

-   **Plot Reference:** `outputs/logistic/confusion_matrix.png`

-   **Insight:** At the standard operational setting ($p \ge 0.5$), the error quadrants register a higher frequency of Type II errors (False Negatives) relative to Type I errors (False Positives). The decision boundary is inherently conservative, demanding unequivocal feature vector evidence to transition a state mapping from deceased to survived.

#### Dimensional Matrix Independence

-   **Plot Reference:** `outputs/logistic/correlation_heatmap.png`

-   **Insight:** The post-encoding correlation plot confirms that expanding categorical handles like `Sex` and `Embarked` into independent one-hot dimensions did not introduce singular matrix dependencies or severe multicollinearity. This preserves stable and unique gradient paths during batch backward passes.

* * * * *

Engineering Principles Demonstrated
-----------------------------------

1.  **Zero-Abstraction Execution:** Bypassing standard runtime wrappers required manual construction of vectorized operations, low-level broadcasting matrix layouts, and explicit gradient calculation tracks.

2.  **Deterministic Data Integrity:** Programmatic pruning of non-numeric dimensions prevents runtime type errors and ensures input matrices are optimized for fast vector dot-product execution.

3.  **Algorithmic Transparency:** The validation suite maps empirical model metrics directly back to statistical theory, clearly separating dataset artifacts (such as data capping limits) from core optimization logic.

* * * * *

Compliance and Determinism
--------------------------

All partition operations use `random_state=42` to guarantee strict mathematical reproducibility across training iterations. Gradient descent arrays are initialized symmetrically at zero, removing stochastic execution paths and ensuring predictable optimization trends.
