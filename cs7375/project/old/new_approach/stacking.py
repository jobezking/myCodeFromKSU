import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

# ============================================================
# 1. Base models (reuse the ones defined above)
# ============================================================

# county_model, nchs_model, socio_model already defined and fit in previous section
# If running standalone, re-define and fit them as before.

# ============================================================
# 2. Stacking ensemble estimator
# ============================================================
class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Build out-of-fold predictions from base models on X_stack to train a meta-model.
    At predict time, combine base predictions through the trained meta-model.
    """

    def __init__(self, county_model, nchs_model, socio_model, n_splits=5, random_state=42):
        self.county_model = county_model
        self.nchs_model = nchs_model
        self.socio_model = socio_model
        self.n_splits = n_splits
        self.random_state = random_state
        self.meta_model = LinearRegression()

        # For re-fitting base models on full data post-stacking
        self.county_model_full_ = None
        self.nchs_model_full_ = None
        self.socio_model_full_ = None

    def _prepare_inputs(self, X):
        Xc = X.copy()
        Xc["StateCounty"] = Xc["State"] + "|" + Xc["County"]

        county_input = Xc[["StateCounty", "LifeExpectancyStandardError"]]
        nchs_input   = Xc[["Year", "Race", "Sex", "AgeAdjustedDeathRate"]]
        socio_input  = Xc[["StateCounty", "MedianIncome", "EducationLevel"]]
        return county_input, nchs_input, socio_input

    def fit(self, X_stack, y_stack,
            X_county_full, y_county_full,
            X_nchs_full, y_nchs_full,
            X_socio_full, y_socio_full):
        """
        Fit stacking ensemble.
        - X_stack, y_stack: shared dataset with all required features for prediction and target for stacking.
        - *_full: full datasets for each base model to refit after stacking.
        """

        X_stack = X_stack.copy()
        y_stack = np.asarray(y_stack).reshape(-1)

        # K-fold OOF predictions
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros((len(X_stack), 3))

        for train_idx, val_idx in kf.split(X_stack):
            X_tr = X_stack.iloc[train_idx]
            X_val = X_stack.iloc[val_idx]

            y_tr = y_stack[train_idx]

            # Fit base models on fold training data
            c_tr, n_tr, s_tr = self._prepare_inputs(X_tr)
            c_val, n_val, s_val = self._prepare_inputs(X_val)

            c_model = Pipeline(self.county_model.steps)  # clone pipeline structure
            n_model = Pipeline(self.nchs_model.steps)
            s_model = Pipeline(self.socio_model.steps)

            c_model.fit(c_tr, y_tr)
            n_model.fit(n_tr, y_tr)
            s_model.fit(s_tr, y_tr)

            # OOF predictions for this fold
            oof_preds[val_idx, 0] = c_model.predict(c_val)
            oof_preds[val_idx, 1] = n_model.predict(n_val)
            oof_preds[val_idx, 2] = s_model.predict(s_val)

        # Train meta-model on OOF predictions
        self.meta_model.fit(oof_preds, y_stack)

        # Refit base models on their full datasets for final predictions
        self.county_model_full_ = Pipeline(self.county_model.steps)
        self.nchs_model_full_   = Pipeline(self.nchs_model.steps)
        self.socio_model_full_  = Pipeline(self.socio_model.steps)

        self.county_model_full_.fit(X_county_full, y_county_full)
        self.nchs_model_full_.fit(X_nchs_full, y_nchs_full)
        self.socio_model_full_.fit(X_socio_full, y_socio_full)

        return self

    def predict(self, X):
        # Predict with refit base models, then pass to meta-model
        c_in, n_in, s_in = self._prepare_inputs(X)
        p_c = self.county_model_full_.predict(c_in)
        p_n = self.nchs_model_full_.predict(n_in)
        p_s = self.socio_model_full_.predict(s_in)
        base_stack = np.column_stack([p_c, p_n, p_s])
        return self.meta_model.predict(base_stack)

# ============================================================
# 3. Example usage
# ============================================================

# Build a stacking dataset with all required features + target y_stack.
# This dataset must have columns needed by all base models:
# State, County, Year, Race, Sex, LifeExpectancyStandardError, AgeAdjustedDeathRate, MedianIncome, EducationLevel, LifeExpectancy

# Example placeholder (replace with real, aligned rows)
X_stack = pd.DataFrame([
    {
        "State": "Alabama",
        "County": "Autauga County",
        "Year": 2010,
        "Race": "All Races",
        "Sex": "Both Sexes",
        "LifeExpectancyStandardError": np.nan,
        "AgeAdjustedDeathRate": np.nan,
        "MedianIncome": 52000,
        "EducationLevel": "Medium"
    },
    {
        "State": "Mississippi",
        "County": "Washington County",
        "Year": 2010,
        "Race": "All Races",
        "Sex": "Both Sexes",
        "LifeExpectancyStandardError": np.nan,
        "AgeAdjustedDeathRate": np.nan,
        "MedianIncome": 41000,
        "EducationLevel": "Low"
    }
])
y_stack = pd.Series([75.2, 71.5])

# Full datasets for refitting base models (use the actual prepared matrices):
X_county_full = df_county[["StateCounty", "LifeExpectancyStandardError"]]
y_county_full = df_county["LifeExpectancy"]

X_nchs_full = df_nchs[["Year", "Race", "Sex", "AgeAdjustedDeathRate"]]
y_nchs_full = df_nchs["LifeExpectancy"]

X_socio_full = df_socio[["StateCounty", "MedianIncome", "EducationLevel"]]
y_socio_full = df_socio["LifeExpectancy"]

stack_ensemble = StackingEnsemble(county_model, nchs_model, socio_model, n_splits=5)
stack_ensemble.fit(
    X_stack, y_stack,
    X_county_full, y_county_full,
    X_nchs_full, y_nchs_full,
    X_socio_full, y_socio_full
)

X_pred = pd.DataFrame([{
    "State": "Alabama",
    "County": "Autauga County",
    "Year": 2010,
    "Race": "All Races",
    "Sex": "Both Sexes",
    "LifeExpectancyStandardError": np.nan,
    "AgeAdjustedDeathRate": np.nan,
    "MedianIncome": 52000,
    "EducationLevel": "Medium"
}])
y_hat = stack_ensemble.predict(X_pred)
print("Stacking-ensemble prediction:", float(y_hat[0]))
