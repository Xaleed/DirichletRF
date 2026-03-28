# DirichletRF 
<p align="center">
  <img src="assets/DirichletRF.jpg" alt="DirichletRF Logo" width="200"/>
</p>

This repository contains an implementation of a **parallel Dirichlet Random Forest**, designed for modeling **compositional data** in accordance with compositional data analysis (CoDA) principles.

---

## 📦 Installation

### Option 1: Install from CRAN
```r
install.packages("DirichletRF")
```

### Option 2: Install from GitHub
```r
devtools::install_github("Xaleed/DirichletRF")
```

### Option 3: Install pre-built binary (no Rtools required — Windows only)

Download and install the latest binary release:
```r
# Replace v0.1.0 with the latest release version
install.packages("https://github.com/Xaleed/DirichletRF/releases/download/v0.1.0/DirichletRF_0.1.0.zip",
                 repos = NULL, type = "win.binary")
```

Or manually:
1. Go to [Releases](https://github.com/Xaleed/DirichletRF/releases)
2. Download the `.zip` file from the latest release
3. In R: `install.packages("path/to/downloaded/file.zip", repos = NULL, type = "win.binary")`

---

## 🚀 Quick Start
```r
library(DirichletRF)

# Generate predictors
n <- 500
p <- 4
X <- matrix(rnorm(n * p), n, p)

# Generate compositional responses
alpha <- c(2, 3, 4)
G <- matrix(rgamma(n * length(alpha), shape = rep(alpha, each = n)), n, length(alpha))
Y <- G / rowSums(G)

# Fit a Dirichlet Forest with 100 trees (uses all cores minus one by default)
forest <- DirichletRF(X, Y, num.trees = 100)

# Print a summary of the fitted model
print(forest)

# Predict on new data
X_test <- matrix(rnorm(10 * p), 10, p)
pred <- predict(forest, X_test)

# Access predictions
print(pred$mean_predictions)       # Mean-based predictions
print(pred$alpha_predictions)      # Estimated Dirichlet alpha parameters

# Normalise alpha predictions to get parameter-based predictions
param_pred <- pred$alpha_predictions / rowSums(pred$alpha_predictions)

# Access fitted values
print(forest$fitted$alpha_hat)     # Estimated alpha parameters (α̂)
print(forest$fitted$mean_based)    # Fitted values from sample means
print(forest$fitted$param_based)   # Fitted values from normalised parameters

# Access residuals
print(forest$residuals$mean_based)  # Residuals for mean-based predictions
print(forest$residuals$param_based) # Residuals for parameter-based predictions
```

---

## 🔧 Key Features

### **Parallel Processing via OpenMP**
- Trees are built in parallel using **OpenMP** directly in C++
- Set `num.cores = -1` (default) to use all available cores minus one
- Set `num.cores = 1` to build sequentially
- No cluster setup or cleanup required

### **Two Prediction Modes**

#### Mean-Based Predictions
Predictions are computed as the average of compositional responses within each terminal node, then averaged across all trees:
```r
pred <- predict(forest, X_test)
print(pred$mean_predictions)
```

#### Parameter-Based Predictions
Predictions are derived from the estimated Dirichlet alpha parameters within each terminal node, then normalised:
```r
param_pred <- pred$alpha_predictions / rowSums(pred$alpha_predictions)
```

### **Parameter Estimation**
Choose between Method of Moments (`est.method = "mom"`, default) or Maximum Likelihood Estimation (`est.method = "mle"`):
```r
forest_mle <- DirichletRF(X, Y, est.method = "mle")
```

### **Fitted Values and Residuals**
The model automatically computes and stores three types of fitted values and two types of residuals after training:
```r
# Fitted values
alpha_hat   <- forest$fitted$alpha_hat     # Estimated concentration parameters (α̂)
mean_fit    <- forest$fitted$mean_based    # Mean-based fitted values
param_fit   <- forest$fitted$param_based   # Parameter-based fitted values

# Residuals (Y - fitted)
resid_mean  <- forest$residuals$mean_based
resid_param <- forest$residuals$param_based

# Compare RMSE
rmse_mean  <- sqrt(mean(resid_mean^2))
rmse_param <- sqrt(mean(resid_param^2))
print(paste("RMSE (mean-based):", round(rmse_mean, 4)))
print(paste("RMSE (param-based):", round(rmse_param, 4)))
```

---

## ⚙️ Function Reference

### `DirichletRF()`
Main function to build a Dirichlet Random Forest.

**Parameters:**
- `X`: Numeric predictor matrix (n × p). Only numeric covariates are supported; use one-hot encoding for categorical variables.
- `Y`: Compositional response matrix (n × k); rows must sum to 1
- `num.trees`: Number of trees (default: `100`)
- `max.depth`: Maximum tree depth (default: `10`)
- `min.node.size`: Minimum observations per leaf (default: `5`)
- `mtry`: Number of candidate features at each split; `-1` uses `sqrt(p)` (default: `-1`)
- `seed`: Random seed for the C++ RNG (default: `123`)
- `est.method`: Dirichlet parameter estimation method, `"mom"` or `"mle"` (default: `"mom"`)
- `num.cores`: Number of OpenMP threads; `-1` uses all cores minus one (default: `-1`)

**Note:** Out-of-bag (OOB) error estimation is not supported in this version.

**Returns:** A `dirichlet_forest` object containing:
- `type`: Parallelisation type (`"openmp"` or `"sequential"`)
- `num.cores`: Number of cores used
- `num.trees`: Total number of trees
- `Y_train`: Training responses
- `fitted`: List with `alpha_hat`, `mean_based`, and `param_based` fitted values
- `residuals`: List with `mean_based` and `param_based` residuals

---

### `predict.dirichlet_forest()`
Make predictions with a trained forest. Called via the standard `predict()` generic.

**Parameters:**
- `object`: A `dirichlet_forest` object returned by `DirichletRF()`
- `newdata`: Numeric matrix of new covariates (n_new × p), or a vector for a single observation
- `...`: Currently unused

**Returns:** A list with:
- `alpha_predictions`: Estimated Dirichlet alpha parameters (n_new × k matrix)
- `mean_predictions`: Mean-based compositional predictions (n_new × k matrix)
```r
pred <- predict(forest, X_test)

# Mean-based predictions
print(pred$mean_predictions)

# Parameter-based predictions (normalise alpha)
param_pred <- pred$alpha_predictions / rowSums(pred$alpha_predictions)

# Single observation prediction
single_pred <- predict(forest, X_test[1, , drop = FALSE])
```

---

### `print.dirichlet_forest()`
Prints a concise summary of the fitted model, suppressing large data structures.
```r
print(forest)
# ============================================
# Dirichlet Forest Model
# ============================================
#  Type:          openmp
#  Total Trees:   100
#  Cores Used:    7
#  Training Data: 500 observations (n) x 3 components (k)
# --------------------------------------------
#  Note: Large data structures (fitted values,
#        residuals) are suppressed.
#
#  Access via:
#    $Y_train
#    $fitted$alpha_hat
#    $fitted$mean_based
#    $fitted$param_based
#    $residuals$mean_based
#    $residuals$param_based
# ============================================
```

---

## 💡 Tips

1. **Numeric covariates only**: The current version does not support categorical covariates directly. Use one-hot encoding as a workaround.
2. **Normalised responses**: Ensure rows of `Y` sum to 1 before fitting.
3. **Core selection**: `num.cores = -1` (default) automatically uses all available cores minus one. Use `num.cores = 1` for fully sequential execution.
4. **OOB not available**: This implementation does not support out-of-bag error estimation.
5. **Parameter vs mean predictions**: Use `mean_predictions` for averaging-based results and normalised `alpha_predictions` for parameter-based results.

---

## 📄 License

GPL-3

---

## 📚 Reference

Masoumifard, K., van der Westhuizen, S., & Gardner-Lubbe, S. (2026).
Dirichlet random forest for predicting compositional data.
In A. Bekker, P. Nagar, J. Ferreira, B. Erasmus, & A. Ramoelo (Eds.),
Environmental Modelling with Contemporary Statistics: Learning, Directionality, and Space-Time Dynamics.
Chapman & Hall/CRC. ISBN: 9781032903910.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.