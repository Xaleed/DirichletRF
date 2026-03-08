setwd("C:\\Users\\Khaled\\Documents\\GitHub\\DirichletRF")  # Set working directory to package root
getwd()  # Verify path
library(Rcpp)

remove.packages("DirichletRF")
.rs.restartR()  # Restart R session to clear loaded package

#_______________________________________________________________________________
# 1. Recompile Rcpp bindings from src/
Rcpp::compileAttributes()

# 2. Regenerate NAMESPACE + man/ documentation
devtools::document()

# 3. Check for any issues
devtools::check()

# 4. Install the package
devtools::install()

library(DirichletRF)
ls("package:DirichletRF")


?DirichletRF










library(DirichletRF)
n <- 200
p <- 4
X <- matrix(rnorm(n * p), n, p)

# Generate Dirichlet responses using base R
alpha <- c(2, 3, 4)
G <- matrix(rgamma(n * length(alpha), shape = rep(alpha, each = n)), n, length(alpha))
Y <- G / rowSums(G)

# ========================================
# FITTING MODELS
# ========================================

# Example 1: Basic Dirichlet forest (sequential for CRAN compliance)
forest1 <- DirichletRF(X, Y, num.trees = 100, num.cores = 3)

# Example 2: Using maximum likelihood instead of method of moments
#forest2 <- DirichletRF(X, Y, num.trees = 50, est.method = "mle")

# ========================================
# ACCESSING FITTED VALUES AND RESIDUALS
# ========================================

# Three types of fitted values
alpha_hat  <- forest1$fitted$alpha_hat    # Estimated Dirichlet parameters
mean_fit   <- forest1$fitted$mean_based   # Predictions from sample means
param_fit  <- forest1$fitted$param_based  # Predictions from normalised parameters

# Corresponding residuals
resid_mean  <- forest1$residuals$mean_based
resid_param <- forest1$residuals$param_based

# ========================================
# MAKING PREDICTIONS
# ========================================

# Create test data
Xtest <- matrix(rnorm(10 * p), 10, p)

# Make predictions
pred <- predict(forest1, Xtest)
?predict
# Access different prediction types
print(pred$mean_predictions)   # Direct mean-based predictions
print(pred$alpha_predictions)  # Estimated Dirichlet parameters

# Parameter-based predictions (normalised alphas)
param_pred <- pred$alpha_predictions / rowSums(pred$alpha_predictions)
print(param_pred)

# Predict on a single observation
single_pred <- predict(forest1, Xtest[1, , drop = FALSE])

# ========================================
# CLEANUP
# ========================================

# Always clean up at the end, especially important on Windows
cleanupForest(forest1)
#cleanupForest(forest2)



