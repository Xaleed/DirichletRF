src_path <- "C:/Users/29827094/Documents/GitHub/DirichletRF/src"

# ============================================================
# Correct Makevars — use CXXFLAGS in both lines
# ============================================================
con <- file(file.path(src_path, "Makevars"), "w")
writeLines(
  c("PKG_CXXFLAGS = $(SHLIB_OPENMP_CXXFLAGS)",
    "PKG_LIBS = $(SHLIB_OPENMP_CXXFLAGS)"),  # CXX not C
  con
)
close(con)

# ============================================================
# Correct Makevars.win — same fix
# ============================================================
con <- file(file.path(src_path, "Makevars.win"), "w")
writeLines(
  c("PKG_CXXFLAGS = $(SHLIB_OPENMP_CXXFLAGS)",
    "PKG_LIBS = $(SHLIB_OPENMP_CXXFLAGS)"),  # CXX not C
  con
)
close(con)

# ============================================================
# Verify
# ============================================================
cat("Makevars:\n")
cat(readLines(file.path(src_path, "Makevars")), sep = "\n")
cat("\n\nMakevars.win:\n")
cat(readLines(file.path(src_path, "Makevars.win")), sep = "\n")







pkg_path <- "C:/Users/29827094/Documents/GitHub/DirichletRF"

# Search ALL files in package for C++11 reference
all_files <- list.files(pkg_path, recursive = TRUE, 
                        full.names = TRUE, all.files = TRUE)

cat("Searching all files for C++11...\n\n")
for (f in all_files) {
  tryCatch({
    lines   <- readLines(f, warn = FALSE)
    matches <- grep("CXX11|CXX_STD|cpp11|c\\+\\+11", 
                    lines, value = TRUE, ignore.case = TRUE)
    if (length(matches) > 0) {
      cat("Found in:", f, "\n")
      cat("  Line:", matches, "\n\n")
    }
  }, error = function(e) NULL)
}
cat("Search complete\n")





desc_path <- "C:/Users/29827094/Documents/GitHub/DirichletRF/DESCRIPTION"

# Remove any SystemRequirements line with C++11
lines     <- readLines(desc_path)
new_lines <- lines[!grepl("SystemRequirements.*C\\+\\+11|CXX11", 
                          lines, ignore.case = TRUE)]

if (length(new_lines) < length(lines)) {
  writeLines(new_lines, desc_path)
  cat("Removed C++11 from DESCRIPTION\n")
} else {
  cat("Not found in source DESCRIPTION\n")
  cat("Issue is in INSTALLED version — just reinstall\n")
}

# Verify
cat("\nCurrent DESCRIPTION:\n")
cat(readLines(desc_path), sep = "\n")





#creat package
setwd("C:\\Users\\29827094\\Documents\\GitHub\\DirichletRF")  # Set working directory to package root

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


environment(predict)
# or
find("predict")



find("predict")
getAnywhere("predict.dirichlet_forest")


# Remove the rogue predict from global environment

# ============================================================
# Remove rogue predict from global environment
# ============================================================

# ============================================================
# Remove rogue predict from global environment
# ============================================================

# Check if predict exists in global environment
if (exists("predict", envir = .GlobalEnv, inherits = FALSE)) {
  cat("Found rogue predict in .GlobalEnv — removing...\n")
  rm("predict", envir = .GlobalEnv)
  cat("Removed successfully\n")
} else {
  cat("No predict in .GlobalEnv\n")
}

# Verify correct predict is now active
cat("\npredict now lives in:", 
    environmentName(environment(predict)), "\n")
# Should show: base  or  package:stats

# Confirm all methods still available
cat("\nAvailable predict methods:\n")
print(methods("predict"))

library(DirichletRF)

# Fix seed for reproducibility
set.seed(42)

n <- 300
p <- 4
X <- matrix(rnorm(n * p), n, p)

# Generate Dirichlet responses using base R
alpha <- c(2, 3, 4)
G <- matrix(rgamma(n * length(alpha), shape = rep(alpha, each = n)), n, length(alpha))
Y <- G / rowSums(G)

# ========================================
# FITTING MODELS
# ========================================

# Fix seed again before forest — ensures same R-level randomness
set.seed(42)
system.time({
  forest1 <- DirichletRF(X, Y, num.trees = 300, num.cores = 4, seed = 42)
})
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
set.seed(42)

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

# Small toy example (auto-tested)
set.seed(42)
n <- 50; p <- 2
X <- matrix(rnorm(n * p), n, p)
G <- matrix(rgamma(n * 3, shape = rep(c(2, 3, 4), each = n)), n, 3)
Y <- G / rowSums(G)
forest <- DirichletRF(X, Y, num.trees = 5, num.cores = 1)
print(forest)
Xtest <- matrix(rnorm(5 * p), 5, p)
pred  <- predict(forest, Xtest)

