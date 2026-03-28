library(Rcpp)
library(microbenchmark)

# Restart R first, then:
sourceCpp("C:\\Users\\29827094\\Documents\\GitHub\\DirichletRF\\dev\\parallel_Rcpp.cpp")   # File 1 functions
sourceCpp("C:\\Users\\29827094\\Documents\\GitHub\\DirichletRF\\dev\\non_parallel_old_version.cpp")   # File 2 functions

# ============================================================
# SIMULATE DATA
# ============================================================
set.seed(42)

simulate_dirichlet <- function(n, alpha, n_features = 5) {
  k <- length(alpha)
  X <- matrix(rnorm(n * n_features), nrow = n, ncol = n_features)
  G <- matrix(NA, n, k)
  for (j in 1:k) G[, j] <- rgamma(n, shape = alpha[j], rate = 1)
  Y <- G / rowSums(G)
  list(X = X, Y = Y)
}

true_alpha <- c(2, 5, 3)
data       <- simulate_dirichlet(n = 200, alpha = true_alpha)
X          <- data$X
Y          <- data$Y

# ============================================================
# PART 1: LIKELIHOOD COMPARISON
# ============================================================
cat("==========================================\n")
cat("PART 1: LIKELIHOOD\n")
cat("==========================================\n")

ll_v1  <- test_loglik_rcpp(Y, true_alpha)
ll_v2  <- test_loglik_v2(Y,   true_alpha)
mom_v1 <- estimate_dirichlet_mom(Y)
mom_v2 <- estimate_dirichlet_mom_v2(Y)
mle_v1 <- estimate_dirichlet_mle(Y)
mle_v2 <- estimate_dirichlet_mle_v2(Y)

cat(sprintf("%-20s | diff: %e | match: %s\n",
            "Log-likelihood",
            abs(ll_v1 - ll_v2),
            isTRUE(all.equal(ll_v1, ll_v2))))

cat(sprintf("%-20s | diff: %e | match: %s\n",
            "MoM estimation",
            max(abs(mom_v1 - mom_v2)),
            isTRUE(all.equal(mom_v1, mom_v2))))

cat(sprintf("%-20s | diff: %e | match: %s\n",
            "MLE estimation",
            max(abs(mle_v1 - mle_v2)),
            isTRUE(all.equal(mle_v1, mle_v2))))

# ============================================================
# PART 2: RANDOM FOREST COMPARISON
# ============================================================
cat("\n==========================================\n")
cat("PART 2: RANDOM FOREST\n")
cat("==========================================\n")
start_time <- Sys.time()

forest_v1 <- DirichletForest(X, Y,
                             B      = 100,
                             d_max  = 10,
                             n_min  = 5,
                             seed   = 123,
                             method = "mom")
end_time <- Sys.time()
runtime <- end_time - start_time
print(runtime)

start_time <- Sys.time()
forest_v2 <- DirichletForest_v2(X, Y,
                                B         = 1000,
                                d_max     = 10,
                                n_min     = 5,
                                seed      = 123,
                                method    = "mom",
                                num_cores = 1)
end_time <- Sys.time()
runtime <- end_time - start_time
print(runtime)

pred_v1 <- PredictDirichletForest(forest_v1,    X)
pred_v2 <- PredictDirichletForest_v2(forest_v2, X)

cat(sprintf("%-20s | diff: %e | match: %s\n",
            "Mean predictions",
            max(abs(pred_v1$mean_predictions -
                      pred_v2$mean_predictions)),
            max(abs(pred_v1$mean_predictions -
                      pred_v2$mean_predictions)) < 1e-10))

cat(sprintf("%-20s | diff: %e | match: %s\n",
            "Alpha predictions",
            max(abs(pred_v1$alpha_predictions -
                      pred_v2$alpha_predictions)),
            max(abs(pred_v1$alpha_predictions -
                      pred_v2$alpha_predictions)) < 1e-10))

# RMSE on new data
set.seed(99)
new_data    <- simulate_dirichlet(n = 50, alpha = true_alpha)
X_new       <- new_data$X
Y_new       <- new_data$Y

pred_new_v1 <- PredictDirichletForest(forest_v1,    X_new)
pred_new_v2 <- PredictDirichletForest_v2(forest_v2, X_new)

rmse        <- function(pred, true) sqrt(mean((pred - true)^2))
rmse_v1     <- rmse(pred_new_v1$mean_predictions, Y_new)
rmse_v2     <- rmse(pred_new_v2$mean_predictions, Y_new)

cat(sprintf("%-20s | v1: %.6f | v2: %.6f | diff: %e\n",
            "RMSE new data",
            rmse_v1, rmse_v2,
            abs(rmse_v1 - rmse_v2)))

# ============================================================
# PART 3: SPEED COMPARISON
# ============================================================
cat("\n==========================================\n")
cat("PART 3: SPEED\n")
cat("==========================================\n")

# 3A: Building blocks
cat("\n--- Building Blocks ---\n")
bb <- microbenchmark(
  mom_v1 = estimate_dirichlet_mom(Y),
  mom_v2 = estimate_dirichlet_mom_v2(Y),
  mle_v1 = estimate_dirichlet_mle(Y),
  mle_v2 = estimate_dirichlet_mle_v2(Y),
  times  = 100
)
s <- summary(bb)
cat(sprintf("%-10s | median: %8.3f ms\n", 
            as.character(s$expr), s$median/1e6))

# 3B: Forest training — vary cores
cat("\n--- Forest Training (n=200, B=100) ---\n")
ft <- microbenchmark(
  v1_nonparallel = DirichletForest(X, Y,
                                   B = 100, d_max = 10,
                                   n_min = 5, seed = 123),
  v2_1core       = DirichletForest_v2(X, Y,
                                      B = 100, d_max = 10,
                                      n_min = 5, seed = 123,
                                      num_cores = 1),
  v2_2core       = DirichletForest_v2(X, Y,
                                      B = 100, d_max = 10,
                                      n_min = 5, seed = 123,
                                      num_cores = 2),
  times = 10
)
s <- summary(ft)
cat(sprintf("%-20s | median: %8.3f sec\n",
            as.character(s$expr), s$median/1e9))

# Speedup
med        <- s$median
speedup_1c <- round(med[1] / med[2], 2)
speedup_2c <- round(med[1] / med[3], 2)
cat(sprintf("\nSpeedup v2_1core vs v1: %.2fx\n", speedup_1c))
cat(sprintf("Speedup v2_2core vs v1: %.2fx\n", speedup_2c))

# 3C: Scale with sample size
cat("\n--- Training Time vs Sample Size (B=50) ---\n")
cat(sprintf("%-8s | %-15s | %-15s | %-15s | speedup\n",
            "n", "v1", "v2_1core", "v2_2core"))

for (n in c(100, 500)) {
  d  <- simulate_dirichlet(n, true_alpha)
  bm <- microbenchmark(
    v1       = DirichletForest(d$X, d$Y,
                               B = 50, d_max = 10,
                               n_min = 5, seed = 123),
    v2_1core = DirichletForest_v2(d$X, d$Y,
                                  B = 50, d_max = 10,
                                  n_min = 5, seed = 123,
                                  num_cores = 1),
    v2_2core = DirichletForest_v2(d$X, d$Y,
                                  B = 50, d_max = 10,
                                  n_min = 5, seed = 123,
                                  num_cores = 2),
    times = 5
  )
  m       <- summary(bm)$median / 1e9
  speedup <- round(m[1] / m[3], 2)
  cat(sprintf("%-8d | %-15.3f | %-15.3f | %-15.3f | %.2fx\n",
              n, m[1], m[2], m[3], speedup))
}

# 3D: Prediction speed
cat("\n--- Prediction Speed (B=100 forest) ---\n")
ps <- microbenchmark(
  pred_v1 = PredictDirichletForest(forest_v1,    X_new),
  pred_v2 = PredictDirichletForest_v2(forest_v2, X_new),
  times   = 50
)
s <- summary(ps)
cat(sprintf("%-10s | median: %8.3f ms\n",
            as.character(s$expr), s$median/1e6))








# ============================================================
# STEP 1: Check if OpenMP actually works in your Rcpp code
# ============================================================

# Add this function temporarily to dirichlet_forest.cpp
# at the bottom, then recompile

# // [[Rcpp::export]]
# int check_openmp(int num_cores) {
# #ifdef _OPENMP
#   omp_set_num_threads(num_cores);
#   int actual = 0;
#   #pragma omp parallel
#   {
#     #pragma omp atomic
#     actual++;
#   }
#   return actual;
# #else
#   return -1;
# #endif
# }

# Then run:
threads <- check_openmp(4)
cat("Requested 4 cores, got:", threads, "\n")
cat("OpenMP compiled:", threads != -1, "\n")
# -1 = OpenMP NOT compiled = parallelism silently disabled
#  1 = OpenMP compiled but only 1 thread running
#  4 = OpenMP working correctly

# ============================================================
# STEP 2: Proper speed test with larger data
# ============================================================
set.seed(42)

simulate_dirichlet <- function(n, alpha, n_features = 5) {
  k <- length(alpha)
  X <- matrix(rnorm(n * n_features), nrow = n, ncol = n_features)
  G <- matrix(NA, n, k)
  for (j in 1:k) G[, j] <- rgamma(n, shape = alpha[j], rate = 1)
  Y <- G / rowSums(G)
  list(X = X, Y = Y)
}

true_alpha <- c(2, 5, 3)

cat("\n--- Speed test across sample sizes ---\n")
cat(sprintf("%-6s | %-10s | %-10s | %-10s | %-10s | speedup\n",
            "n", "v1", "v2_1core", "v2_2core", "v2_4core"))

for (n in c(200, 500, 1000, 2000)) {
  d <- simulate_dirichlet(n, true_alpha)
  
  t_v1 <- system.time(
    DirichletForest(d$X, d$Y,
                    B = 100, d_max = 10,
                    n_min = 5, seed = 123)
  )["elapsed"]
  
  t_1c <- system.time(
    DirichletForest_v2(d$X, d$Y,
                       B = 100, d_max = 10,
                       n_min = 5, seed = 123,
                       num_cores = 1)
  )["elapsed"]
  
  t_2c <- system.time(
    DirichletForest_v2(d$X, d$Y,
                       B = 100, d_max = 10,
                       n_min = 5, seed = 123,
                       num_cores = 2)
  )["elapsed"]
  
  t_4c <- system.time(
    DirichletForest_v2(d$X, d$Y,
                       B = 100, d_max = 10,
                       n_min = 5, seed = 123,
                       num_cores = 4)
  )["elapsed"]
  
  cat(sprintf("%-6d | %-10.2f | %-10.2f | %-10.2f | %-10.2f | %.2fx\n",
              n, t_v1, t_1c, t_2c, t_4c,
              t_v1 / t_4c))
}