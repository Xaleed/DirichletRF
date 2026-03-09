#' Build a Dirichlet Regression Forest for Compositional Responses
#'
#' Build a Dirichlet regression forest for compositional responses. In
#' compositional data analysis (CoDA), parts reside in the simplex, and this
#' regression forest ensures model output abide by CoDA principles. The
#' implementation can be done using parallel processing. Note that
#' out-of-bag (OOB) error is not available in the current version; each tree
#' is built on the entire training sample with no sample fraction used.
#'
#' The forest provides two types of fitted values: mean-based predictions
#' (derived from sample means at each leaf) and parameter-based predictions
#' (derived from normalised Dirichlet alpha parameters).
#'
#' @param X A numeric (n x p) matrix of covariates. Note that the current
#'   version only allows numeric covariates. Users may use one-hot encoding
#'   to possibly include categorical covariates.
#' @param Y A numeric (n x k) matrix of compositional responses. Each row
#'   should sum to 1. That is, data should already be normalised if needed.
#' @param num.trees Number of trees grown in the forest. Default is 100.
#' @param max.depth Maximum depth of trees. Default is 10.
#' @param min.node.size Minimum size of observations in each tree leaf.
#'   Default is 5. Note that nodes with sizes smaller than \code{min.node.size}
#'   can occur.
#' @param mtry Number of covariates randomly selected as candidates at each
#'   split. Default is \code{sqrt(p)}, indicated by \code{-1}.
#' @param seed The seed of the C++ random number generator.
#' @param est.method Parameter estimation method for the Dirichlet distribution
#'   when splitting is done. Users may either use maximum likelihood
#'   (\code{"mle"}) or method of moments (\code{"mom"}). Default is
#'   \code{"mom"}.
#' @param num.cores Building a Dirichlet tree is computationally heavy.
#'   Therefore, the forest can be built over multiple cores. The default is
#'   \code{-1} which uses all the cores on the system minus 1. Users may also
#'   specify \code{1} which means that the forest will be built sequentially.
#'
#' @return A \code{dirichlet_forest} object containing the following
#'   user-accessible components:
#' \describe{
#'   \item{\code{type}}{Parallelisation type used: \code{"sequential"},
#'     \code{"fork"}, or \code{"cluster"}.}
#'   \item{\code{num.cores}}{Number of cores used.}
#'   \item{\code{trees.per.worker}}{Number of trees assigned to each worker.}
#'   \item{\code{num.trees}}{Total number of trees in the forest.}
#'   \item{\code{Y_train}}{Training data used (no OOB).}
#'   \item{\code{fitted}}{A list of fitted values:
#'     \describe{
#'       \item{\code{alpha_hat}}{Estimated Dirichlet alpha parameters
#'         (n x k matrix).}
#'       \item{\code{mean_based}}{Mean-based fitted values (n x k matrix).}
#'       \item{\code{param_based}}{Parameter-based fitted values obtained by
#'         normalising \code{alpha_hat} (n x k matrix).}
#'     }
#'   }
#'   \item{\code{residuals}}{A list of residuals (Y - fitted):
#'     \describe{
#'       \item{\code{mean_based}}{Residuals from mean-based predictions.}
#'       \item{\code{param_based}}{Residuals from parameter-based predictions.}
#'     }
#'   }
#' }
#'
#' @references
#' Masoumifard, K., van der Westhuizen, S., & Gardner-Lubbe, S. (In press).
#' Dirichlet-random forest for predicting compositional data. In A. Bekker,
#' J. Ferreira, & P. Nagar (Eds.), \emph{Environmental Statistics: Innovative
#' Methods and Applications}. CRC Press.
#'
#' @examples
#' \donttest{
#' library(DirichletRF)
#' n <- 300; p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#'
#' # Generate Dirichlet with three parts from independent Gamma random variables
#' alpha <- c(2, 3, 4)
#' G <- matrix(rgamma(n * length(alpha), shape = rep(alpha, each = n)), n, length(alpha))
#' Y <- G / rowSums(G)
#'
#' # ========================================
#' # FITTING MODELS
#' # ========================================
#'
#' # Fit models in two examples. 1) Using "mom", and 2) using "mle"
#' # Example 1 - Using method of moments
#' forest1 <- DirichletRF(X, Y, num.trees = 100, num.cores = 1)
#'
#' # Example 2 - Using maximum likelihood
#' forest2 <- DirichletRF(X, Y, num.trees = 50, est.method = "mle", num.cores = 1)
#'
#' # ========================================
#' # ACCESSING FITTED VALUES AND RESIDUALS
#' # ========================================
#'
#' # Three types of fitted values
#' alpha_hat  <- forest1$fitted$alpha_hat    # Estimated Dirichlet parameters
#' mean_fit   <- forest1$fitted$mean_based   # Predictions from sample means
#' param_fit  <- forest1$fitted$param_based  # Predictions from normalised parameters
#'
#' # Corresponding residuals
#' resid_mean  <- forest1$residuals$mean_based
#' resid_param <- forest1$residuals$param_based
#'
#' # ========================================
#' # MAKING PREDICTIONS
#' # ========================================
#'
#' # Create test data
#' Xtest <- matrix(rnorm(10 * p), 10, p)
#'
#' # Make predictions
#' pred <- predict(forest1, Xtest)
#'
#' # Access different prediction types
#' print(pred$mean_predictions)   # Direct mean-based predictions
#' print(pred$alpha_predictions)  # Estimated Dirichlet parameters
#'
#' # Parameter-based predictions (normalised alphas)
#' param_pred <- pred$alpha_predictions / rowSums(pred$alpha_predictions)
#' print(param_pred)
#'
#'
#' # ========================================
#' # CLEANUP
#' # ========================================
#'
#' # Always clean up at the end, especially important on Windows
#' cleanupForest(forest1)
#' cleanupForest(forest2)
#' }
#'
#' @seealso
#' \code{\link{predict.dirichlet_forest}} for making predictions on new data.
#' \code{\link{cleanupForest}} for releasing cluster resources on Windows.
#' @importFrom stats predict
#' @export
DirichletRF <- function(X, Y, num.trees = 100, max.depth = 10, min.node.size = 5,
                        mtry = -1, seed = 123, est.method = "mom",
                        num.cores = -1) {

  # In this version, this should be hardcoded, but later we will need to
  # import it as an input variable.
  store_samples <- FALSE
  # use_leaf_predictions = TRUE is also hardcoded for now.

  # Input validation
  if (!is.matrix(X) || !is.matrix(Y)) {
    stop("X and Y must be matrices")
  }

  if (nrow(X) != nrow(Y)) {
    stop("X and Y must have the same number of rows")
  }

  # Handle parallel package dependency
  if (num.cores != 1) {
    if (!requireNamespace("parallel", quietly = TRUE)) {
      stop("Package 'parallel' is required for distributed computing but not available.\n",
           "Please install it with: install.packages('parallel')\n",
           "Or set num.cores = 1 to use sequential processing.")
    }
  }

  # Force sequential if num.cores = 1
  if (num.cores == 1) {
    forest_seq <- DirichletForest(X, Y, num.trees, max.depth, min.node.size,
                                  mtry, seed, est.method, store_samples)
    result <- list(
      type             = "sequential",
      forest           = forest_seq,   # internal use only
      num.cores        = 1,
      trees.per.worker = num.trees,
      num.trees        = num.trees,
      Y_train          = Y
    )
    class(result) <- c("dirichlet_forest", "list")

    cat("Computing fitted values and residuals...\n")
    fitted_preds <- predict(result, X, est.method = est.method)
    alpha_means  <- fitted_preds$alpha_predictions /
                    rowSums(fitted_preds$alpha_predictions)

    result$fitted <- list(
      alpha_hat   = fitted_preds$alpha_predictions,
      mean_based  = fitted_preds$mean_predictions,
      param_based = alpha_means
    )
    result$residuals <- list(
      mean_based  = Y - fitted_preds$mean_predictions,
      param_based = Y - alpha_means
    )
    return(result)
  }

  # Determine cores for parallel processing
  if (num.cores == -1) {
    num.cores <- max(1, parallel::detectCores() - 1)
  }
  num.cores <- max(1, min(num.cores, num.trees))

  # For small forests, fall back to sequential
  if (num.trees < max(4, num.cores)) {
    forest_seq <- DirichletForest(X, Y, num.trees, max.depth, min.node.size,
                                  mtry, seed, est.method, store_samples)
    result <- list(
      type             = "sequential",
      forest           = forest_seq,   # internal use only
      num.cores        = 1,
      trees.per.worker = num.trees,
      num.trees        = num.trees,
      Y_train          = Y
    )
    class(result) <- c("dirichlet_forest", "list")

    cat("Computing fitted values and residuals...\n")
    fitted_preds <- predict(result, X, est.method = est.method)
    alpha_means  <- fitted_preds$alpha_predictions /
                    rowSums(fitted_preds$alpha_predictions)

    result$fitted <- list(
      alpha_hat   = fitted_preds$alpha_predictions,
      mean_based  = fitted_preds$mean_predictions,
      param_based = alpha_means
    )
    result$residuals <- list(
      mean_based  = Y - fitted_preds$mean_predictions,
      param_based = Y - alpha_means
    )
    return(result)
  }

  cat("Building distributed forest with", num.cores, "workers for", num.trees, "trees\n")

  # Distribute trees across workers
  trees_per_core <- rep(num.trees %/% num.cores, num.cores)
  remainder      <- num.trees %% num.cores
  if (remainder > 0) {
    trees_per_core[1:remainder] <- trees_per_core[1:remainder] + 1
  }

  # Create seeds for each worker
  worker_seeds <- seq(seed, seed + num.cores * 99991, length.out = num.cores)
  cat("Tree distribution:", paste(trees_per_core, collapse = ", "), "\n")

  if (.Platform$OS.type != "windows") {
    # Unix/Mac: fork-based
    cat("Using fork-based parallelization\n")

    worker_forests <- parallel::mclapply(seq_len(num.cores), function(i) {
      DirichletForest(X, Y, B = trees_per_core[i], d_max = max.depth,
                      n_min = min.node.size, m_try = mtry,
                      seed = worker_seeds[i], method = est.method,
                      store_samples = store_samples)
    }, mc.cores = num.cores)

    result <- list(
      type             = "fork",
      worker_forests   = worker_forests,   # internal use only
      num.cores        = num.cores,
      trees.per.worker = trees_per_core,
      num.trees        = sum(trees_per_core),
      Y_train          = Y
    )
    class(result) <- c("dirichlet_forest", "list")

    cat("Computing fitted values and residuals...\n")
    fitted_preds <- predict(result, X, est.method = est.method)
    alpha_means  <- fitted_preds$alpha_predictions /
                    rowSums(fitted_preds$alpha_predictions)

    result$fitted <- list(
      alpha_hat   = fitted_preds$alpha_predictions,
      mean_based  = fitted_preds$mean_predictions,
      param_based = alpha_means
    )
    result$residuals <- list(
      mean_based  = Y - fitted_preds$mean_predictions,
      param_based = Y - alpha_means
    )
    return(result)

  } else {
    # Windows: cluster-based — keep workers alive for predictions
    cat("Using persistent cluster (Windows)\n")

    cl <- parallel::makeCluster(num.cores, type = "PSOCK")

    # Setup workers with Rcpp functions
    # for package:
    setup_cluster_workers_installed(cl)
    # for test:
    # setup_cluster_workers(cl)

    # Export variables to workers
    parallel::clusterExport(cl, c("X", "Y", "max.depth", "min.node.size", "mtry",
                                  "est.method", "trees_per_core", "worker_seeds",
                                  "store_samples"),
                            envir = environment())

    # Build forests in each worker
    parallel::clusterApply(cl, seq_len(num.cores), function(worker_id) {
      worker_forest <- DirichletForest(X, Y, B = trees_per_core[worker_id],
                                       d_max = max.depth, n_min = min.node.size,
                                       m_try = mtry,
                                       seed = worker_seeds[worker_id],
                                       method = est.method,
                                       store_samples = store_samples)
      # NOTE: Assigning to .GlobalEnv is intentional here. This code runs
      # inside a separate cluster worker process (not the main R session).
      # Workers need these objects in their global env for retrieval via
      # subsequent clusterApply calls. This is the standard pattern for
      # persistent PSOCK cluster workers.
      assign("worker_forest", worker_forest, envir = .GlobalEnv) # nolint
      assign("Y_train", Y,                   envir = .GlobalEnv) # nolint
      return(worker_forest$n_trees)
    })

    result <- list(
      type             = "cluster",
      cluster          = cl,           # internal use only
      num.cores        = num.cores,
      trees.per.worker = trees_per_core,
      num.trees        = sum(trees_per_core),
      Y_train          = Y
    )
    class(result) <- c("dirichlet_forest", "list")

    cat("Computing fitted values and residuals...\n")
    fitted_preds <- predict(result, X, est.method = est.method)
    alpha_means  <- fitted_preds$alpha_predictions /
                    rowSums(fitted_preds$alpha_predictions)

    result$fitted <- list(
      alpha_hat   = fitted_preds$alpha_predictions,
      mean_based  = fitted_preds$mean_predictions,
      param_based = alpha_means
    )
    result$residuals <- list(
      mean_based  = Y - fitted_preds$mean_predictions,
      param_based = Y - alpha_means
    )
    return(result)
  }
}


#' Custom Print Method for dirichlet_forest Objects
#'
#' Suppresses the display of large data matrices (Y_train, fitted, residuals)
#' when the object is printed, while keeping them accessible via \code{$}.
#'
#' @param x A \code{dirichlet_forest} object.
#' @param ... Further arguments passed to or from other methods.
#' @export
print.dirichlet_forest <- function(x, ...) {

  cat("============================================\n")
  cat(" Dirichlet Forest Model\n")
  cat("============================================\n")

  # 1. Model metadata
  cat(" Type:", x$type, "\n")
  cat(" Total Trees:", x$num.trees, "\n")
  cat(" Cores Used:", x$num.cores, "\n")

  # 2. Worker details
  if (length(x$trees.per.worker) > 1 && x$num.cores > 1) {
    cat(" Trees per Worker:", paste(x$trees.per.worker, collapse = ", "), "\n")
  }

  # 3. Cluster/worker status
  if (x$type == "cluster" && !is.null(x$cluster)) {
    cat(" Cluster Status: Active (", length(x$cluster), " workers)\n", sep = "")
  } else if (x$type == "fork") {
    cat(" Worker Status: Forked (", x$num.cores, " workers)\n", sep = "")
  } else if (x$type == "sequential") {
    cat(" Worker Status: Sequential\n")
  }

  # 4. Data dimensions
  N <- NULL
  K <- NULL
  if (!is.null(x$Y_train)) {
    N <- nrow(x$Y_train)
    K <- ncol(x$Y_train)
  } else if (x$type == "sequential" && !is.null(x$forest$Y_train)) {
    N <- nrow(x$forest$Y_train)
    K <- ncol(x$forest$Y_train)
  }

  if (!is.null(N) && !is.null(K)) {
    cat(paste0(" Training Data Size: ", N, " observations (n) x ", K, " components (k)\n"))
  }

  cat("--------------------------------------------\n")
  cat(" Note: Large data structures (fitted values, residuals) are suppressed.\n")
  cat("\n Access data via the following components:\n")
  cat(" - Training Data (Y):                   $Y_train\n")
  cat(" - Fitted Values:\n")
  cat("   - Estimated Alpha Parameters:        $fitted$alpha_hat\n")
  cat("   - Mean-based Predictions:            $fitted$mean_based\n")
  cat("   - Parameter-based Predictions:       $fitted$param_based\n")
  cat(" - Residuals:\n")
  cat("   - Mean-based Residuals:              $residuals$mean_based\n")
  cat("   - Parameter-based Residuals:         $residuals$param_based\n")
  cat("============================================\n")

  invisible(x)
}


#' Clean Up a Dirichlet Forest
#'
#' Properly releases resources held by a \code{dirichlet_forest} object.
#' This is especially important on Windows where a persistent PSOCK cluster
#' remains open until explicitly stopped.
#'
#' @param forest A \code{dirichlet_forest} object returned by
#'   \code{\link{DirichletRF}}.
#'
#' @examples
#' \donttest{
#' X <- matrix(rnorm(100 * 4), 100, 4)
#' G <- matrix(rgamma(100 * 3, shape = rep(c(2, 3, 4), each = 100)), 100, 3)
#' Y <- G / rowSums(G)
#'
#' forest <- DirichletRF(X, Y, num.trees = 50, num.cores = 1)
#'
#' pred <- predict(forest, X)
#'
#' # Always clean up at the end, especially on Windows
#' cleanupForest(forest)
#'
#' # After cleanup, the cluster is no longer available.
#' # This would fail: predict(forest, X)
#' }
#' @export
cleanupForest <- function(forest) {
  if (forest$type == "cluster" && !is.null(forest$cluster)) {
    cat("Stopping cluster workers\n")
    parallel::stopCluster(forest$cluster)
    forest$cluster <- NULL
  }
}


#' Predict with a Dirichlet Forest
#'
#' Makes predictions using a fitted \code{dirichlet_forest} object returned
#' by \code{\link{DirichletRF}}.
#'
#' @param object A \code{dirichlet_forest} object.
#' @param newdata A numeric matrix of new covariates (n_new x p).
#' @param ... Currently unused.
#'
#' @return A list with the following elements:
#' \describe{
#'   \item{\code{alpha_predictions}}{Estimated Dirichlet alpha parameters for
#'     each new observation (n_new x k matrix).}
#'   \item{\code{mean_predictions}}{Mean-based compositional predictions
#'     (n_new x k matrix).}
#' }
#'
#' @examples
#' \donttest{
#' n <- 300; p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' alpha <- c(2, 3, 4)
#' G <- matrix(rgamma(n * length(alpha), shape = rep(alpha, each = n)), n, length(alpha))
#' Y <- G / rowSums(G)
#'
#' forest <- DirichletRF(X, Y, num.trees = 50, num.cores = 1)
#'
#' Xtest <- matrix(rnorm(10 * p), 10, p)
#'
#' pred <- predict(forest, Xtest)
#' print(pred$mean_predictions)
#' print(pred$alpha_predictions)
#'
#' # Parameter-based predictions (normalised alphas)
#' param_pred <- pred$alpha_predictions / rowSums(pred$alpha_predictions)
#'
#'
#' cleanupForest(forest)
#' }
#' @export
predict.dirichlet_forest <- function(object, newdata,  ...) {

  distributed_forest <- object
  X_new <- newdata

  # Input validation and coercion
  if (!is.matrix(X_new)) {
    if (is.data.frame(X_new)) {
      X_new <- as.matrix(X_new)
    } else if (is.vector(X_new) || is.numeric(X_new)) {
      X_new <- matrix(X_new, nrow = 1)
      warning("Input was a vector. Converting to 1-row matrix. ",
              "Consider using newdata[i, , drop = FALSE] when subsetting matrices.")
    } else {
      stop("newdata must be a matrix, data frame, or numeric vector")
    }
  }

  if (!is.numeric(X_new)) {
    stop("newdata must contain numeric values")
  }

  n_samples <- nrow(X_new)

  # Hardcoded for this version; will be exposed as arguments in a future release.
  store_samples        <- FALSE
  use_leaf_predictions <- TRUE
  est.method = "mom" # This is also hardcoded for now, but will be an argument later for quantile random forest

  pred_mode <- if (!store_samples) " (pre-computed)" else " (pre-computed leaf values)"
  cat("Prediction mode:", pred_mode, "\n")

  if (distributed_forest$type == "sequential") {
    return(PredictDirichletForest(distributed_forest$forest, X_new,
                                  method = est.method,
                                  use_leaf_predictions = use_leaf_predictions))
  }

  if (distributed_forest$type == "fork") {
    cat("Predicting with", distributed_forest$num.cores, "fork workers\n")

    worker_predictions <- parallel::mclapply(seq_len(distributed_forest$num.cores), function(i) {
      wf <- distributed_forest$worker_forests[[i]]
      if (wf$n_trees > 0) {
        pred_result <- PredictDirichletForest(wf, X_new,
                                              method = est.method,
                                              use_leaf_predictions = use_leaf_predictions)
        if (is.list(pred_result) &&
            !is.null(pred_result$alpha_predictions) &&
            !is.null(pred_result$mean_predictions)) {
          return(pred_result)
        }
      }
      return(NULL)
    }, mc.cores = distributed_forest$num.cores, mc.preschedule = FALSE)

    valid_predictions <- Filter(function(p) {
      !is.null(p) && is.list(p) && !is.null(p$alpha_predictions)
    }, worker_predictions)

  } else if (distributed_forest$type == "cluster") {
    cat("Predicting with", distributed_forest$num.cores, "cluster workers\n")

    cl <- distributed_forest$cluster
    parallel::clusterExport(cl, c("X_new", "est.method", "use_leaf_predictions"),
                            envir = environment())

    worker_predictions <- parallel::clusterApply(cl, seq_len(distributed_forest$num.cores),
      function(worker_id) {
        if (exists("worker_forest", envir = .GlobalEnv)) {
          forest <- get("worker_forest", envir = .GlobalEnv)
          if (forest$n_trees > 0) {
            pred_result <- PredictDirichletForest(forest, X_new,
                                                  method = est.method,
                                                  use_leaf_predictions = use_leaf_predictions)
            if (is.list(pred_result) && !is.null(pred_result$alpha_predictions)) {
              return(pred_result)
            }
          }
        }
        return(NULL)
      })

    valid_predictions <- Filter(function(p) {
      !is.null(p) && is.list(p) && !is.null(p$alpha_predictions)
    }, worker_predictions)
  }

  if (length(valid_predictions) == 0) {
    stop("No valid predictions from workers")
  }

  cat("Combining predictions from", length(valid_predictions), "workers\n")

  first_pred <- valid_predictions[[1]]
  n_classes  <- ncol(first_pred$alpha_predictions)

  combined_alpha <- array(0, dim = c(n_samples, n_classes))
  combined_mean  <- array(0, dim = c(n_samples, n_classes))

  total_trees <- sum(distributed_forest$trees.per.worker[seq_along(valid_predictions)])

  for (i in seq_along(valid_predictions)) {
    pred   <- valid_predictions[[i]]
    weight <- distributed_forest$trees.per.worker[i] / total_trees
    combined_alpha <- combined_alpha + weight * pred$alpha_predictions
    combined_mean  <- combined_mean  + weight * pred$mean_predictions
  }

  return(list(
    alpha_predictions = combined_alpha,
    mean_predictions  = combined_mean
  ))
}