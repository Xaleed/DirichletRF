#' Build a Dirichlet Random Forest for Compositional Responses
#'
#' Build a Dirichlet random forest for compositional responses. In
#' compositional data analysis (CoDA), parts reside in the simplex, and this
#' random forest ensures model output abide by CoDA principles. The
#' implementation uses OpenMP for parallel tree building.
#'
#' @param X A numeric (n x p) matrix of covariates. Note that the current
#'   version only allows numeric covariates. Users may use one-hot encoding
#'   to possibly include categorical covariates.
#' @param Y A numeric (n x k) matrix of compositional responses. Each row
#'   should sum to 1. That is, data should already be normalised if needed.
#' @param num.trees Number of trees grown in the forest. Default is 100.
#' @param max.depth Maximum depth of trees. Default is 10.
#' @param min.node.size Minimum size of observations in each tree leaf.
#'   Default is 5. Note that nodes with sizes smaller than
#'   \code{min.node.size} can occur.
#' @param mtry Number of covariates randomly selected as candidates at each
#'   split. Default is \code{sqrt(p)}, indicated by \code{-1}.
#' @param seed The seed of the C++ random number generator.
#' @param est.method Parameter estimation method for the Dirichlet
#'   distribution when splitting is done. Users may either use maximum
#'   likelihood (\code{"mle"}) or method of moments (\code{"mom"}).
#'   Default is \code{"mom"}.
#' @param num.cores Number of OpenMP threads used for parallel tree building.
#'   The default is \code{-1} which uses all the cores on the system minus 1.
#'   Users may also specify \code{1} which means that the forest will be
#'   built sequentially.
#' @param replace Logical. If \code{TRUE}, each tree is grown on a bootstrap
#'   sample drawn with replacement. If \code{FALSE} (default), each tree is
#'   grown on a subsample drawn without replacement. When \code{replace = FALSE}
#'   and \code{sample.fraction = 1} (the default), every tree sees all \code{n}
#'   observations and tree diversity comes entirely from random feature
#'   subsetting controlled by \code{mtry}. When \code{replace = FALSE} and
#'   \code{sample.fraction < 1}, each tree sees a different random subset of
#'   the data, enabling out-of-bag estimation.
#' @param sample.fraction Numeric. Fraction of observations used to grow each
#'   tree, as a proportion of \code{n}. Default is \code{1.0}. When
#'   \code{replace = FALSE}, must be in \code{(0, 1]}; values greater than 1
#'   are not allowed since you cannot draw more unique observations than
#'   available. When \code{replace = TRUE}, values greater than 1 are allowed
#'   (e.g. \code{1.5} draws \code{1.5n} bootstrap observations), though
#'   values in \code{(0, 1]} are most common. A warning is issued when
#'   \code{sample.fraction < 0.1} regardless of \code{replace}, as trees
#'   grown on very few observations tend to be unreliable.
#' @param compute.oob Logical. If \code{TRUE}, computes out-of-bag (OOB)
#'   predictions after the forest is built, using only trees for which each
#'   observation was not in training. Both the OOB prediction matrix and a
#'   scalar MSE are returned via \code{$oob$predictions} and \code{$oob$mse}.
#'   Not available when \code{replace = FALSE} and \code{sample.fraction = 1}
#'   since no held-out observations exist. Default is \code{FALSE}.
#' 
#' @details
#' \strong{Out-of-Bag (OOB) Predictions}
#'
#' When \code{compute.oob = TRUE}, each observation is predicted by averaging
#' over only the trees for which it was out-of-bag. This requires
#' \code{replace = TRUE} or \code{replace = FALSE} with
#' \code{sample.fraction < 1}. The reported \code{$oob$mse} is the MSE
#' between OOB predictions and true responses, averaged over components and
#' OOB observations. Note that MSE is not universally accepted for
#' compositional data since it ignores the simplex geometry — the Aitchison
#' distance, which operates in log-ratio space, is an alternative. The full
#' OOB prediction matrix \code{$oob$predictions} (n x k, with \code{NA} for
#' observations never out-of-bag) is returned so users can apply any
#' alternative error measure directly.
#'
#' @return A list of class \code{dirichlet_forest} which contains the
#'   following elements:
#' \describe{
#'   \item{\code{type}}{Parallelisation type used: \code{"openmp"} or
#'     \code{"sequential"}.}
#'   \item{\code{num.cores}}{Number of cores used.}
#'   \item{\code{num.trees}}{Total number of trees in the forest.}
#'   \item{\code{replace}}{Logical indicating whether bootstrap sampling
#'     was used.}
#'   \item{\code{sample.fraction}}{The fraction of observations used per
#'     tree.}
#'   \item{\code{compute.oob}}{Logical indicating whether OOB prediction was
#'     computed.}
#'   \item{\code{Y_train}}{The training compositional response matrix.}
#'   \item{\code{fitted}}{A list of fitted values on the training data:
#'     \describe{
#'       \item{\code{alpha_hat}}{Estimated Dirichlet alpha parameters
#'         (n x k matrix).}
#'       \item{\code{mean_based}}{Mean-based fitted values (n x k matrix),
#'         derived from sample means at each leaf.}
#'       \item{\code{param_based}}{Parameter-based fitted values (n x k
#'         matrix), obtained by normalising \code{alpha_hat} so rows sum
#'         to 1.}
#'     }
#'   }
#'   \item{\code{residuals}}{A list of residuals (Y - fitted values):
#'     \describe{
#'       \item{\code{mean_based}}{Residuals from mean-based predictions.}
#'       \item{\code{param_based}}{Residuals from parameter-based
#'         predictions.}
#'     }
#'   }
#'   \item{\code{importance}}{A list of feature importance measures:
#'     \describe{
#'       \item{\code{gain}}{Raw total likelihood gain per feature, summed
#'         over all trees and all splits where the feature was selected.}
#'       \item{\code{gain_normalised}}{Gain divided by total gain across
#'         all features, summing to 1. Recommended for interpretation and
#'         comparison across forests.}
#'       \item{\code{count}}{Number of times each feature was selected as
#'         the best split variable across all trees and all internal
#'         nodes.}
#'     }
#'   }
#'   \item{\code{oob}}{A list of OOB results. Both elements are \code{NA}
#'     when \code{compute.oob = FALSE}:
#'     \describe{
#'       \item{\code{mse}}{Scalar OOB mean squared error, averaged over
#'         all components and all observations that appeared OOB at least
#'         once.}
#'       \item{\code{predictions}}{An (n x k) matrix of OOB predictions.
#'         Rows corresponding to observations that never appeared OOB are
#'         \code{NA}.}
#'     }
#'   }
#' }
#'
#' @examples
#' # ── Minimal example (auto-tested) ─────────────────────────────────────────
#' set.seed(42)
#' n <- 50; p <- 2
#' X <- matrix(rnorm(n * p), n, p)
#' colnames(X) <- paste0("X", 1:p)
#' G <- matrix(rgamma(n * 3, shape = rep(c(2, 3, 4), each = n)), n, 3)
#' Y <- G / rowSums(G)
#'
#' # Default: no bootstrap, no OOB, fastest configuration
#' forest <- DirichletRF(X, Y, num.trees = 5, num.cores = 1)
#' print(forest)
#'
#' # Feature importance
#' importance(forest)
#'
#' # Prediction on new data
#' Xtest <- matrix(rnorm(5 * p), 5, p)
#' colnames(Xtest) <- paste0("X", 1:p)
#' pred  <- predict(forest, Xtest)
#' pred$mean_predictions
#'
#' \donttest{
#' # ── Larger example with informative and noise covariates ───────────────────
#' set.seed(42)
#' n <- 200; p <- 6
#' X <- matrix(rnorm(n * p), n, p)
#' colnames(X) <- paste0("X", 1:p)
#'
#' # X1 and X2 are informative, X3-X6 are noise
#' alpha_mat <- cbind(
#'   2 + 3 * (X[, 1] > 0),
#'   3 + 3 * (X[, 2] > 0),
#'   rep(4, n)
#' )
#' G <- matrix(rgamma(n * 3, shape = as.vector(t(alpha_mat))), n, 3,
#'             byrow = TRUE)
#' Y <- G / rowSums(G)
#'
#' # Default: no bootstrap, no OOB
#' forest <- DirichletRF(X, Y, num.trees = 100, num.cores = 1)
#'
#' # Feature importance — X1 and X2 should dominate
#' importance(forest)
#'
#' # Fitted values and residuals
#' head(forest$fitted$mean_based)
#' head(forest$residuals$mean_based)
#'
#' # ── Bootstrap with OOB ───────────────────────────────────────────────
#' forest_oob <- DirichletRF(X, Y, num.trees = 100, num.cores = 1,
#'                            replace = TRUE, sample.fraction = 1.0,
#'                            compute.oob = TRUE)
#' forest_oob$oob$mse
#' head(forest_oob$oob$predictions)
#'
#' # ── Subsampling without replacement with OOB ───────────────────────────────
#' forest_sub <- DirichletRF(X, Y, num.trees = 100, num.cores = 1,
#'                            replace = FALSE, sample.fraction = 0.632,
#'                            compute.oob = TRUE)
#' forest_sub$oob$mse
#'
#' # ── Prediction ────────────────────────────────────────────────────────────
#' Xtest <- matrix(rnorm(10 * p), 10, p)
#' colnames(Xtest) <- paste0("X", 1:p)
#' pred <- predict(forest, Xtest)
#' head(pred$mean_predictions)
#' param_pred <- pred$alpha_predictions / rowSums(pred$alpha_predictions)
#' }
#'
#' @references
#' Masoumifard, K., van der Westhuizen, S., & Gardner-Lubbe, S. (2026).
#' Dirichlet random forest for predicting compositional data.
#' In A. Bekker, P. Nagar, J. Ferreira, B. Erasmus, & A. Ramoelo (Eds.),
#' Environmental Modelling with Contemporary Statistics: Learning,
#' Directionality, and Space-Time Dynamics.
#' Chapman & Hall/CRC. ISBN: 9781032903910.
#'
#' @seealso
#' \code{\link{predict.dirichlet_forest}} for making predictions on new data.
#' \code{\link{importance.dirichlet_forest}} for a summary of feature importance.
#' @importFrom stats predict
#' @export
DirichletRF <- function(X, Y, num.trees = 100, max.depth = 10,
                        min.node.size = 5, mtry = -1, seed = 123,
                        est.method = "mom", num.cores = -1,
                        replace = FALSE, sample.fraction = 1.0,
                        compute.oob = FALSE) {

  # Hardcoded for this version; will be exposed in a future release.
  store_samples <- FALSE

  # Input validation
  if (!is.matrix(X) || !is.matrix(Y)) stop("X and Y must be matrices")
  if (nrow(X) != nrow(Y)) stop("X and Y must have the same number of rows")

  # applies to both
  if (sample.fraction <= 0)
    stop("sample.fraction must be positive")

  # only meaningful for replace = FALSE
  if (!replace && sample.fraction > 1)
    stop("sample.fraction must be in (0, 1] when replace = FALSE")

  if (sample.fraction < 0.1)
    warning("sample.fraction < 0.1: trees will be grown on very few observations")

  # OOB requires held-out observations
  if (compute.oob && !replace && sample.fraction == 1.0)
    stop("OOB is not available when replace = FALSE and sample.fraction = 1")

  # Resolve num.cores
  if (num.cores == -1) {
    num.cores <- max(1L, parallel::detectCores() - 1L)
  }
  num.cores <- max(1L, as.integer(num.cores))

  par_type <- if (num.cores == 1L) "sequential" else "openmp"
  message("Building ", par_type, " forest with ", num.cores,
          " thread(s) for ", num.trees, " trees")

  # All parallelism handled inside C++ via OpenMP
  forest <- DirichletForest(X, Y,
                            B               = num.trees,
                            d_max           = max.depth,
                            n_min           = min.node.size,
                            m_try           = mtry,
                            seed            = seed,
                            method          = est.method,
                            store_samples   = store_samples,
                            num_cores       = num.cores,
                            replace         = replace,
                            sample_fraction = sample.fraction,
                            compute_oob     = compute.oob)

  # Extract feature importance from the C++ forest object
  imp_gain  <- forest$importance_gain   # sum of likelihood gains per feature
  imp_count <- forest$importance_count  # number of times selected for a split

  # Normalise gain importance to sum to 1 (makes forests comparable)
  imp_gain_norm <- if (sum(imp_gain) > 0) imp_gain / sum(imp_gain)
                   else rep(0, length(imp_gain))

  # Attach feature names if X has column names
  feat_names <- if (!is.null(colnames(X))) colnames(X)
                else paste0("X", seq_len(ncol(X)))
  names(imp_gain)      <- feat_names
  names(imp_gain_norm) <- feat_names
  names(imp_count)     <- feat_names

  result <- list(
    type            = par_type,
    forest          = forest,
    num.cores       = num.cores,
    num.trees       = num.trees,
    replace         = replace,
    sample.fraction = sample.fraction,
    compute.oob     = compute.oob,
    Y_train         = Y,
    importance = list(
      gain            = imp_gain,
      gain_normalised = imp_gain_norm,
      count           = imp_count
    ),
    oob = list(
      mse         = forest$oob_mse,
      predictions = forest$oob_predictions
    )
  )
  class(result) <- c("dirichlet_forest", "list")

  message("Computing fitted values and residuals...")
  fitted_preds <- predict(result, X)
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


#' Custom Print Method for dirichlet_forest Objects
#'
#' Suppresses the display of large data matrices (Y_train, fitted,
#' residuals) when the object is printed, while keeping them accessible
#' via \code{$}.
#'
#' @param x A \code{dirichlet_forest} object.
#' @param ... Further arguments passed to or from other methods.
#'
#' @return Invisibly returns \code{x}, the \code{dirichlet_forest} object
#'   unchanged. Called primarily for its side effect of printing a summary
#'   of the model to the console.
#' @export
print.dirichlet_forest <- function(x, ...) {

  N <- nrow(x$Y_train)
  K <- ncol(x$Y_train)
  train_info <- if (!is.null(N) && !is.null(K))
    paste0(N, " observations (n) x ", K, " components (k)")
  else
    "unknown"

  oob_str <- if (is.na(x$oob$mse)) "not computed" else round(x$oob$mse, 6)
  samp_str <- paste0(if (x$replace) "with replacement" else "without replacement",
                     ", fraction = ", x$sample.fraction)

  cat(
    "============================================\n",
    "Dirichlet Forest Model\n",
    "============================================\n",
    " Type:          ", x$type, "\n",
    " Total Trees:   ", x$num.trees, "\n",
    " Cores Used:    ", x$num.cores, "\n",
    " Sampling:      ", samp_str, "\n",
    " Training Data: ", train_info, "\n",
    " OOB MSE:       ", oob_str, "\n",
    "--------------------------------------------\n",
    " Note: Large data structures (fitted values,\n",
    "       residuals) are suppressed.\n",
    "\n Access via:\n",
    "   $Y_train\n",
    "   $fitted$alpha_hat\n",
    "   $fitted$mean_based\n",
    "   $fitted$param_based\n",
    "   $residuals$mean_based\n",
    "   $residuals$param_based\n",
    "   $importance$gain\n",
    "   $importance$gain_normalised\n",
    "   $importance$count\n",
    "   $oob$mse\n",
    "   $oob$predictions\n",
    " Use importance(forest) for a summary table.\n",
    "============================================\n",
    sep = ""
  )

  invisible(x)
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
#'   \item{\code{alpha_predictions}}{Estimated Dirichlet alpha parameters
#'     for each new observation (n_new x k matrix).}
#'   \item{\code{mean_predictions}}{Mean-based compositional predictions
#'     (n_new x k matrix).}
#' }
#'
#' @examples
#' # Small toy example (auto-tested)
#' set.seed(42)
#' n <- 50; p <- 2
#' X <- matrix(rnorm(n * p), n, p)
#' G <- matrix(rgamma(n * 3, shape = rep(c(2, 3, 4), each = n)), n, 3)
#' Y <- G / rowSums(G)
#' forest  <- DirichletRF(X, Y, num.trees = 5, num.cores = 1)
#' Xtest   <- matrix(rnorm(5 * p), 5, p)
#' pred    <- predict(forest, Xtest)
#' pred$mean_predictions
#'
#' \donttest{
#' n <- 500; p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' alpha <- c(2, 3, 4)
#' G <- matrix(rgamma(n * length(alpha), shape = rep(alpha, each = n)),
#'             n, length(alpha))
#' Y <- G / rowSums(G)
#' forest <- DirichletRF(X, Y, num.trees = 50, num.cores = 1)
#' Xtest  <- matrix(rnorm(10 * p), 10, p)
#' pred   <- predict(forest, Xtest)
#' param_pred  <- pred$alpha_predictions / rowSums(pred$alpha_predictions)
#' single_pred <- predict(forest, Xtest[1, , drop = FALSE])
#' }
#' @export
predict.dirichlet_forest <- function(object, newdata, ...) {

  # Hardcoded for this version; will be exposed in a future release.
  est.method           <- "mom"
  use_leaf_predictions <- TRUE

  X_new <- newdata

  # Input validation and coercion
  if (!is.matrix(X_new)) {
    if (is.data.frame(X_new)) {
      X_new <- as.matrix(X_new)
    } else if (is.vector(X_new) || is.numeric(X_new)) {
      X_new <- matrix(X_new, nrow = 1)
      warning("Input was a vector. Converting to 1-row matrix. ",
              "Consider using newdata[i, , drop = FALSE] when ",
              "subsetting matrices.")
    } else {
      stop("newdata must be a matrix, data frame, or numeric vector")
    }
  }

  if (!is.numeric(X_new)) stop("newdata must contain numeric values")

  return(PredictDirichletForest(object$forest, X_new,
                                method               = est.method,
                                use_leaf_predictions = use_leaf_predictions))
}


#' Feature Importance for a Dirichlet Forest
#'
#' Returns a data frame summarising feature importance from a fitted
#' \code{dirichlet_forest} object. Two measures are provided:
#' \describe{
#'   \item{\code{gain}}{Total likelihood gain accumulated across all splits
#'     where this feature was selected (raw, summed over all trees).}
#'   \item{\code{gain_normalised}}{Same as \code{gain} but normalised to
#'     sum to 1 across all features, making values comparable across
#'     forests of different sizes.}
#'   \item{\code{count}}{Number of times the feature was chosen as the
#'     best split variable across all trees and all internal nodes.}
#' }
#' The data frame is sorted by \code{gain_normalised} in descending order.
#'
#' @param object A \code{dirichlet_forest} object returned by
#'   \code{\link{DirichletRF}}.
#' @param ... Currently unused.
#'
#' @return A data frame with columns \code{feature}, \code{gain},
#'   \code{gain_normalised}, and \code{count}, sorted by
#'   \code{gain_normalised} descending.
#'
#' @examples
#' set.seed(42)
#' n <- 50; p <- 4
#' X <- matrix(rnorm(n * p), n, p)
#' colnames(X) <- paste0("X", 1:p)
#' G <- matrix(rgamma(n * 3, shape = rep(c(2, 3, 4), each = n)), n, 3)
#' Y <- G / rowSums(G)
#' forest <- DirichletRF(X, Y, num.trees = 10, num.cores = 1)
#' importance(forest)
#'
#' @export
importance <- function(object, ...) UseMethod("importance")
#' @export
importance.dirichlet_forest <- function(object, ...) {
  imp <- object$importance
  if (is.null(imp))
    stop("No importance information found. Please refit with the current version.")

  df <- data.frame(
    feature         = names(imp$gain),
    gain            = unname(imp$gain),
    gain_normalised = unname(imp$gain_normalised),
    count           = unname(imp$count),
    stringsAsFactors = FALSE
  )

  df <- df[order(df$gain_normalised, decreasing = TRUE), ]
  rownames(df) <- NULL
  df
}