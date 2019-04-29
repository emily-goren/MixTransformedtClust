

# Wrapper function.
tmix.YJ <- function(x, initialize = c("emEM", "randEM"), Ks = 3:8, trans = "YJ", model = "UUU", IDs = NULL,
                    max.iter = 1e3, tol = 1e-2, approx.df = TRUE, scale = TRUE,
                    nstarts = 100, nruns = 3) {
  startclock <- proc.time()[1]
  if (scale) {
    X <- scale(x)
  } else {
    X <- as.matrix(x)
  }
  n <- nrow(X); p <- ncol(X)
  if (initialize == "randEM") {
    message(paste("Determining initial values using randEM with", nstarts, "random starts."))
    starts <- lapply(Ks, function(nclusters) get.starts(X, nclusters, trans, model, 
                                                        nruns = 1, nstarts = nstarts, nbest = nruns, Z = NULL))
  } else if (initialize == "emEM") {
        message(paste("Determining initial values using emEM with", nstarts, "random starts."))
        starts <- lapply(Ks, function(nclusters) get.starts(X, nclusters, trans, model, 
                                                            nruns = 4, nstarts = nstarts, nbest = nruns, Z = NULL))
  } else if (initialize == "IDs") {
    starts <- lapply(Ks, function(nclusters) get.starts(X, nclusters, trans, model, Z = IDs))
  } else {
    stop("Initialization method not recognized.")
  }
  message(paste("Running AECM for the best", nruns, "random starts."))
  Kfits <- lapply(starts, function(Kst) {
    runs <- lapply(Kst, function(st) try(skewtmix_cpp(X, st, trans, model, max.iter, tol, approx.df)))
    LLs <- sapply(runs, function(f) max(f$loglik, na.rm = TRUE))
    stbest <- which.max(LLs)
    if (length(stbest) == 0) stbest <- 1
    out <- runs[[stbest]]
    return(out)
  })
  message("Finishing up, returning best (according to largest loglik) run that converged.")
  best <- sapply(Kfits, function(out) ifelse(out$converged == TRUE, out$bic, NA))
  index <- which.max(best)
  if (length(index) == 0) index <- 1
  fit <- Kfits[[index]]
  if (!is.null(IDs)) fit <- c(fit, adjustedRand = mclust::adjustedRandIndex(IDs, fit$classification))
  stopclock <- proc.time()[1]
  fit <- c(fit, runtime = stopclock - startclock)
  return(fit)
}


# Perform one EM iteration (of all AECM steps).
EM_iter <- function(oldpars, x, trans, sigma.constr, df.constr, lam.constr, approx.df = FALSE) {
  pis <- oldpars$pi
  nus <- oldpars$nu
  lambdas <- oldpars$lambda
  mus <- oldpars$mu
  Sigmas <- oldpars$Sigma
  # 1st cycle: (pi, mu, Sigma)
  z <- up_Z(x, mus, as.numeric(Sigmas), nus, lambdas, pis, trans)
  w <- up_W(x, mus, as.numeric(Sigmas), nus, lambdas, pis, trans)
  pis <- as.numeric(up_pi(z))
  mus <- up_mu(x, z, w, lambdas, trans)
  Sigmas <- up_Sigma(x, z, w, lambdas, mus, trans, sigma.constr)
  # 2ns cycle: nu
  z <- up_Z(x, mus, as.numeric(Sigmas), nus, lambdas, pis, trans)
  w <- up_W(x, mus, as.numeric(Sigmas), nus, lambdas, pis, trans)
  nus <- as.numeric(up_nu(x, z, w, nus, df.constr, approx.df))
  # 3rd cycle: lambda
  z <- up_Z(x, mus, as.numeric(Sigmas), nus, lambdas, pis, trans)
  w <- up_W(x, mus, as.numeric(Sigmas), nus, lambdas, pis, trans)
  lambdas <- up_lam(x, z, w, mus, lambdas, as.numeric(Sigmas), trans, lam.constr)
  newpars <- list(pis, nus, lambdas, mus, Sigmas)
  names(newpars) <- c("pi", "nu", "lambda", "mu", "Sigma")
  return(newpars)
}

# Main function.
skewtmix_cpp <- function(X,
                         initial.values,
                         trans = c("YJ", "none"), # YJ = yeo johnson
                         model = "CCC", # constraints on df, lambda, Sigma.
                         max.iter = 1000,
                         tol = 1e-3,
                         approx.df = FALSE) {
  old <- initial.values
  parnames <- c("pi", "nu", "lambda", "mu", "Sigma")
  names(old) <- parnames
  K <- length(old$pi)
  n <- nrow(X); p <- ncol(X)
  df.constr <- substr(model, 1, 1) == "C"
  lam.constr <- substr(model, 2, 2) == "C"
  sigma.constr <- substr(model, 3, 3) == "C"
  del <- 1e6 # Initialize change in loglik holder.
  iter <- 0
  LLs <- rep(NA, max.iter) # Store loglikelihood at each iteration.
  converged <- FALSE
  while (del > tol) {
    iter <- iter + 1
    if (iter > max.iter) {
      break
    }
    new <- EM_iter(old, X, trans, sigma.constr, df.constr, lam.constr, approx.df)
    names(new) <- parnames
    oldLL <- sapply(1:K, function(k) {old$pi[k] * h(X, old$mu[k,], old$Sigma[,,k], old$nu[k], old$lambda[k,], trans)})
    newLL <- sapply(1:K, function(k) {new$pi[k] * h(X, new$mu[k,], new$Sigma[,,k], new$nu[k], new$lambda[k,], trans)})
    del <- sum(log(rowSums(newLL))) - sum(log(rowSums(oldLL))) # Change in loglikelihoods.
    LLs[iter] <- sum(log(rowSums(newLL)))
    old <- new
  }
  if (del < tol) converged <- TRUE
  Zs <- up_Z(X, new$mu, as.numeric(new$Sigma), new$nu, new$lambda, new$pi, trans)
  classification <- apply(Zs, 1, which.max)
  sigp <- p*(p+1)/2
  npar <- (K-1) + K*p * ifelse(df.constr, 1, K) + ifelse(sigma.constr, sigp, K*sigp) + ifelse(lam.constr, K*p, p)*(trans != "none")
  BIC <- 2*max(LLs, na.rm = TRUE) - npar*log(n)
  res <- list(new, iter, Zs, classification, LLs[1:iter], BIC, converged)
  names(res) <- c("estimates", "iterations", "Zs", "classification", "loglik", "bic", "converged")
  return(res)
}


# Function to obtain starting values.
get.starts <- function(X, nclusters, trans = "YJ", model = "UUU", 
                       nruns = 2, nstarts = 100, nbest = 4, Z = NULL) {
  parnames <- c("pi", "nu", "lambda", "mu", "Sigma")
  p <- ncol(X)
  n <- nrow(X)
  k <- nclusters
  df.constr <- substr(model, 1, 1) == "C"
  lam.constr <- substr(model, 2, 2) == "C"
  sigma.constr <- substr(model, 3, 3) == "C"
  starts <- lapply(1:nstarts, function(ns) {
    # Draw transformation parameter, degrees of freedom, proportions uniformly.
    if (lam.constr) nel <- p else nel <- k*p
    lambdas <- matrix(runif(min = 0, max = 2, n = nel), ncol = p, nrow = k, byrow = TRUE)
    if (df.constr) nus <- rep(runif(1, 5, 25), k) else nus <- runif(k, 5, 25)
    pis <- runif(k, 0.2, 0.8); pis <- pis / sum(pis)
    # Initial IDs.
    if (is.null(Z)) Z <- sample(c(1:k), n, replace = TRUE, prob = pis)
    # Transform data using IDs.
    Y <- t(sapply(1:n, function(nn) yeo_johnson(X[nn,], lambda = lambdas[Z[nn],])))
    Y[!is.finite(Y)] <- 0
    # Use updated IDs to set remaining parameters.
    tmp <- lapply(1:k, function(kk) {
      ans <- tryCatch(cov.wt(Y[Z == kk,]), error = function(e) cov.wt(Y))
      })
    mus <- do.call("rbind", lapply(tmp, function(o) o$center))
    Sigmas <- array(unlist(lapply(tmp, function(o) o$cov)), dim = c(p,p,k))
    for (kk in 1:k) Sigmas[,,kk] <- convert_posdef(Sigmas[,,kk])
    if (sigma.constr) {
      S <- lapply(1:k, function(kk) pis[kk]*Sigmas[,,k])
      S <- convert_posdef(Reduce("+", S))
      Sigmas <- array(unlist(S), dim = c(p,p,k))
    }
    out <- list(pis, nus, lambdas, mus, Sigmas)
    names(out) <- parnames
    return(out)
  })
  shortEM <- lapply(starts, function(st) {
    try(skewtmix_cpp(X, st, trans, model, nruns, 1e-3, TRUE))
    })
  LLs <- sapply(shortEM, function(f) max(f$loglik, na.rm = TRUE))
  best <- order(LLs, decreasing = TRUE)[1:nbest]
  return(starts[best])
}
