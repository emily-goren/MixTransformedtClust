# Example code to run clustering algorithm (multivariate transformed t mixtures)
# on iris dataset

library(Rcpp); library(RcppArmadillo); library(RcppGSL)
library(foreach); library(doParallel)

source('./AECM.R')
Rcpp::sourceCpp('./AECM.cpp')

d <- iris[,1:4]
dim(d)


devtools::session_info()

registerDoParallel(detectCores())

set.seed(20180823)


# Function to run clustering algorithm for a given number of clusters, k
fun.K <- function(k, data) {
    cat('Beginning k = ', k, '\n')       
    ans_k <- tryCatch(
        tmix.YJ(
            x = data, initialize = 'emEM', 
            Ks = k, trans = 'YJ', model = 'UUU', IDs = NULL,
            max.iter = 1e3, tol = 1e-3, approx.df = TRUE, scale = FALSE,
            nstarts = k*prod(dim(d)), nruns = 5),
        error = function(e) list(bic = NA))
    save(ans_k, file = paste0('./results-k=', k, '.rda'))
    ans_k
}

# Run algorithm for multiple test k (in parallel)
testK <- 1:8
res <- foreach(i = testK, .combine = c) %dopar% {
    list(soln = fun.K(k = i, data = d))
}


# Plot and show results to determine best k via BIC
library(ggplot2)
theme_set(theme_bw(base_size = 16))
bics.marg <- sapply(res, function(x) x$bic)
qplot(x = testK, y = bics.marg, geom = 'point', xlab = 'Number of Clusters', ylab = 'BIC')
cbind(testK, bics.marg)

# Estimates for best k (according to BIC)
K.hat <- testK[which.max(bics.marg)]
ests <- res[[which.max(bics.marg)]]$estimates
ests