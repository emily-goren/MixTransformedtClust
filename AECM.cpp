#include <RcppArmadillo.h>
#include <RcppGSL.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>

using namespace Rcpp;

// Source this function into an R session using Rcpp::sourceCpp().
  
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::depends(RcppGSL)]]

// Make a symmetric singular matrix non-singular by adding epsilon to the zero eigenvalues.
// [[Rcpp::export]]
arma::mat convert_posdef(arma::mat sigma, double tol = 1e-2) {
  int p = sigma.n_rows;
  arma::mat eigvec(p,p);
  arma::vec eigval(p);
  arma::eig_sym(eigval, eigvec, sigma);
  arma::vec failures(p); failures.zeros();
  arma::mat ans(p,p);
  double counter = 0.0;
  for (int j=0; j<p; j++) {
    bool fails = ((eigval(j) / eigval(p-1)) < tol);
    if (fails) {
      counter += 1.0;
      failures(j) = 1;
    }
  }
  if (counter > 0) {
    for (int j=0; j<p; j++) {
      if (failures(j) == 1) {
        eigval(j) = tol / (counter/p); 
      }
    }
    ans = eigvec * arma::diagmat(eigval) * eigvec.t();
  } else {
    ans = sigma;
  }
  return ans;
}


//' Mahalanobis distance
//' 
//' @description Compute the squared Mahalanobis distance.
//' 
//' @references Mahalanobis, Prasanta Chandra. "On the Generalised Distance in Statistics." 
//' Proceedings of the National Institute of Sciences of India 2 (1936): 49–55.
//' 
//' @param x Numeric. A vector (of length \eqn{p}) or matrix (with \eqn{p} columns).
//' @param mu Numeric. A vector of length \eqn{p}.
//' @param sigma Numeric. A \eqn{p \times p} non-negative definite matrix.
//' @param ischol Logical. Set to \code{TRUE} if \code{sigma} is provided as a Cholesky decomposition.
//' 
//' @return The squared Mahalanobis distance for all rows in \code{x} and the mean vector \code{mu} 
//' with respect to covariance matrix \code{sigma}, defined as \eqn{(x - \mu)' \Sigma^{-1}(x - \mu)}.
//' 
//' @export
//' 
// [[Rcpp::export]]
arma::vec mahalanobis(arma::mat x, arma::vec mu, arma::mat sigma, bool ischol = false) {
  // Check inputs.
  if (mu.n_elem != sigma.n_cols) {
    Rcpp::stop("The supplied mean vector and covariance matrix have incompatible dimensions.");
  }
  if (x.n_cols != sigma.n_cols)  {
    Rcpp::stop("The supplied data matrix and covariance matrix have incompatible dimensions.");
  }
  // Cholesky decomp of covariance matrix -- take lower triangle.
  int n = x.n_rows, p = x.n_cols;
  arma::mat A(p,p);
  if (!ischol) {
    arma::mat Atmp(p,p);
    bool success = arma::chol(Atmp, sigma);
    if (!success) {
      Atmp = arma::chol(convert_posdef(sigma));
    }
    A = arma::trimatl(Atmp.t());
  } else {
    A = arma::trimatl(sigma.t());
  }
  arma::vec D = A.diag();
  // Solve linear system.
  arma::vec ans(n), tmp(p);
  double s;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      s = 0.0;
      for (int k = 0; k < j; k++) {
        s += tmp(k) * A(j, k);
      }
      tmp(j) = ( x(i, j) - mu(j) - s ) / D(j);
    }
    ans.at(i) = sum(square(tmp)); 
  }
  return ans;
}

//' Multivariate t distribution
//' 
//' @description Compute the density of the multivariate t distribution.
//' 
//' @param x Numeric. A vector (of length \eqn{p}) or matrix (with \eqn{p} columns).
//' @param nu Numeric. A postitive scalar.
//' @param mu Numeric. A vector of length \eqn{p} representing the degrees of freedom. 
//' If nonpositive, the multivariate normal density is computed.
//' @param sigma Numeric. A \eqn{p \times p} non-negative definite matrix (or its Cholesky decomposition).
//' @param logans Logical. If \code{TRUE}, the log density is returned.
//' @param ischol Logical. Set to \code{TRUE} of \code{sigma} is provided as a Cholesky decomposition.
//' 
//' @return The multivariate t density for all rows in \code{x} using 
//' degrees of freedom \code{nu}, mean vector \code{mu}, and covariance matrix \code{sigma}.
//' 
//' @export
//' 
// [[Rcpp::export]]
arma::vec dMVT(arma::mat x, arma::vec mu, arma::mat sigma, double nu, bool logans = false, bool ischol = false) {
  // Check inputs.
  if (mu.n_elem != sigma.n_cols) {
    Rcpp::stop("The supplied mean vector and covariance matrix have incompatible dimensions.");
  }
  if (x.n_cols != sigma.n_cols)  {
    Rcpp::stop("The supplied data matrix and covariance matrix have incompatible dimensions.");
  }
  int p = x.n_cols, n = x.n_rows;
  arma::mat A(p,p);
  if (!ischol) {
    bool success = arma::chol(A, sigma);
    if (!success) {
      A = arma::chol(convert_posdef(sigma));
    }
  } else {
    A = sigma;
  }
  arma::vec ans(n);
  arma::vec maha = mahalanobis(x, mu, A, true);
  double logDet = sum(arma::log(A.diag()));
  if (nu <= 0.0) { // MVN
    ans = -0.5*maha - (p/2.0)*std::log(2.0*M_PI) + logDet;
  } else {
    double c = lgamma((nu + p) / 2.0) - lgamma(nu / 2.0) - (p / 2.0) * std::log(nu*M_PI) - logDet;
    for (int i=0; i<n; i++) {
      ans(i) = c - 0.5*(nu+p) * log1p(maha(i)/nu);
    }
  }
  if (!logans) ans = exp(ans);
  return ans;
}

//' Yeo-Johnson transformation
//' 
//' @description Compute the Yeo-Johnson power transformation to improve symmetry.
//' 
//' @references Yeo, In-Kwon, and Richard A. Johnson. "A New Family of Power 
//' Transformations to Improve Normality or Symmetry." Biometrika 87, no. 4 (2000): 954-59.
//' 
//' @param x Numeric. A vector.
//' @param lambda Numeric. A vector of the same length as \code{x}.
//' @param inverse Logical. If \code{TRUE}, returns the inverse transformation.
//' 
//' @return The Yeo-Johnson transformation of \code{x}, or if specified, 
//' its inverse or derivative with respect to \code{lambda}.
//' 
//' @export
//' 
// [[Rcpp::export]]
arma::vec yeo_johnson(arma::vec x, arma::vec lambda, bool inverse = false, double tol = 1e-4) {
  if (x.n_elem != lambda.n_elem)
    Rcpp::stop("Length of 'x' does not match length of 'lambda'.");
  int p = x.n_elem;
  arma::vec ans(p);
  if (inverse) {
    for (int i=0; i<p; i++) {
      if (x(i) < 0) {
        if (std::abs(lambda(i) - 2.0) < tol)
          ans(i) = 1 - exp(-x(i));
        else 
          ans(i) = 1 - pow(1 + x(i)*(lambda(i) - 2), 1 / (2-lambda(i)));
      } else {
        if (std::abs(lambda(i)) < tol)
          ans(i) = exp(x(i)) - 1;
        else
          ans(i) = pow(1 + x(i)*lambda(i), 1/lambda(i)) - 1;
      }
    }
  } else {
    for (int i=0; i<p; i++) {
      if (x[i] < 0) {
        if (lambda(i) == 2)
          ans(i) = -log(1 - x(i));
        else
          ans(i) = -(pow(1 - x(i), 2 - lambda(i)) - 1) / (2 - lambda(i));
      } else {
        if (lambda(i) == 0)
          ans(i) = log(1 + x(i));
        else
          ans(i) = (pow(1 + x(i), lambda(i)) - 1) / (lambda(i));
      }
    }
  }
  return ans;
}

// Derivative of the symmetrizing transformation wrt x.
// Currently only supports Yeo-Johnson or no transformation.
// [[Rcpp::export]]
arma::vec gprime(arma::vec x, arma::vec lambda, std::string trans = "YJ") {
  if (x.n_elem != lambda.n_elem)
    Rcpp::stop("Length of 'x' does not match length of 'lambda'.");
  int p = x.n_elem;
  arma::vec ans(p);
  if (trans == "YJ") {
    for (int i=0; i<p; i++) {
      if (x(i) < 0) {
        if (lambda(i) == 2)
          ans(i) = 1.0 / (-x(i) + 1);
        else 
          ans(i) = pow((-x(i) + 1), 1.0 - lambda(i));
      } else {
        if (lambda(i) == 0)
          ans(i) = 1.0 / (x(i) + 1);
        else
          ans(i) = pow((x(i) + 1), lambda(i) - 1.0);
      }
    }
  } else if (trans == "none") {
    ans.ones();
  } else {
    Rcpp::stop("Specified transformation not recognized.");
  }
  return ans;
}

// Apply symmetrizing transformation to observed p-variate data.
// Currently only supports Yeo-Johnson or no transformation.
// [[Rcpp::export]]
arma::mat f(arma::mat x, arma::vec lambda, std::string trans = "YJ") {
  int p = x.n_cols, n = x.n_rows;
  arma::mat ans(n, p);
  if (trans == "YJ") {
    for (int i=0; i<n; i++) 
      ans.row(i) = arma::trans(yeo_johnson(x.row(i).t(), lambda));
  } else if (trans == "none") {
    ans = x;
  } else {
    Rcpp::stop("Specified transformation not recognized.");
  }
  return ans;
}


// Jacobian associated with symmetrizing transformation.
// [[Rcpp::export]]
arma::vec jac(arma::mat x, arma::vec lambda, std::string trans = "YJ") {
  int p = x.n_cols, n = x.n_rows;
  arma::vec ans(n), tmp(p);
  ans.ones();
  for (int i=0; i<n; i++) {
    tmp = abs(gprime(x.row(i).t(), lambda, trans));
    for (int j=0; j<p; j++) {
      ans(i) = ans(i) * tmp(j);
    }
  }
  return ans;
}

// Density of transformed p-variate data.
// [[Rcpp::export]]
arma::vec h(arma::mat x, arma::vec mu, arma::mat sigma, double nu,
            arma::vec lambda, std::string trans = "YJ") {
  arma::mat y = f(x, lambda, trans);
  arma::vec J = jac(x, lambda, trans);
  arma::vec dens = dMVT(y, mu, sigma, nu, false, false);
  arma::vec ans = dens % J; // % = elementwise vector multiplication.
  return ans;
}


// Make a vector into an array (in arma talk, cube) 
// because Rcpp can't read in an array AFAIK.
// [[Rcpp::export]]
arma::cube to_array(arma::vec sigmas, int K, int p) {
  // Inelegant way to convert vector to cube... yuck yuck yuck....
  NumericVector stmp = as<NumericVector>(wrap(sigmas)); 
  arma::cube Sigmas(stmp.begin(), p, p, K); 
  return Sigmas;
}

// E-step: update Z.
// [[Rcpp::export]]
arma::mat up_Z(arma::mat x, arma::mat mus, arma::vec sigmas, arma::vec nus, 
               arma::mat lambdas, arma::vec pis, std::string trans) {
  int K = mus.n_rows, n = x.n_rows, p = x.n_cols;
  arma::cube Sigmas = to_array(sigmas, K, p);
  arma::mat ans(n, K);
  for (int k=0; k<K; k++) {
    ans.col(k) = pis(k) * h(x, mus.row(k).t(), Sigmas.slice(k), nus(k), lambdas.row(k).t(), trans);
  }
  for (int i=0; i<n; i++) {
    double rowsum = 0.0;
    for (int k=0; k<K; k++) {
      rowsum += ans(i,k);
    }
    ans.row(i) = ans.row(i) / rowsum;
  }
  return ans;
}


// E-step: update W.
// [[Rcpp::export]]
arma::mat up_W(arma::mat x, arma::mat mus, arma::vec sigmas, arma::vec nus, 
               arma::mat lambdas, arma::vec pis, std::string trans) {
  int K = mus.n_rows, n = x.n_rows, p = x.n_cols;
  arma::cube Sigmas = to_array(sigmas, K, p);
  arma::mat ans(n, K);
  for (int k=0; k<K; k++) {
    arma::mat y = f(x, lambdas.row(k).t(), trans);
    arma::vec maha = mahalanobis(y, mus.row(k).t(), Sigmas.slice(k), false);
    for (int i=0; i<n; i++) {
      ans(i,k) = ( nus(k) + p ) / ( nus(k) + maha(i) );
    }
  }
  return ans;
}


// update pis.
// [[Rcpp::export]]
arma::vec up_pi(arma::mat z) {
  int n = z.n_rows, K = z.n_cols;
  arma::vec ans(K); 
  ans.zeros();
  for (int k=0; k<K; k++) {
    for (int i=0; i<n; i++) {
      ans(k) += z(i,k);
    }
    ans(k) = ans(k) / n;
  }
  return ans;
}


// update mus.
// [[Rcpp::export]]
arma::mat up_mu(arma::mat x, arma::mat z, arma::mat w, arma::mat lambdas, std::string trans) {
  int p = x.n_cols, n = x.n_rows, K = z.n_cols;
  arma::mat num(K, p), den(K, p);
  num.zeros(); den.zeros();
  for (int k=0; k<K; k++) {
    arma::mat y = f(x, lambdas.row(k).t(), trans);
    for (int i=0; i<n; i++) {
      num.row(k) += z(i,k) * w(i,k) * y.row(i);
      den.row(k) += z(i,k) * w(i,k);
    }
  }
  arma::mat ans = num / den;
  return ans;
}


// CM=step 3: update Sigmas.
// [[Rcpp::export]]
arma::cube up_Sigma(arma::mat x, arma::mat z, arma::mat w, arma::mat lambdas, 
                    arma::mat mus, std::string trans, bool constr) {
  int p = x.n_cols, n = x.n_rows, K = z.n_cols;
  arma::cube Sigmas(p,p,K);
  for (int k=0; k<K; k++) {
    arma::mat y = f(x, lambdas.row(k).t(), trans);
    arma::mat num(p, p); num.zeros();
    double den = 0.0;
    for (int i=0; i<n; i++) {
      arma::vec u = y.row(i).t() - mus.row(k).t();
      num = num + z(i,k) * w(i,k) * u * u.t();
      den = den + z(i,k) * w(i,k);
    }
    Sigmas.slice(k) = convert_posdef(num / den);
  }
  if (constr) {
    arma::vec pis = up_pi(z);
    arma::mat S(p,p); S.zeros();
    for (int k=0; k<K; k++) {
      S += pis(k) * Sigmas.slice(k);
    }
    S = convert_posdef(S);
    for (int k=0; k<K; k++) {
      Sigmas.slice(k) = S;
    }
  }
  return Sigmas;
}



// update nus -- helper functions for updating cluster-specifc degrees of freedom.
double approx_nu(arma::vec z_k, arma::vec w_k, double nu_k, int p, int n) {
  double zsum = 0, asum = 0; 
  for (int i=0; i<n; i++) {
    asum += z_k[i] * ( log(w_k[i]) - w_k[i] );
    zsum += z_k[i];
  }
  double cc = 1.0 + R::digamma((nu_k + p)/2.0) - log((nu_k + p)/2.0) + (asum/zsum);
  cc = -cc;
  double ans = (-exp(cc)+2.0*(exp(cc))*(exp(R::digamma(nu_k/2.0)) - (nu_k/2.0-0.5)))/(1.0-exp(cc));
  return ans;
}
struct nu_pars {
  NumericVector z_k; 
  NumericVector w_k; 
  double nu_k; 
  int p; 
  int n;
};
double objective_nu(double df, void *params) {
  struct nu_pars *pars;
  pars = (struct nu_pars *)params;
  int n = pars->n;
  int p = pars->p;
  double nu_k = pars->nu_k;
  NumericVector z_k(n);
  z_k = pars->z_k;
  NumericVector w_k(n);
  w_k = pars->w_k;
  double zsum = 0, asum = 0; 
  for (int i=0; i<n; i++) {
    asum += z_k[i] * ( log(w_k[i]) - w_k[i] );
    zsum += z_k[i];
  }
  double ans = 1.0 - R::digamma(df/2.0) + log(df/2.0) + R::digamma((nu_k + p)/2.0) - log((nu_k + p)/2.0) + (asum/zsum);
  return ans;
}
double rootsolver_nu(arma::vec z_k, arma::vec w_k, double nu_k, int p, int n, int iter_max = 1e6, double tol = 1e-3, double min_df = 1e-3, double max_df = 1e4) {
  int status, iter = 0;
  const gsl_root_fsolver_type *T;
  gsl_root_fsolver *solver;
  double ans;
  gsl_function F;
  NumericVector zk = as<NumericVector>(wrap(z_k)); 
  NumericVector wk = as<NumericVector>(wrap(w_k)); 
  struct nu_pars params = {zk, wk, nu_k, p, n};
  F.function = &objective_nu;
  F.params = &params;
  T = gsl_root_fsolver_brent;
  solver = gsl_root_fsolver_alloc(T);
  gsl_root_fsolver_set (solver, &F, min_df, max_df);
  do
    {
      iter++;
      status = gsl_root_fsolver_iterate (solver);
      ans = gsl_root_fsolver_root (solver);
      min_df = gsl_root_fsolver_x_lower (solver);
      max_df = gsl_root_fsolver_x_upper (solver);
      status = gsl_root_test_interval (min_df, max_df, 0, tol);
    }
  while (status == GSL_CONTINUE && iter < iter_max);
  gsl_root_fsolver_free (solver);
  return ans;
}
// update nus -- helper functions for updating constrained (same for all clusters) degrees of freedom.
double approx_nu_constr(arma::mat z, arma::mat w, double nu, int p, int n, int K) {
  double asum = 0;
  for (int k=0; k<K; k++) {
    for (int i=0; i<n; i++) {
      asum += z(i,k) * ( log(w(i,k)) - w(i,k) );
    }
  }
  double cc = 1.0 + R::digamma((nu + p)/2.0) - log((nu + p)/2.0) + (1.0/n)*asum;
  cc = -cc;
  double ans = (-exp(cc)+2.0*(exp(cc))*(exp(R::digamma(nu/2.0)) - (nu/2.0-0.5)))/(1.0-exp(cc));
  return ans;
}
struct nu_constr_pars {
  NumericMatrix z; 
  NumericMatrix w; 
  double nu; 
  int p; 
  int K; 
  int n;
};
double objective_nu_constr(double df, void *params) {
  struct nu_constr_pars *pars;
  pars = (struct nu_constr_pars *)params;
  int n = pars->n;
  int p = pars->p;
  int K = pars->K;
  double nu = pars->nu;
  NumericMatrix z = pars->z;
  NumericMatrix w = pars->w;
  double asum = 0;
  for (int k=0; k<K; k++) {
    for (int i=0; i<n; i++) {
      asum += z(i,k) * ( log(w(i,k)) - w(i,k) );
    }
  }
  double ans = 1.0 - R::digamma(df/2.0) + log(df/2.0) + R::digamma((nu + p)/2.0) - log((nu + p)/2.0) + (1.0/n)*asum;
  return ans;
}
double rootsolver_nu_constr(arma::mat z, arma::mat w, double nu, int p, int n, int K, int iter_max = 1e6, double tol = 1e-3, double min_df = 1e-3, double max_df = 1e4) {
  int status, iter = 0;
  const gsl_root_fsolver_type *T;
  gsl_root_fsolver *solver;
  double ans;
  gsl_function F;
  NumericMatrix Z = as<NumericMatrix>(wrap(z)); 
  NumericMatrix W = as<NumericMatrix>(wrap(w)); 
  struct nu_constr_pars params = {Z, W, nu, p, K, n};
  F.function = &objective_nu_constr;
  F.params = &params;
  T = gsl_root_fsolver_brent;
  solver = gsl_root_fsolver_alloc (T);
  gsl_root_fsolver_set (solver, &F, min_df, max_df);
  do
    {
      iter++;
      status = gsl_root_fsolver_iterate (solver);
      ans = gsl_root_fsolver_root (solver);
      min_df = gsl_root_fsolver_x_lower (solver);
      max_df = gsl_root_fsolver_x_upper (solver);
      status = gsl_root_test_interval (min_df, max_df, 0, tol);
    }
  while (status == GSL_CONTINUE && iter < iter_max);
  gsl_root_fsolver_free (solver);
  return ans;
}

// update nus.
// [[Rcpp::export]]
arma::vec up_nu(arma::mat x, arma::mat z, arma::mat w, arma::vec nus, bool constr = false, bool approx = false) {
  int p = x.n_cols, n = x.n_rows, K = z.n_cols;
  arma::vec ans(K); ans.zeros();
  if (constr) {
    double tmp;
    if (approx) {
      tmp = approx_nu_constr(z, w, nus(0), p, n, K);
    } else {
      tmp = rootsolver_nu_constr(z, w, nus(0), p, n, K);
    }
    if (tmp < 3.0) {
      tmp = 3.0;
    }
    if (tmp > 200) {
      tmp = 200;
    }
    for (int k=0; k<K; k++) {
      ans(k) = tmp;
    }
  } else if (!constr) {
    for (int k=0; k<K; k++) {
      if (approx) {
        ans(k) = approx_nu(z.col(k), w.col(k), nus(k), p, n);
      } else {
        ans(k) = rootsolver_nu(z.col(k), w.col(k), nus(k), p, n);
      }
      if (ans(k) < 3.0) {
        ans(k) = 3.0;
      }
      if (ans(k) > 200) {
        ans(k) = 200;
      }
    }
  } else {
    Rcpp::stop("Degree of freedom constraint option must be boolean.");
  }
  return ans;
}



// update lambdas -- helper functions for updating cluster-specifc skewness parameter.
struct lam_pars {
  arma::mat x; 
  arma::vec w_k; 
  arma::vec z_k; 
  arma::vec mu_k; 
  arma::mat Sigma_k; 
  std::string trans; 
  int p; 
  int n;
};
double objective_lam(const gsl_vector *v, void *params) {
  struct lam_pars *pars;
  pars = (struct lam_pars *)params;
  int p = pars->p;
  int n = pars->n;
  arma::vec lambda(p);
  for (int j=0; j<p; j++) {
    lambda(j) = gsl_vector_get(v, j);
  }
  std::string trans = pars->trans;
  arma::mat Sigma_k(p,p), x(n,p);
  arma::vec w_k(n), z_k(n), mu_k(p);
  z_k = pars->z_k;
  w_k = pars->w_k;
  x = pars->x;
  Sigma_k = pars->Sigma_k;
  mu_k = pars->mu_k;
  arma::mat y = f(x, lambda, trans);
  arma::vec maha = mahalanobis(y, mu_k, Sigma_k, false);
  double ans = 0.0;
  for (int i=0; i<n; i++) {
    arma::vec gp = gprime(x.row(i).t(), lambda, trans);
    double lgp = 0.0;
    for (int j=0; j<p; j++) {
      lgp += log( gp(j) );
    }
    ans += z_k(i) * (-0.5 * w_k(i) * maha(i) + lgp);
  }
  return -ans;
}
arma::vec minimizer_lam(arma::mat x, arma::vec w_k, arma::vec z_k, arma::vec lambda_k, 
                        arma::vec mu_k, arma::mat Sigma_k, std::string trans, int p, int n,
                        int iter_max = 1e4, double step_size = 1e-2, double tol = 1e-2) {
  size_t np = p, iter = 0;
  gsl_multimin_fminimizer *s;
  gsl_vector *ss, *v; 
  gsl_multimin_function obj; 
  int status;
  double size;
  ss = gsl_vector_alloc(np);
  gsl_vector_set_all(ss, step_size); // Step size.
  v = gsl_vector_alloc(np);
  for (int j=0; j<p; j++) {
    gsl_vector_set(v, j, lambda_k(j)); // Start at value from previous iteration.
  }
  struct lam_pars params = {x, w_k, z_k, mu_k, Sigma_k, trans, p, n};
  obj.f = &objective_lam;
  obj.n = np;
  obj.params = &params;
  s = gsl_multimin_fminimizer_alloc (gsl_multimin_fminimizer_nmsimplex2, np);   
  gsl_multimin_fminimizer_set(s, &obj, v, ss); 
  do
    {
      iter++;
      status = gsl_multimin_fminimizer_iterate(s);
      if (status)
	break;
      size = gsl_multimin_fminimizer_size(s);
      status = gsl_multimin_test_size(size, tol);
    }
  while (status == GSL_CONTINUE && iter < iter_max);
  arma::vec ans(p);
  for (int j=0; j<p; j++) {
    ans(j) = gsl_vector_get(s->x, j);
  }
  gsl_vector_free(v);
  gsl_vector_free(ss);
  gsl_multimin_fminimizer_free(s);
  return ans; 
}

// update lambdas -- helper functions for updating constrained (same for all clusters) skewness parameter.
struct lam_constr_pars {
  arma::mat x; 
  arma::mat w; 
  arma::mat z; 
  arma::mat mus; 
  arma::cube Sigmas; 
  std::string trans; 
  int p; 
  int n; 
  int K;
};
double objective_lam_constr(const gsl_vector *v, void *params) {
  struct lam_constr_pars *pars;
  pars = (struct lam_constr_pars *)params;
  int p = pars->p;
  int n = pars->n;
  int K = pars->K;
  arma::vec lambda(p);
  for (int j=0; j<p; j++) {
    lambda(j) = gsl_vector_get(v, j);
  }
  std::string trans = pars->trans;
  arma::cube Sigmas(p,p,K);
  arma::mat x(n,p), w(n,K), z(n,K), mus(K,p);
  z = pars->z;
  w = pars->w;
  x = pars->x;
  Sigmas = pars->Sigmas;
  mus = pars->mus;
  double ans = 0.0;
  arma::mat y = f(x, lambda, trans);
  for (int k=0; k<K; k++) {
    arma::vec maha = mahalanobis(y, mus.row(k).t(), Sigmas.slice(k), false);
    for (int i=0; i<n; i++) {
      arma::vec gp = gprime(x.row(i).t(), lambda, trans);
      double lgp = 0.0;
      for (int j=0; j<p; j++) {
        lgp += log( gp(j) );
      }
      ans += z(i,k) * (-0.5 * w(i,k) * maha(i) + lgp);
    }
  }
  return -ans;
}
arma::vec minimizer_lam_constr(arma::mat x, arma::mat w, arma::mat z, arma::vec lambda, arma::mat mus, 
                               arma::cube Sigmas, std::string trans, int p, int n, int K,
                               int iter_max = 1e4, double step_size = 1e-2, double tol = 1e-2) {
  size_t np = p, iter = 0;
  gsl_multimin_fminimizer *s;
  gsl_vector *ss, *v; 
  gsl_multimin_function obj; 
  int status;
  double size;
  ss = gsl_vector_alloc(np);
  gsl_vector_set_all(ss, step_size); // Step size.
  v = gsl_vector_alloc(np);
  for (int j=0; j<p; j++) {
    gsl_vector_set(v, j, lambda(j)); // Start at value from previous iteration.
  }
  struct lam_constr_pars params = {x, w, z, mus, Sigmas, trans, p, n, K};
  obj.f = &objective_lam_constr;
  obj.n = np;
  obj.params = &params;
  s = gsl_multimin_fminimizer_alloc (gsl_multimin_fminimizer_nmsimplex2, np);   
  gsl_multimin_fminimizer_set(s, &obj, v, ss); 
  do
    {
      iter++;
      status = gsl_multimin_fminimizer_iterate(s);
      if (status)
	break;
      size = gsl_multimin_fminimizer_size(s);
      status = gsl_multimin_test_size(size, tol);
    }
  while (status == GSL_CONTINUE && iter < iter_max);
  arma::vec ans(p);
  for (int j=0; j<p; j++) {
    ans(j) = gsl_vector_get(s->x, j);
  }
  gsl_vector_free(v);
  gsl_vector_free(ss);
  gsl_multimin_fminimizer_free(s);
  return ans; 
}

// update lambdas (use Nelder-Mead Simplex Method).
// [[Rcpp::export]]
arma::mat up_lam(arma::mat x, arma::mat z, arma::mat w, arma::mat mus, arma::mat lambdas, 
                 arma::vec sigmas, std::string trans, bool constr = false) {
  int p = x.n_cols, n = x.n_rows, K = z.n_cols;
  arma::cube Sigmas = to_array(sigmas, K, p);
  arma::mat ans(K,p); ans.ones();
  if (trans != "none") {
    if (constr) {
      arma::vec tmp = minimizer_lam_constr(x, w, z, lambdas.row(0).t(), mus, Sigmas, trans, p, n, K);
      for (int k=0; k<K; k++) {
        ans.row(k) = tmp.t();
      }
    } else if (!constr) {
      for (int k=0; k<K; k++) {
	arma::vec tmp = minimizer_lam(x, w.col(k), z.col(k), lambdas.row(k).t(), mus.row(k).t(), Sigmas.slice(k), trans, p, n);
        ans.row(k) = tmp.t();
      }
    } else {
      Rcpp::stop("Transformation parameter constraint option must be boolean.");
    }
  }
  return ans;
}

