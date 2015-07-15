#include <RcppArmadillo.h>
#include <omp.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

// Presumably no need to compile with these flags to make this parallel anymore
// Sys.setenv("PKG_CXXFLAGS" = "-fopenmp")
// Sys.setenv("PKG_LIBS" = "-fopenmp")
// TODO: Add stopping condition by tolerance
// TODO: Add early stopping
// TODO: Use sparse arma::matrices wherever possible


// [[Rcpp::export]]
void updateImplicitX(arma::mat & X, const arma::mat & Y, const arma::mat & P, const arma::mat & C, 
  double lambda, int cores) {
  int num_users = C.n_rows;
  int num_prods = C.n_cols;
  int num_factors = Y.n_cols; // or X.n_cols

  Rprintf(".");
  arma::mat YTY = Y.t() * Y;
  arma::mat fact_eye = eye(num_prods, num_prods);
  arma::mat lambda_eye = lambda * eye(num_factors, num_factors);

#pragma omp parallel for num_threads(cores)
  for (int u = 0; u < C.n_rows; u++) {
    arma::mat Cu = diagmat(C.row(u));
    arma::mat YTCuIY = Y.t() * (Cu) * Y;
    arma::mat YTCupu = Y.t() * (Cu + fact_eye) * P.row(u).t();
    arma::mat WuT = YTY + YTCuIY + lambda_eye;
    arma::mat xu = solve(WuT, YTCupu);

    // Update gradient -- maybe a slow operation in parallel?
    X.row(u) = xu.t();
  }
}

// [[Rcpp::export]]
void updateImplicitY(const arma::mat & X, arma::mat & Y, const arma::mat & P, const arma::mat & C, 
  double lambda, int cores) {
  int num_users = C.n_rows;
  int num_prods = C.n_cols;
  int num_factors = Y.n_cols; // or X.n_cols

  Rprintf(".");
  arma::mat XTX = X.t() * X;
  arma::mat fact_eye = eye(num_users, num_users);
  arma::mat lambda_eye = lambda * eye(num_factors, num_factors);

#pragma omp parallel for num_threads(cores)
  for (int i = 0; i < C.n_cols; i++) {
    arma::mat Ci = diagmat(C.col(i));
    arma::mat YTCiIY = X.t() * (Ci) * X;
    arma::mat YTCipi = X.t() * (Ci + fact_eye) * P.col(i);
    arma::mat yu = solve(XTX + YTCiIY + lambda_eye, YTCipi);

    // Update gradient
    Y.row(i) = yu.t();
  }
}

// [[Rcpp::export]]
double implicitCost(const arma::mat & X, const arma::mat & Y, const arma::mat & P, const arma::mat & C, double lambda,
  int cores) {
  double delta = 0.0;
#pragma omp parallel for num_threads(cores)
  for (int u = 0; u < C.n_rows; u++) {
    delta += accu(dot(C.row(u), square(P.row(u) - X.row(u) * Y.t())));
  }
  return (delta + 
    lambda * (pow(accu(X),2) + pow(accu(Y),2)));
}

// [[Rcpp::export]]
List implicit(const arma::mat & init_X, const arma::mat & init_Y, const arma::mat & P, const arma::mat & C,
        double lambda, int batches,
        double epsilon, int checkInterval, int cores = 1) {
  //const double epsilon = 0.1;
  //const int checkInterval = 1;
  arma::mat X(init_X); arma::mat Y(init_Y);
  double prevJ;

  Rprintf("Initial cost\t%d\n", implicitCost(X, Y, P, C, lambda, cores));

  for (int b = 1; b <= batches; b++) {
    Rprintf("batch %d", b);
    updateImplicitX(X, Y, P, C, lambda, cores);
    updateImplicitY(X, Y, P, C, lambda, cores);

    double J = implicitCost(X, Y, P, C, lambda, cores);
    Rprintf("\tcost\t%f\n", J);
  }

  List ret;
  // Could also add dimension attributes
  ret["X"] = X;
  ret["Y"] = Y;

  return ret;
}

// [[Rcpp::export]]
arma::mat explain_predict(const arma::mat & X, const arma::mat & Y, const arma::mat & P, const arma::mat & C, double lambda, int u) {
  int num_users = C.n_rows;
  int num_prods = C.n_cols;
  int num_factors = Y.n_cols; // or X.n_cols

  Rprintf("In explain_predict()");
  arma::mat YTY = Y.t() * Y;
  arma::mat fact_eye = eye(num_prods, num_prods);
  arma::mat lambda_eye = lambda * eye(num_factors, num_factors);

  arma::mat Cu = diagmat(C.row(u));
  arma::mat YTCuIY = Y.t() * (Cu) * Y;
  arma::mat YTCupu = Y.t() * (Cu + fact_eye) * P.row(u).t();
  arma::mat WuT = YTY + YTCuIY + lambda_eye;
  //arma::mat xu = solve(WuT, YTCupu);
  
  arma::mat sij = Y * solve(WuT, Y.t());
  arma::mat p = sij * Cu;
  
//  Numericarma::matrix parma::mat(wrap(p));
//  parma::mat.attr("colnames") = C.attr("colnames");
  
//  colnames(p) = colnames(C);
//  rownames(p) = colnames(C);
  
  return p;
}

