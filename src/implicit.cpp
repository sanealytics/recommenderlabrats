#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

// Presumably no need to compile with these flags to make this parallel anymore
// Sys.setenv("PKG_CXXFLAGS" = "-fopenmp")
// Sys.setenv("PKG_LIBS" = "-fopenmp")
// TODO: Add stopping condition by tolerance
// TODO: Add early stopping
// TODO: Use sparse arma::sp_matrices wherever possible


// [[Rcpp::export]]
void updateImplicitX(arma::sp_mat & X, const arma::sp_mat & Y, const arma::sp_mat & P, const arma::sp_mat & C, 
  double lambda, int cores) {
  int num_users = C.n_rows;
  int num_prods = C.n_cols;
  int num_factors = Y.n_cols; // or X.n_cols

  Rprintf(".");
  arma::sp_mat YTY = Y.t() * Y;
  arma::sp_mat fact_eye = arma::speye(num_prods, num_prods);
  arma::sp_mat lambda_eye = lambda * speye(num_factors, num_factors);

#pragma omp parallel for num_threads(cores)
  for (int u = 0; u < C.n_rows; u++) {
    arma::sp_mat Cu = diagmat(C.row(u));
    arma::sp_mat YTCuIY = Y.t() * (Cu) * Y;
    arma::sp_mat YTCupu = Y.t() * (Cu + fact_eye) * P.row(u).t();
    arma::sp_mat WuT = YTY + YTCuIY + lambda_eye;
    arma::sp_mat xu = arma::spsolve(WuT, YTCupu, "superlu");

    // Update gradient -- maybe a slow operation in parallel?
    X.row(u) = xu.t();
  }
}

// [[Rcpp::export]]
void updateImplicitY(const arma::sp_mat & X, arma::sp_mat & Y, const arma::sp_mat & P, const arma::sp_mat & C, 
  double lambda, int cores) {
  int num_users = C.n_rows;
  int num_prods = C.n_cols;
  int num_factors = Y.n_cols; // or X.n_cols

  Rprintf(".");
  arma::sp_mat XTX = X.t() * X;
  arma::sp_mat fact_eye = arma::speye(num_users, num_users);
  arma::sp_mat lambda_eye = lambda * arma::speye(num_factors, num_factors);

#pragma omp parallel for num_threads(cores)
  for (int i = 0; i < C.n_cols; i++) {
    arma::sp_mat Ci = diagmat(C.col(i));
    arma::sp_mat YTCiIY = X.t() * (Ci) * X;
    arma::sp_mat YTCipi = X.t() * (Ci + fact_eye) * P.col(i);
    arma::sp_mat yu = arma::spsolve(XTX + YTCiIY + lambda_eye, YTCipi);

    // Update gradient
    Y.row(i) = yu.t();
  }
}

// [[Rcpp::export]]
double implicitCost(const arma::sp_mat & X, const arma::sp_mat & Y, const arma::sp_mat & P, const arma::sp_mat & C, double lambda,
  int cores) {
  double delta = 0.0;
#pragma omp parallel for num_threads(cores)
  for (int u = 0; u < C.n_rows; u++) {
    delta += accu(dot(C.row(u), square(P.row(u) - X.row(u) * Y.t())));
  }
  return (delta + 
    lambda * (pow(accu(X),2) + pow(accu(Y),2))) / (C.n_rows * C.n_cols);
}

// [[Rcpp::export]]
List implicit(const arma::sp_mat & init_X, const arma::sp_mat & init_Y, const arma::sp_mat & P, const arma::sp_mat & C,
        double lambda, int batches,
        double epsilon, int checkInterval, int cores = 1) {
  //const double epsilon = 0.1;
  //const int checkInterval = 1;
  arma::sp_mat X(init_X); arma::sp_mat Y(init_Y);
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
arma::sp_mat explain_predict(const arma::sp_mat & X, const arma::sp_mat & Y, const arma::sp_mat & P, const arma::sp_mat & C, double lambda, int u) {
  int num_users = C.n_rows;
  int num_prods = C.n_cols;
  int num_factors = Y.n_cols; // or X.n_cols

  Rprintf("In explain_predict()");
  arma::sp_mat YTY = Y.t() * Y;
  arma::sp_mat fact_eye = speye(num_prods, num_prods);
  arma::sp_mat lambda_eye = lambda * speye(num_factors, num_factors);

  arma::sp_mat Cu = diagmat(C.row(u));
  arma::sp_mat YTCuIY = Y.t() * (Cu) * Y;
  arma::sp_mat YTCupu = Y.t() * (Cu + fact_eye) * P.row(u).t();
  arma::sp_mat WuT = YTY + YTCuIY + lambda_eye;
  //arma::sp_mat xu = solve(WuT, YTCupu);
  
  arma::sp_mat sij = Y * spsolve(WuT, Y.t());
  arma::sp_mat p = sij * Cu;
  
//  Numericarma::sp_matrix parma::sp_mat(wrap(p));
//  parma::sp_mat.attr("colnames") = C.attr("colnames");
  
//  colnames(p) = colnames(C);
//  rownames(p) = colnames(C);
  
  return p;
}

