#include <RcppArmadillo.h>
#include <omp.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// compile with these flags to make this parallel
// Sys.setenv("PKG_CXXFLAGS" = "-fopenmp")
// Sys.setenv("PKG_LIBS" = "-fopenmp")
// TODO: Add stopping condition by tolerance
// TODO: Add early stopping
// TODO: Use sparse matrices wherever possible


// [[Rcpp::export]]
void updateX(mat & X, const mat & Y, const mat & P, const mat & C, double lambda) {
  int num_users = C.n_rows;
  int num_prods = C.n_cols;
  int num_factors = Y.n_cols; // or X.n_cols

  Rprintf(".");
  mat YTY = Y.t() * Y;
  mat fact_eye = eye(num_prods, num_prods);
  mat lambda_eye = lambda * eye(num_factors, num_factors);

  // TODO: Randomize order of users
#pragma omp parallel for
  for (int u = 0; u < C.n_rows; u++) {
    mat Cu = diagmat(C.row(u));
    mat YTCuIY = Y.t() * (Cu) * Y;
    mat YTCupu = Y.t() * (Cu + fact_eye) * P.row(u).t();
    mat WuT = YTY + YTCuIY + lambda_eye;
    mat xu = solve(WuT, YTCupu);

    // Update gradient -- maybe a slow operation in parallel?
    X.row(u) = xu.t();
  }
}

// [[Rcpp::export]]
void updateY(const mat & X, mat & Y, const mat & P, const mat & C, double lambda) {
  int num_users = C.n_rows;
  int num_prods = C.n_cols;
  int num_factors = Y.n_cols; // or X.n_cols

  Rprintf(".");
  mat XTX = X.t() * X;
  mat fact_eye = eye(num_users, num_users);
  mat lambda_eye = lambda * eye(num_factors, num_factors);

  // TODO: Randomize order of items
#pragma omp parallel for
  for (int i = 0; i < C.n_cols; i++) {
    mat Ci = diagmat(C.col(i));
    mat YTCiIY = X.t() * (Ci) * X;
    mat YTCipi = X.t() * (Ci + fact_eye) * P.col(i);
    mat yu = solve(XTX + YTCiIY + lambda_eye, YTCipi);

    // Update gradient
    Y.row(i) = yu.t();
  }
}

double cost(const mat &X, const mat &Y, const mat &P, const mat &C, double lambda) {
  return accu(dot(C, square(P - X * Y.t())) + 
    lambda * (pow(accu(X),2) + pow(accu(Y),2)));
}

// [[Rcpp::export]]
List implicit(const mat & init_X, const mat & init_Y, const mat & P, const mat & C,
        double lambda, int batches,
        double epsilon, int checkInterval) {
  //const double epsilon = 0.1;
  //const int checkInterval = 1;
  mat X(init_X); mat Y(init_Y);
  double prevJ;

  Rprintf("Initial cost\t%d\n", cost(X, Y, P, C, lambda));

  for (int b = 1; b <= batches; b++) {
    Rprintf("batch %d", b);
    updateX(X, Y, P, C, lambda);
    updateY(X, Y, P, C, lambda);

    double J = cost(X, Y, P, C, lambda);
    Rprintf("\tcost\t%f\n", J);
  }

  List ret;
  // Could also add dimension attributes
  ret["X"] = X;
  ret["Y"] = Y;

  return ret;
}

// [[Rcpp::export]]
mat explain_predict(const mat & X, const mat & Y, const mat & P, const mat & C, double lambda, int u) {
  int num_users = C.n_rows;
  int num_prods = C.n_cols;
  int num_factors = Y.n_cols; // or X.n_cols

  Rprintf("In explain_predict()");
  mat YTY = Y.t() * Y;
  mat fact_eye = eye(num_prods, num_prods);
  mat lambda_eye = lambda * eye(num_factors, num_factors);

  mat Cu = diagmat(C.row(u));
  mat YTCuIY = Y.t() * (Cu) * Y;
  mat YTCupu = Y.t() * (Cu + fact_eye) * P.row(u).t();
  mat WuT = YTY + YTCuIY + lambda_eye;
  //mat xu = solve(WuT, YTCupu);
  
  mat sij = Y * solve(WuT, Y.t());
  mat p = sij * Cu;
  
//  NumericMatrix pmat(wrap(p));
//  pmat.attr("colnames") = C.attr("colnames");
  
//  colnames(p) = colnames(C);
//  rownames(p) = colnames(C);
  
  return p;
}

