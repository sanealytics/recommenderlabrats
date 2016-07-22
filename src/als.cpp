#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

arma::mat calculate_delta(const arma::mat & X, const arma::mat & Theta, const arma::mat & Y, const arma::mat & R) {
  // Model error = actual - prediction
  return (Y - (X * trans(Theta)) % R); // % -> schur product -> dot product
}

// [[Rcpp::export]]
double alsCost(const arma::mat & X, const arma::mat & Theta, const arma::mat & Y, const arma::mat & R, double lambda) {
    arma::mat delta = calculate_delta(X, Theta, Y, R);
    
    return .5 * accu(pow(delta, 2)) + 
      .5 * lambda * (accu(pow(X,2)) + accu(pow(Theta, 2)));
}


// [[Rcpp::export]]
void alsUpdateX(arma::mat & X, const arma::mat & Theta, const arma::mat & Y, const arma::mat & R, 
  double lambda, double alpha, bool batchMode) {

  if (batchMode) {
    arma::mat delta = calculate_delta(X, Theta, Y, R);
    arma::mat X_grad = delta * Theta + lambda * X;
    X += X_grad * alpha;
  } else {
#pragma omp parallel for
    for (int i = 0; i < Y.n_rows; i++) {
      arma::mat delta = calculate_delta(X.row(i), Theta, Y.row(i), R.row(i));
      
      // Update gradient
      arma::mat X_grad = delta * Theta + lambda * X.row(i);
      X.row(i) += X_grad * alpha; // update rule
    }
  }
}

// [[Rcpp::export]]
void alsUpdateTheta(const arma::mat & X, arma::mat & Theta, const arma::mat & Y, const arma::mat & R, 
  double lambda, double alpha, bool batchMode) {

  if (batchMode) {
    arma::mat delta = calculate_delta(X, Theta, Y, R);
    
    // Update gradient
    arma::mat Theta_grad = trans(delta) * X + lambda * Theta;
    Theta += Theta_grad * alpha; // update rule
  } else {
#pragma omp parallel for
    for (int i = 0; i < Y.n_cols; i++) {
      arma::mat delta = calculate_delta(X, Theta.row(i), Y.col(i), R.col(i));
      
      // Update gradient
      arma::mat Theta_grad = trans(delta) * X + lambda * Theta.row(i);
      Theta.row(i) += Theta_grad * alpha; // update rule
    }
  }
}

// [[Rcpp::export]]
List als(const arma::mat init_X, const arma::mat init_Theta, const arma::mat & Y, const arma::mat & R, 
        double lambda, double alpha, int batches,
        double epsilon, int checkInterval, bool batchMode) {
          
  double prevJ;
  arma::mat X(init_X);
  arma::mat Theta(init_Theta);
  std::vector<int> Costs;
  
  for (int b = 1; b <= batches; b++) {

    // Figure out user to categories
    alsUpdateX(X, Theta, Y, R, lambda, alpha, batchMode);
    
    // Figure out item to categories
    alsUpdateTheta(X, Theta, Y, R, lambda, alpha, batchMode);

    double J = alsCost(X, Theta, Y, R, lambda);
    if (b%checkInterval == 0 || b == batches) {
      Rprintf("batch\t%d, cost\t%f\n", b, J);
    }
    prevJ = J;
    Costs.push_back(J);
  }
  
  return List::create(
    Named("X") = X,
    Named("Theta") = Theta,
    Named("Costs") = Costs);
}
