REAL_RSVD_SPLIT <- function(data, parameter= NULL) {
  
  p <- .get_parameters(list(
    categories = min(100, round(sqrt(dim(data@data)[2]))),
    normalize = "center",
    lambda = 1.5, # regularization
    optim_more = FALSE,
    minRating = NA,
    alpha = 1,
    X = NA,
    itmNormalize = FALSE,
    scaleFlg = FALSE,
    sampSize = NULL,
    maxit = 100 # Number of iterations for optim
  ), parameter)
  
  Y <- t(as.matrix(data@data)) # This changes NAs to 0
  if(!is.null(p$sampSize)) {
    Y <- Y[, sample(ncol(Y), p$sampSize)]
    print(paste("Took sample of size ", p$sampSize))
  }
  
  R <- 1 * (Y != 0)
  
  Y.avg  <- apply(Y, 1, function(x) {tmp = x
                                     tmp[tmp==0] <- NA
                                     mn = mean(tmp, na.rm = TRUE)
                                     if(is.na(mn)) 0 else mn
  })
  
  # Adjusted
  if(p$itmNormalize) {
    print("Mean normalize by items")
    Y      <- (Y - Y.avg) * R
  } else {
    print("Did not Mean normalize")
  }
  
  num_movies <- dim(Y)[1]
  num_users  <- dim(Y)[2]
  num_features <- p$categories
  lambda = p$lambda
  alpha = p$alpha
  maxit = p$maxit 
  # We are going to scale the data so that optim converges quickly
  scale.fctr <- base::max(base::abs(Y))
  if(p$scaleFlg) {
    print("Scaling")
    Y <- Y / scale.fctr
  } else {
    print("Did not scale")
  }
  
  if(length(p$X) == 1) {
    X <- matrix(runif(num_movies * num_features), 
                nrow=num_movies)
  } else {
    print("Using provided pre-computed X")
    X <- p$X
    p$X <- NULL
    dimnames(X) <- NULL
  }
  
  Theta <- matrix(runif(num_users * num_features), 
                  nrow=num_users)
  
  print(system.time(
    res <- optim(par = c(X, Theta), 
                 fn = J_cost, gr = grr, 
                 Y=Y, R=R, 
                 num_users=num_users, num_movies=num_movies,num_features=num_features, 
                 lambda=lambda, alpha = alpha, method = "L-BFGS-B", control=list(maxit=maxit, trace=1)) 
  ))    
  
  print(paste("final cost: ", res$value, " convergence: ", res$convergence, 
              res$message, " counts: ", res$counts["function"]))
  
  unrolled <- unroll_Vecs(res$par, Y, R, num_users, num_movies, num_features)
  
  X_final     <- unrolled$X
  Theta_final <- unrolled$Theta
  rownames(X_final) <- rownames(Y)
  rownames(Theta_final) <- colnames(Y)
  
  model <- c(list(
    description = "full matrix",
    scale.fctr = scale.fctr,
    Y.avg = Y.avg,
    X_final = X_final,
    Theta_final = Theta_final
  ), p)
  
  predict <- function(model, newdata, n = 10, maxit = model$maxit,
                      type=c("topNList", "ratings"), denormalize=TRUE, 
                      item_bias_fn=function(x) {0}, removeOldRtngsFlg = TRUE, ...) {  
    
    type <- match.arg(type)
    n <- as.integer(n)
        
    # new user
    Y <- as.matrix(t(newdata@data)) # This changes NAs to 0
    R <- 1 * (Y != 0)
    
    Y.avg  <- model$Y.avg
    
    # Adjusted
    if(model$itmNormalize)
      Y      <- (Y - Y.avg) * R
    
    num_movies <- dim(Y)[1]
    num_users  <- dim(Y)[2]
    num_features <- model$categories
    lambda = model$lambda
    alpha = model$alpha
    # We are going to scale the data so that optim converges quickly
    scale.fctr <- model$scale.fctr
    if (model$scaleFlg)
      Y <- Y / scale.fctr
    
    X_final <- model$X_final
    
    print(system.time(
      res <- optim(par = c(runif(num_users * num_features)), 
                   fn = J_cost_Theta, gr = grr_Theta, 
                   Y=Y, R=R, 
                   num_users=num_users, num_movies=num_movies,num_features=num_features, 
                   lambda=lambda, alpha=alpha, X=X_final,
                   method = "L-BFGS-B", control=list(maxit=maxit, trace=1)) 
    ))    
    
    print(paste("final cost: ", res$value, " convergence: ", res$convergence, 
                res$message, " counts: ", res$counts["function"]))
    
    unrolled <- unroll_Vecs_Theta(res$par, Y, R, num_users, num_movies, num_features, X_final)
    
    theta_final <- unrolled$Theta
    
    Y_final <- (X_final %*% t(theta_final) ) 
    if (model$scaleFlg)
      Y_final <- Y_final * scale.fctr 
    
    rm(res, R, X_final, theta_final) # To free some memory
    FUN <- match.fun(item_bias_fn)
    Y.adj   <- FUN(Y.avg)
    print(paste("applying item_bias_fn  range ", min(Y.adj), " to ", max(Y.adj)))
    Y_final <- Y_final + Y.adj
    
    dimnames(Y_final) = dimnames(Y)
    
    rm(Y)
    gc()
    print("Converting to ratings")
    
    ratings <- t(Y_final)
    
    # Only need to give back new users
    ratings <- new("realRatingMatrix", data=as(ratings, "dgCMatrix"))
    
    if(removeOldRtngsFlg)
      ratings <- removeKnownRatings(ratings, newdata)
    
    if(type=="ratings") return(ratings)
    
    getTopNLists(ratings, n=n, minRating=model$minRating)
    
  }
  
  ## construct recommender object
  new("Recommender", method = "RSVD", dataType = class(data),
      ntrain = nrow(data), model = model, predict = predict)
}

# For both X and Theta
unroll_Vecs <- function (params, Y, R, num_users, num_movies, num_features) {
  endIdx <- num_movies * num_features
  
  X     <- matrix(params[1:endIdx], nrow = num_movies, ncol = num_features)
  Theta <- matrix(params[(endIdx + 1): (endIdx + (num_users * num_features))], 
                  nrow = num_users, ncol = num_features)
  
  Y_dash     <-   (((X %*% t(Theta)) - Y) * R)
  
  return(list(X = X, Theta = Theta, Y_dash = Y_dash))
}

J_cost <-  function(params, Y, R, num_users, num_movies, num_features, lambda, alpha) {
  
  unrolled <- unroll_Vecs(params, Y, R, num_users, num_movies, num_features)
  X <- unrolled$X
  Theta <- unrolled$Theta
  Y_dash <- unrolled$Y_dash
  
  J <-  .5 * sum(   Y_dash ^2)         + lambda/2 * sum(Theta^2) + lambda/2 * sum(X^2)
  
  return (J) #list(J, grad))
}

grr <- function(params, Y, R, num_users, num_movies, num_features, lambda, alpha) {
  
  unrolled <- unroll_Vecs(params, Y, R, num_users, num_movies, num_features)
  X <- unrolled$X
  Theta <- unrolled$Theta
  Y_dash <- unrolled$Y_dash
  
  X_grad     <- ((   Y_dash  %*% Theta) + lambda * X     )
  Theta_grad <- (( t(Y_dash) %*% X)     + lambda * Theta )
  
  grad = c(X_grad, Theta_grad)
  return(grad)
}

# For Theta only
# Given X
# TODO: Could add bias term here but no need really
unroll_Vecs_Theta <- function (params, Y, R, num_users, num_movies, num_features, X) {
  
  Theta <- matrix(params, 
                  nrow = num_users, ncol = num_features)
  
  Y_dash     <-   (((X %*% t(Theta)) - Y) * R)
  
  return(list(Theta = Theta, Y_dash = Y_dash))
}

J_cost_Theta <-  function(params, Y, R, num_users, num_movies, num_features, lambda, alpha, X) {
  
  unrolled <- unroll_Vecs_Theta(params, Y, R, num_users, num_movies, num_features, X)
  
  Theta <- unrolled$Theta
  Y_dash <- unrolled$Y_dash
  
  J <-  .5 * sum(   Y_dash ^2)         + lambda/2 * sum(Theta^2) + lambda/2 * sum(X^2)
  
  return (J) 
}

grr_Theta <- function(params, Y, R, num_users, num_movies, num_features, lambda, alpha, X) {
  
  unrolled <- unroll_Vecs_Theta(params, Y, R, num_users, num_movies, num_features, X)
  
  Theta <- unrolled$Theta
  Y_dash <- unrolled$Y_dash
  
  Theta_grad <- alpha * (( t(Y_dash) %*% X)     + lambda * Theta)
  
  grad = c(Theta_grad)
  return(grad)
}

# Helper functions
# if( !is.null(recommenderRegistry["RSVD_SPLIT", "realRatingMatrix"])) {
#   recommenderRegistry$delete_entry(
#     method="RSVD_SPLIT", dataType = "realRatingMatrix", fun=REAL_RSVD_SPLIT,
#     description="Recommender based on Low Rank Matrix Factorization (real data).")
# }

# register recommender
recommenderRegistry$set_entry(
  method="RSVD_SPLIT", dataType = "realRatingMatrix", fun=REAL_RSVD_SPLIT,
  description="Recommender based on Low Rank Matrix Factorization (real data).")
