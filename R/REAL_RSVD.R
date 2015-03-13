REAL_RSVD <- function(data, parameter= NULL) {
  
  p <- .get_parameters(list(
    categories = min(100, round(dim(data@data)[2]/2)),
    normalize = "center",
    lambda = 1.5, # regularization
    optim_more = FALSE,
    minRating = NA,
    itmNormalize = FALSE,
    sampSize = NULL,
    scaleFlg = FALSE,
    item_bias_fn=function(x) {0},
    maxit = 100, # Number of iterations for optim
    optimize = function(...) {optim(method = "L-BFGS-B", ...)}
  ), parameter)
  
  model <- c(list(
    description = "full matrix",
    data = data
  ), p)
  
  predict <- function(model, newdata, n = 10,
                      type=c("topNList", "ratings"), ...) {
    
    type <- match.arg(type)
    n <- as.integer(n)
    
    # Combine new user
    combineddata <- model$data@data
    combineddata <- rBind(combineddata, newdata@data)
    
    Y <- t(as.matrix(combineddata)) # This changes NAs to 0
    if(!is.null(model$sampSize)) {
      Y <- Y[, sample(ncol(Y), model$sampSize)]
      print("Took sample of size ", model$sampSize)
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
    
    # initialization
    num_movies <- dim(Y)[1]
    num_users  <- dim(Y)[2]
    num_features <- model$categories
    lambda = model$lambda
    maxit = model$maxit
    # We are going to scale the data so that optim converges quickly
    scale.fctr <- base::max(base::abs(Y))
    if (model$scaleFlg) {
      print("scaling down")
      Y <- Y / scale.fctr
    }
    
    print(system.time(
      res <- model$optimize(par = runif(num_movies * num_features + num_users * num_features), 
                   fn = J_cost, gr = grr, 
                   Y=Y, R=R, 
                   num_users=num_users, num_movies=num_movies,num_features=num_features, 
                   lambda=lambda, control=list(maxit=maxit, trace=1)) 
    ))    
    
    print(paste("final cost: ", res$value, " convergence: ", res$convergence, 
                res$message, " counts: ", res$counts["function"]))
    
    unrolled <- unroll_Vecs(res$par, Y, R, num_users, num_movies, num_features)
    
    X_final     <- unrolled$X
    theta_final <- unrolled$Theta
    
    Y_final <- (X_final %*% t(theta_final) )
    if (model$scaleFlg) {
      Y_final <- Y_final * scale.fctr 
    }
    
    FUN <- match.fun(model$item_bias_fn)
    Y.adj   <- FUN(Y.avg)
    print(paste0("applying item_bias_fn  range ", min(Y.adj), " to ", max(Y.adj)))
    Y_final <- Y_final + Y.adj
    
    dimnames(Y_final) = dimnames(Y)
    
    ratings <- t(Y_final)
    
    # Only need to give back new users
    ratings <- ratings[(dim(model$data@data)[1]+1):dim(ratings)[1],]
    
    ratings <- new("realRatingMatrix", data=drop0(ratings))
    
    ratings@normalize <- newdata@normalize
    
    ratings <- removeKnownRatings(ratings, newdata)
    
    if(type=="ratings") return(ratings)
    
    getTopNLists(ratings, n=n, minRating=model$minRating)
    
  }
  
  # Helper functions
  
  unroll_Vecs <- function (params, Y, R, num_users, num_movies, num_features) {
    endIdx <- num_movies * num_features
    
    X     <- matrix(params[1:endIdx], nrow = num_movies, ncol = num_features)
    Theta <- matrix(params[(endIdx + 1): (endIdx + (num_users * num_features))], 
                    nrow = num_users, ncol = num_features)
    
    Y_dash     <-   (((X %*% t(Theta)) - Y) * R)
    
    return(list(X = X, Theta = Theta, Y_dash = Y_dash))
  }
  
  J_cost <-  function(params, Y, R, num_users, num_movies, num_features, lambda) {
    
    unrolled <- unroll_Vecs(params, Y, R, num_users, num_movies, num_features)
    X <- unrolled$X
    Theta <- unrolled$Theta
    Y_dash <- unrolled$Y_dash
    
    J <-  .5 * sum(   Y_dash ^2)  + lambda/2 * sum(Theta^2) + lambda/2 * sum(X^2)
    
    return (J) #list(J, grad))
  }
  
  grr <- function(params, Y, R, num_users, num_movies, num_features, lambda) {
    
    unrolled <- unroll_Vecs(params, Y, R, num_users, num_movies, num_features)
    X <- unrolled$X
    Theta <- unrolled$Theta
    Y_dash <- unrolled$Y_dash
    
    X_grad     <- (   Y_dash  %*% Theta) + lambda * X
    Theta_grad <- ( t(Y_dash) %*% X)     + lambda * Theta
    
    grad = c(X_grad, Theta_grad)
    return(grad)
  }
  
  ## construct recommender object
  new("Recommender", method = "RSVD", dataType = class(data),
      ntrain = nrow(data), model = model, predict = predict)
}

# Helper functions
# if( !is.null(recommenderRegistry["RSVD", "realRatingMatrix"])) {
#   recommenderRegistry$delete_entry(
#     method="RSVD", dataType = "realRatingMatrix", fun=REAL_RSVD,
#     description="Recommender based on Low Rank Matrix Factorization (real data).")
# }

# register recommender
recommenderRegistry$set_entry(
  method="RSVD", dataType = "realRatingMatrix", fun=REAL_RSVD,
  description="Recommender based on Low Rank Matrix Factorization (real data).")