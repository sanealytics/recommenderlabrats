REAL_IMPLICIT <- function(data, parameter= NULL) {
  
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
    maxit = 100, # Number of iterations 
    num_cores_per_batch = 1
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
    
    C = t(Y)
    P = (C > 0) * 1
    X = matrix(runif(num_users * num_features), ncol = num_features)
    Y = matrix(runif(num_movies * num_features), ncol = num_features)    
    
    print(system.time(
      res <- implicit(init_X = X, init_Y = Y, 
                      C=C, P=P, 
                      lambda=lambda, batches = maxit, 
                      epsilon = 0.01, checkInterval = 10,
                      cores = model$num_cores_per_batch) 
    ))
    
    
    Y_final <- (res$X %*% t(res$Y) )
    dimnames(Y_final) = dimnames(C)
    
#     if (model$scaleFlg) {
#       Y_final <- Y_final * scale.fctr 
#     }
#     
#     FUN <- match.fun(model$item_bias_fn)
#     Y.adj   <- FUN(Y.avg)
#     print(paste0("applying item_bias_fn  range ", min(Y.adj), " to ", max(Y.adj)))
#     Y_final <- Y_final + Y.adj
#     
#     dimnames(Y_final) = dimnames(Y)
    
    ratings <- Y_final
    
    # Only need to give back new users
    ratings <- ratings[(dim(model$data@data)[1]+1):dim(ratings)[1],]
    
    ratings <- new("realRatingMatrix", data=drop0(ratings))
    
    ratings@normalize <- newdata@normalize
    
    ratings <- removeKnownRatings(ratings, newdata)
    
    if(type=="ratings") return(ratings)
    
    getTopNLists(ratings, n=n, minRating=model$minRating)
    
  }
  
  ## construct recommender object
  new("Recommender", method = "REAL_IMPLICIT", dataType = class(data),
      ntrain = nrow(data), model = model, predict = predict)
}

# Helper functions
if( !is.null(recommenderRegistry["IMPLICIT", "realRatingMatrix"])) {
  recommenderRegistry$delete_entry(
    method="IMPLICIT", dataType = "realRatingMatrix", fun=REAL_IMPLICIT,
    description="Recommender based on Low Rank Matrix Factorization (real data).")
}

# register recommender
recommenderRegistry$set_entry(
  method="IMPLICIT", dataType = "realRatingMatrix", fun=REAL_IMPLICIT,
  description="Recommender based on Implicit Matrix Factorization (real data).")