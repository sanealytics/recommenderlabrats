REAL_ALS <- function(data, parameter= NULL) {
  
  p <- .get_parameters(list(
    categories = min(100, round(dim(data@data)[2]/2)),
    normalize = "center",
    lambda = .001, # regularization
    epsilon = 0.1,
    alpha = .001,
    optim_more = FALSE,
    minRating = NA,
    itmNormalize = FALSE,
    sampSize = NULL,
    scaleFlg = FALSE,
    batchMode = FALSE,
    item_bias_fn=function(x) {0},
    maxit = 100 # Number of iterations
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
    
    Y <- as.matrix(combineddata) # This changes NAs to 0
    R <- 1 * (Y != 0)
    
    # normalize by Movies
    Y.avg  <- apply(Y, 2, function(x) {tmp = x
                                       tmp[tmp==0] <- NA
                                       mn = mean(tmp, na.rm = TRUE)
                                       if(is.na(mn)) 0 else mn
    })
    
    # Adjusted
    if(p$itmNormalize) {
      print("TODO: Check formula")
      print("Mean normalize by items")
      #Y      <- t(t(Y) - Y.avg) * R
    } else {
      print("Did not Mean normalize")
    }
    
    print("initializing params")
    
    # initialization
    num_users  <- nrow(Y)
    num_movies <- ncol(Y)
    num_features <- model$categories
    lambda = model$lambda
    maxit = model$maxit
    epsilon = model$epsilon
    alpha = model$alpha

    # We are going to scale the data so that things converge quickly
    scale.fctr <- base::max(base::abs(Y))
    if (model$scaleFlg) {
      print("scaling down")
      Y <- Y / scale.fctr
    } else {
      print("Did not scale")
    }
    
    init_X = matrix(runif(num_users * num_features), ncol = num_features)
    init_Theta = matrix(runif(num_movies * num_features), ncol = num_features)
    print("Starting ALS")

    print(system.time(res <- als(init_X, init_Theta,
                                 Y = Y, R = R, 
                                 lambda = lambda, alpha = alpha, batches = maxit,
                                 epsilon = epsilon, checkInterval = 1, batchMode = model$batchMode)
    ))
    
    X_final     <- res$X
    theta_final <- res$Theta
    
    Y_final <- (X_final %*% t(theta_final) )
    if (model$scaleFlg) {
      Y_final <- Y_final * scale.fctr 
    }
    
    FUN <- match.fun(model$item_bias_fn)
    Y.adj   <- FUN(Y.avg)
    print(paste0("applying item_bias_fn  range ", min(Y.adj), " to ", max(Y.adj)))
    #Y_final <- Y_final + Y.adj
    
    dimnames(Y_final) = dimnames(Y)
    
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
  new("Recommender", method = "ALS", dataType = class(data),
      ntrain = nrow(data), model = model, predict = predict)
}

# Sys.setenv("PKG_CXXFLAGS" = "-fopenmp")
# Sys.setenv("PKG_LIBS" = "-fopenmp")

# recommenderRegistry$delete_entry(
#   method="ALS", dataType = "realRatingMatrix", fun=REAL_ALS,
#   description="Recommender based on Low Rank Matrix Factorization ALS (real data).")
# 
recommenderRegistry$set_entry(
  method="ALS", dataType = "realRatingMatrix", fun=REAL_ALS,
  description="Recommender based on Low Rank Matrix Factorization ALS (real data).")