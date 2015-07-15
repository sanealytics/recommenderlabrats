# recommenderlabrats
Some recommendation algorithms and research

To install, run -

# basic
require(devtools) # Install this if you don't have it already
install_github("sanealytics/recommenderlabrats")

# If you have openmp and are running this on a multicore machine
# To take full use of the performance offered, do
require(devtools) # Install this if you don't have it already
Sys.setenv("PKG_CXXFLAGS" = "-fopenmp -DARMA_64BIT_WORD")
Sys.setenv("PKG_LIBS" = "-fopenmp")
install_github("sanealytics/recommenderlabrats")
