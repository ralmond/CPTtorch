.onLoad <- function(libname, pkgname) {
  # Set the default device to "cuda" if available, otherwise "cpu"
  if (torch::cuda_is_available()) {
    torch::local_device("cuda")
  } else {
    torch::local_device("cpu")
  }
}
