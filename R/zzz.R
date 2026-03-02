.onLoad <- function(libname, pkgname) {
  # Set the default device to "cuda" if available, otherwise "cpu"
  assign(
    x = "TORCH_DEVICE",
    value = if (cuda_is_available()) torch_device("cuda") else torch_device("cpu"),
    envir = parent.env(environment())  # package namespace
  )
}
