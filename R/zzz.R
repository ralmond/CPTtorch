TORCH_DEVICE <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")

.onAttach <- function(libname,pkgname) {
  # require(torch)
  # torch:::load_cudatoolkit_libs()
}

.onLoad <- function(libname, pkgname) {
  # Set the default device to "cuda" if available, otherwise "cpu"
  assign(
    x = "TORCH_DEVICE",
    value = if (cuda_is_available()) torch_device("cuda") else torch_device("cpu"),
    envir = parent.env(environment())  # package namespace
  )
  if (is.null(getOption("CPTtorch_device")))
    options(CPTtorch_device={
      if (cuda_is_available()) torch_device("cuda")
      else torch_device("cpu")
    })
  # assign(
  #   x = "TORCH_DEVICE",
  #   value = torch_device("cpu"),
  #   envir = parent.env(environment())  # package namespace
  # )
}
