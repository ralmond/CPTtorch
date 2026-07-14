\name{as_torch_tensor}
\alias{as_torch_tensor}
\alias{torch_tensor}
\title{Coerces an Object into a `torch_tensor'.}
\description{

  If the argument is not already a
  \code{\link[torch]{torch_tensor}}, then turn
  it into one.  Note that \code{torch_tensor} is
  exported from the \code{torch} package.

}
\usage{
as_torch_tensor(x, device = CPTtorch_device())
torch_tensor(data, dtype = NULL, device = NULL,
  requires_grad = FALSE, pin_memory = FALSE)
}
\arguments{
  \item{x, data}{Object to be turned into a tensor.}
  \item{device}{Whether the tensor should be
    \code{\link[torch]{torch_device}("cpu")} or
    \code{\link[torch]{torch_device}("gpu")}.}
  \item{dtype, requires_grad, pin_memory}{See
    \code{\link[torch]{torch_tensor}}.}
}
\value{
  A \code{\link[torch]{torch_tensor}}.
}
\author{Russell Almond}
\seealso{
  \code{\link[torch]{torch_tensor}},
  \code{\link{CPTtorch_device}}
}
\examples{

  tt1 <- as_torch_tensor(1:3,torch::torch_device("cpu"))
  tt2 <- as_torch_tensor(tt1)

}
\keyword{ classes }
\concept{ torch_tensor }
