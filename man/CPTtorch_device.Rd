\name{CPTtorch_device}
\alias{CPTtorch_device}
\title{Determines which torch_device to Use}
\description{

  Torch can run using either the cpu (\code{\link[torch]{torch_device}("cpu")}),
  or if cuda is installed, the gpu (\code{\link[torch]{torch_device}("gpu")}).
  Which one is used is controlled by \code{\link[base]{getOption}("CPTtorch_device")}.
  This function is just an alias for that expression.

}
\usage{
CPTtorch_device()
}
\details{

  If the option is not already set, the default value is set when the function is
  first called.  It calls \code{\link[torch]{cuda_is_available}()} to determine
  whether to use gpu or cpu.

}
\value{
  Returns either \code{\link[torch]{torch_device}("cpu")} or
  \code{\link[torch]{torch_device}("gpu")}
}
\author{Russell Almond}
\note{
Primarily used for default for `device` argument on various functions.
}
\seealso{
  \code{\link[base]{options}}, \code{\link[torch]{torch_device}},
  \code{\link[torch]{cuda_is_available}}

  Used in:
  \code{\link{readCPT}}, \code{\link{as_torch_tensor}},
  \code{\linkS4class{CombinationRule}}, \code{\link{defaultParameter10}},
  \code{\link{effectiveTheta10}}, \code{\link{buildpTheta10}},
  \code{\linkS4class{CPT_Link}}, \code{\linkS4class{CPT_Model}},
  \code{\link{penalty_fun}}, \code{\linkS4class{CognitivelyDiagnosticModel}}


}
\examples{
  CPTtorch_device()
}
\keyword{ environment }
\concept{ torch_device }

