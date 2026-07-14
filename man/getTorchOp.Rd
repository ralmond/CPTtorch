\name{getTorchOp}
\alias{getTorchOp}
\alias{getTorchSummaryOp}
\alias{getTorchUnaryOp}
\alias{getTorchVectorOp}
\alias{getZeroOp}
\title{Return the Torch Equivalent of an R Operator}
\description{

  This takes the name of a function and returns the
  \code{torch} equivalent.

  The function \code{getZeroOp} returns the zero value
  for a summary operator.
}
\usage{
getTorchOp(op)
getTorchSummaryOp(op)
getTorchUnaryOp(opname)
getTorchVectorOp(opname)
getZeroOp(op)
}
\arguments{
  \item{op,opname}{An operator or the name of an operator.}
}
\value{

  An equvalent \code{torch} operator.

  In the case of \code{getZeroOp} the value to
  use as zero with a particular summary.

}
\author{Russell Almond}
\note{

  This is a generally useful facility and not particularly
  CPTtorch specific (although there are a few additional
  functions specific to CPTtorch).  I might move this to
  another package (or upstream to torch).
}
\examples{

  getTorchOp("*")
  getTorchSummaryOp(prod)
  getZeroOp(prod)
}
\keyword{ utilities }
