\name{logsumexp}
\alias{logsumexp}
\alias{prod_1}
\alias{torch_prod_1}
\alias{prodq}
\alias{torch_prodq}
\alias{sumrootk}
\alias{torch_sumrootk}
\title{Functions which Summarize a Vector}
\description{

  These functions reduce a vector to a scalar.
}
\usage{
logsumexp(x)
prod_1(x)
torch_prod_1(x, dim=-1L, keepdim = FALSE)
prodq(x)
torch_prodq(x, dim=-1L, keepdim = FALSE)
sumrootk(x)
torch_sumrootk(x, dim= -1L, keepdim = FALSE)
}
\arguments{
  \item{x}{The vector (or array for torch objects) to be
    reduced.}
  \item{dim}{Which dimension will be reduced
    (default is all).}
  \item{keepdim}{Whether the output tensor has its dim
    retained.}
}
\details{

  The function \code{logsumexp} produces the log of the
  sum of the exponentials of \code{x}.

  The function \code{prod_1} returns 1 minus the
  product of \code{x}. Used by \code{\link{NoisyOrRule}}.

  The function \code{prodq} returns 1 minus the product
  of \code{1-x}.  This is a product of \eqn{q=1-p}.

  The function \code{sumrootk} returns the sum of \code{x}
  divided by the square root of the length of \code{x}.
  This is a variance stablization transformation as the
  variance of the sum would otherwise grow with the square
  root of the length.  Used by \code{\link{CompensatoryRule}}.

}
\value{

  A scalar value or tensor (\code{torch_} functions) giving
  the result of the operation.

}
\author{Russell Almond}
\seealso{
  \code{\link{CompensatoryRule}}, \code{\link{NoisyOrRule}}
}
\examples{

  logsumexp(c(0,1))
  prod_1(c(.5,.5))
  prodq(c(.25,.25))
  sumrootk(c(1,1))

}
\keyword{ mathematics }
\concept{ summaries }

% Use only one concept per line.
