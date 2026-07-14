\name{guessmat}
\alias{guessmat}
\alias{slipmat}
\alias{torch_guessmat}
\alias{torch_slipmat}
\title{Produces a Matrix to Adjust for Slips and Guesses}
\description{

  A slip moves the output to a lower category and a
  guess to a higher category.  These functions produce
  a matrix for adjusting the output of a
  \code{\linkS4class{CPT_Link}} for slipping and guessing.

}
\usage{
guessmat(n, g)
torch_guessmat(n, g)
slipmat(n,s)
torch_slipmat(n, s)
}
\arguments{
  \item{n}{The size of the matrix; number of states.}
  \item{g,s}{The guessing or slip parameter.  For the
  torch version, should be a torch_tensor.}
}
\details{

  If the states are in increasing order, the guess matrix is

  \deqn{\left [ \array{ccc}{1-g-g^2 \tab g \tab g^2 \cr
    0 \tab 1-g \tab g \cr
    0 \tab 0 \tab 1 \cr} \right ] .}

  The slip matrix is

  \deqn{\left [ \array{ccc}{1 \tab 0 \tab 0 \cr
    s \tab 1-s \tab 0 \cr
    s^2 \tab s \tab 1-s-s^2 \cr} \right ] .}

  If states are ordered from low to high, then reverse
  the columns of the result
  (\code{\link[torch]{torch_fliplr}}).


}
\value{

  A square matrix giving the slip or guess value.  For the
  \code{torch_} functions, as a torch_tensor of the same type
  as the \code{g} or \code{s} argument.

}
\author{Russell Almond}
\note{

  The default assumes \code{high2low=FALSE}, so be careful.

}
\seealso{
  \code{\linkS4class{CPT_Link}}
}
\examples{

  guessmat(3,.2)
  slipmat(4,.1)

  torch_matmul(torch_guessmat(3,torch_tensor(.1)),
    torch_slipmat(3,torch_tensor(.05)))
}
\keyword{ mathematics }
\concept{ slipping }
\concept{ guessing }
