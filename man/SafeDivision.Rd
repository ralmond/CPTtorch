\name{safeDivision}
\alias{\%//\%}
\title{Division with Special Handling for Infinite Divisors}
\description{

  If \code{e2} is finite then this is ordinary decision.
  If \code{e2} is infinite, the value is zero unless
  \code{e1} is also infinite

}
\usage{
e1 \%//\% e2
}
\arguments{
  \item{e1,e2}{ Values to be divided.}
}
\value{
  The quotiant
}
\author{Russell Almond}
\seealso{
  \code{\link{softmax}}
}
\examples{
  1 \%//\% 2
  1 \%//\% Inf
  Inf \%//\% Inf
}
\keyword{ mathematics }
