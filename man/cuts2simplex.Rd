\name{cuts2simplex}
\alias{cuts2simplex}
\alias{torch_simplexify}
\alias{torch_simplexify_}
\title{Changes Cumulative Probabilities to Category Probabilities}
\description{

  Assume that there is a probability distribution over
  a variable \eqn{X} with \eqn{K} states.  The function
  \code{cuts2simplex} takes probabilities of the form \eqn{P(X>k)}
  and transforms them into the form \eqn{P(X=k)}.  The function
  \code{torch_simplexify} transforms vector into a non-negative
  vector that sums to one.

}
\usage{
cuts2simplex(et)
torch_simplexify(x,dim=-1L)
torch_simplexify_(x,dim=-1L)
}
\arguments{
  \item{et}{A torch matrix of probabilities with rows running
    highest category to lowest.}
  \item{x}{A torch vector or array to be normalized.}
  \item{dim}{Which dimenson to sum over, defaults to all.}
}
\details{

  A simplex is a vector whose values are non-negative and sum
  to one.  These functions return matrixes whose rows are
  simplexes.

  The function \code{cuts2simplex} is the heart of the
  \code{\link{DifferenceLink}} and \code{\link{GradedResponseLink}}.
  Assuming that the values of the target variable, \eqn{X},
  run from 0 to \eqn{K}, the values in the input are
  \eqn{P(X<k| row i)=y_{ik}} where \eqn{k} is the column number.
  The output for element \eqn{i,k} is \eqn{y_{i(k+1)}-y_{ik}}, where
  \eqn{y_{i0}=0} and \eqn{y_{i(K+1)}=1}.  The result is a
  matrix with one more column than the input.

  The function \code{torch_simplexify} drops the signs
  from the input and then divides by the sum so that all
  values are between 0 and 1 and sum to 1.

}
\value{

  A tensor whose rows are simplexes.

}
\author{Russell Almond}
\note{

  This version of \code{cuts2simplex} is best suited to the
  case where the states go from lowest to highest.  If the
  other way around, try:

  \code{torch_fliplr(cuts2simplex(torch_fliplr(et)))}.

  Maybe I should add a \code{high2low} switch.

}
\seealso{
  \code{\link{GradedResponseLink}},
  \code{\link[torch]{torch_fliplr}}
}
\examples{

  tm <- torch_tensor(matrix(c(.25,.75,.3,.7),2,2,byrow=TRUE))
  cuts2simplex(tm)

  torch_simplexify(torch_tensor(matrix(1,1,3)))


}
\keyword{ mathematics }
\concept{ simplex }
\concept{ graded response }

