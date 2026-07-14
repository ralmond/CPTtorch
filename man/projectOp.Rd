\name{projectOp}
\alias{projectOp}
\alias{projectOp,array,array-method}
\alias{projectOp,array,numeric-method}
\alias{projectOp,numeric,array-method}
\alias{projectOp,numeric,numeric-method}
\alias{projectOp,torch_tensor,torch_tensor-method}
\alias{marginalize}
\alias{marginalize,array-method}
\alias{marginalize,torch_tensor-method}
\alias{\%^*\%}
\alias{\%^+\%}
\alias{\%^-\%}
\alias{\%^/\%}
\title{Operate on Arrays while Projecting Dimensions}
\description{

  When combining two arrays, ordinarily the dimensions must
  match.  However, if one array has a dimension with extent 1,
  and the other has dimension \eqn{k}, then the combination
  can be done by replicating the smaller array \eqn{k} times.
  The function \code{projectOp} does this combination of
  projection and combination.  The operators \code{\%^+\%},
  \code{\%^-\%}, \code{\%^*\%}, and \code{\%^/\%} are shortcuts.
  The function \code{marginalize} goes in the opposite
  direction.

}
\usage{
projectOp(m1, m2, op = "+")
\S4method{projectOp}{array,array}(m1, m2, op = "+")
\S4method{projectOp}{numeric,array}(m1, m2, op = "+")
\S4method{projectOp}{array,numeric}(m1, m2, op = "+")
\S4method{projectOp}{numeric,numeric}(m1, m2, op = "+")
\S4method{projectOp}{torch_tensor,torch_tensor}(m1, m2, op = "+")
m1 \%^+\% m2
m1 \%^-\% m2
m1 \%^*\% m2
m1 \%^/\% m2
marginalize(pot, dim = 1, op = "sum")
\S4method{marginalize}{array}(pot, dim = 1, op = "sum")
\S4method{marginalize}{torch_tensor}(pot, dim = 1, op = "sum")
}
\arguments{
  \item{m1,m2}{Potentials (arrays) to be combined.}
  \item{pot}{Potential (array) to be marginalized.}
  \item{op}{The operator used to combine (\code{projectOp},
    should be binary) or reduce (\code{marginalize},
    summarizing operator).}
  \item{dim}{An integer vector giving the dimensions
    which should be eliminated.}
}
\details{

  Let \eqn{m1} and \eqn{m2} with dimensions
  \eqn{(k1_1,k1_2,\ldots,k1_J)} and
  \eqn{(k2_1,k2_2,\ldots,k2_J)}.  First assume that
  the dimensions are identical, then the combined potential
  is
  \deqn{(m1\odot m2)[k_1,k_2,\ldots,k_J] =
  m1[k_1,k_2,\ldots,k_J] \odot m2[k_1,k_2,\ldots,k_J] .}

  Now, assume that \eqn{k1_2 >1} and \eqn{k2_2=1}.  To
  make the combination work, replicate the second potential
  over the second dimension.  This yeilds the following
  rule:

  \deqn{(m1\odot m2)[k_1,k_2,\ldots,k_J] =
  m1[k_1,k_2,\ldots,k_J] \odot m2[k_1,1,\ldots,k_J] .}

  Note that torch functions, e.g.,
  \code{\link[torch]{torch_add}} and
  \code{\link[torch]{torch_mul}} do this projection already
  so \code{projectOp} only works with regular R arrays, not
  tensors.


  The \code{marginalize} function goes in the opposite
  direction.  It takes one or more dimensions and
  collapses them using a summary function, e.g., sum, max,
  mean, &c.
}
\value{

  The function \code{projectOp} returns an array with
  the same number of dimensions as its inputs, with the
  sizes the maximum of the two dimensions.

  The function \code{marginalize} returns an array or a
  tensor (depending on its input) with the same number of
  dimensions as the input.


}
\references{

  Almond, R. G. (1995) \emph{Graphical Belief Modelling.}
  Chapman and Hall.

}
\author{Russell Almond}
\seealso{

  \code{\link[torch]{torch_add}}, \code{\link[torch]{torch_mul}}
  \code{\link[torch]{torch_sum}}, \code{\link[base]{apply}}

}
\examples{
  mat <- torch_tensor(matrix((1:6)+.1,3,2))
  col <- torch_tensor(matrix(1:2,2,1))
  row <- torch_tensor(matrix(1:3,1,3))

  mat \%^-\% mat
  projectOp(row,col)
  projectOp(col,row,"*")

  arr <- array((1:24)+.1,c(4,3,2))
  marginalize(arr)
  marginalize(arr,2,"max")

}
\keyword{ arrays }
\concept{ potential }

