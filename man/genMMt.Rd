\name{genMMt}
\alias{genMMt}
\alias{genMMt.tt}
\alias{genMMt.matrix}
\alias{genMMt,matrix,matrix-method}
\alias{genMMt,torch_tensor,torch_tensor-method}
\alias{genMMtQ}
\alias{genMMtQ.tt}
\alias{genMMtQ.matrix}
\alias{genMMtQ,matrix,matrix,matrix-method}
\alias{genMMtQ,torch_tensor,torch_tensor,matrix-method}
\alias{genMMtQ,torch_tensor,torch_tensor,torch_tensor-method}
\title{Generalized Matrix Multiplication}
\description{

  Matrix multiplication uses multiplication as a combination
  operation and sum as a summary operation.  The function \code{genMMt}
  allows substituting any operation for multiplication and
  sum.  The function \code{genMMtQ} also removes entries
  corresponding FALSE cells in the Q-matrix.

}
\usage{
genMMt(m1, m2, combOp, summaryOp)
\S4method{genMMt}{matrix,matrix}(m1, m2, combOp, summaryOp)
genMMt.matrix(m1, m2, combOp, summaryOp)
\S4method{genMMt}{torch_tensor,torch_tensor}(m1, m2, combOp, summaryOp)
genMMt.tt(m1, m2, combOp, summaryOp)
genMMtQ(m1, m2, QQ, combOp, summaryOp)
\S4method{genMMtQ}{matrix,matrix,matrix}(m1, m2, QQ, combOp, summaryOp)
genMMtQ.matrix(m1, m2, QQ, combOp, summaryOp)
\S4method{genMMtQ}{torch_tensor,torch_tensor,matrix}(m1, m2, QQ, combOp, summaryOp)
\S4method{genMMtQ}{torch_tensor,torch_tensor,torch_tensor}(m1, m2, QQ, combOp, summaryOp)
genMMtQ.tt(m1, m2, QQ, combOp, summaryOp)
}
\arguments{
  \item{m1,m2}{The matrixes to be combined. Both matrixes
    should have the same number of rows.  These should both be
    R matrixes or tensors}
  \item{QQ}{A logical matrix with number of rows equal
    to the number of rows of \code{m2} and columns
    equal to the columns of \code{m2}.  Cells that arer false
    are not included in the result.  This can be either a
    tensor or an R matrix.}
  \item{combOp}{The operator to use when combining two rules.
    Should be a binary operator, like \code{*}.}
  \item{summaryOp}{The operator to use when summarizing cells.
    Should take a vector to a scalar, like \code{sum}.}
}
\details{

  The function \code{genMMt} does generalized matrix
  multiplication, however, the second argument is transposed,
  so that it is the number of rows in each input matrix
  that must match.  Let \eqn{\bigoplus} be the summary operator
  and \eqn{\odot} be the combination operator and \eqn{\otimes}
  refer to the combined operation.  Then the output matrix
  is

  \deqn{(m1\otimes m2)[j,k] = \bigoplus_{i}
    m1[i,j] \otimes m2[i,k] .}

  If a Q-matrix is supplied, it indicates that certain cells,
  corresponding to the FALSE cells in the Q-matrix, are not
  to be included.  So for \code{genMMtQ} has the equation:

  \deqn{(m1\otimes m2)[j,k] = \bigoplus_{i:Q[k,i]==TRUE}
    m1[i,j] \otimes m2[i,k] .}

}
\value{

  Returns a matrix or a tensor, depending on the
  type of the inputs \code{m1} and \code{m2}.

}
\author{Russell Almond}
\seealso{
  \code{\linkS4class{CombinationRule}}
}
\examples{
  et <- cbind(rep(c(-1,0,1),3),rep(c(-1,0,1),each=3))
  a1 <- matrix(c(1,2),1,2)
  a2 <- matrix(c(1,0,0,1),2,2)
  genMMt(et,a1,"*","sum")
  genMMt(et,a2,"+","max")
  genMMt(torch_tensor(et),torch_tensor(a1),
    torch_mul,torch_sum)

  a3 <- matrix(1,3,2)
  qq <- matrix(c(TRUE,FALSE,TRUE, FALSE,TRUE,TRUE),3,2)
  genMMtQ(et,a3,qq,"*","sum")


}
\keyword{ mathematics }
\keyword{ matrix }
\concept{ matrix multiplication }
\concept{ Q-matrix }

