\name{pMat2pVec}
\alias{pMat2pVec}
\alias{pMat2pVec10}
\alias{pMat2rowlist}
\alias{pMat2collist}
\alias{list2vec}
\alias{pVec2pMat}
\alias{pVec2pMat10}
\alias{collist2pMat10}
\alias{vec2rowlist}
\alias{vec2collist}
\alias{list2vec}
\title{Transform matrixes to vectors and back.}
\description{

  Transforms from the matrix form of a parameter to the vector form or back.
  Parameters which are not used (see \code{\link{whichUsed}}) are eliminated from
  the vector form.

}
\usage{
pMat2pVec(pType, pMat)
pMat2pVec10(pType, pMat)
pMat2rowlist(pType, pMat)
pMat2collist(pType, pMat)
list2vec(plist)
pVec2pMat(pType,pVec)
pVec2pMat10(pType,pVec)
collist2pMat10(pType,clist)
vec2rowlist(pType,pVec)
vec2collist(pType,pVec)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{pType}{An object of class \code{\linkS4class{PType}} which gives the
    dimensions of the matrix as well as unused cels.}
  \item{pMat}{A matrix form parameter.}
  \item{pVec}{A vector form parameter.}
  \item{plist}{A list of vectors to combine.}
  \item{clist}{A list of column vectors to combine}
}
\details{

  The functions \code{pMat2pVec}, \code{pMat2pVec10} take the matrix form of a
  parameter and flatten it.  The functions \code{pVec2pMat}, \code{pVec2pMat10}
  do the inverse.  When flattening a matrix, cells of the matrix corresponding
  to \code{\link{whichUsed}(pType)=FALSE} are eliminated from the vector, and
  are filled with default values when going back to matrix form.

  When the \code{pType} argument is of the class \code{cpMat} the transformation
  is done row-by-row.  So the functions \code{pMat2rowlist} and
  \code{pVec2rowlist} are used along with the function \code{list2vec} to
  collapse the separate vectors into a single vector.

  When the \code{pType} argument is of the class \code{incrK} the transformation
  is done column-by-column.  So the functions \code{pMat2collist} and
  \code{pVec2collist} are used along with the function \code{collist2pMat}
  to recombine the vectors.
}
\value{

  The functions \code{pMat2pVec} and \code{list2vec} return vectors, with
  \code{pMat2pVec10} returning a one-dimensional tensor.

  The function \code{pVec2pMat} returns a matrix, with
  \code{pVec2pMat10} and \code{collist2pMat10} returning a
  two-dimensional tensors.

  The functions \code{pMat2rowlist}, \code{pMat2collist},
  \code{vec2rowlist} and \code{vec2collist} all return lists of vectors.

}
\author{Russell Almond}
\seealso{
  \code{\link{natpar2Rvec}} explains the transformation part of the conversion
  from natural to real vector parameters.

  \code{\linkS4class{PType}}
}
\examples{
  pt <- setpTypeDim(PType("pos",dim=c(2,3)))
  natpar <- matrix(1:6,2,3)
  qq <- matrix(c(FALSE,TRUE,TRUE,TRUE,TRUE,FALSE),2,3,byrow=TRUE)

  pMat2pVec(pt,natpar)

  whichUsed(pt) <- qq
  pv <- pMat2pVec(pt,natpar)
  pVec2pMat(pt,pv)

  rl <- pMat2rowlist(pt,natpar)
  pvv <- list2vec(rl)
  vec2rowlist(pt,pvv)

}
\keyword{ mathematics }
\concept{ transformations }
