\name{natpar2Rvec}
\alias{natpar2Rvec}
\alias{Rvec2natpar}
\alias{natpar2tvec}
\alias{tvec2natpar}
\alias{natpar2Rvec.real}
\alias{Rvec2natpar.real}
\alias{natpar2tvec.real}
\alias{tvec2natpar.real}
\alias{natpar2Rvec.pos}
\alias{Rvec2natpar.pos}
\alias{natpar2tvec.pos}
\alias{tvec2natpar.pos}
\alias{natpar2Rvec.real}
\alias{Rvec2natpar.real}
\alias{natpar2tvec.real}
\alias{tvec2natpar.real}
\alias{natpar2Rvec.unit}
\alias{Rvec2natpar.unit}
\alias{natpar2tvec.unit}
\alias{tvec2natpar.unit}
\alias{natpar2Rvec.pVec}
\alias{Rvec2natpar.pVec}
\alias{natpar2tvec.pVec}
\alias{tvec2natpar.pVec}
\alias{natpar2Rvec.cpMat}
\alias{Rvec2natpar.cpMat}
\alias{natpar2tvec.cpMat}
\alias{tvec2natpar.cpMat}
\alias{natpar2Rvec.const}
\alias{Rvec2natpar.const}
\alias{natpar2tvec.const}
\alias{tvec2natpar.const}
\alias{natpar2Rvec.incrK}
\alias{Rvec2natpar.incrK}
\alias{natpar2tvec.incrK}
\alias{tvec2natpar.incrK}
\title{Transform between natural matrix and real vector parameters}
\description{

  Parameters exist in two forms, a natural form which is used in
  constructing rules and links, and a compact vector form used in
  optimization, the `Rvec` or `tvec` (the latter for tensors).  The vector
  form is transformed so that its range is the real numbers and it
  can be used with unconstrained optimization.

}
\usage{
natpar2Rvec(pType, natpar)
Rvec2natpar(pType, Rvec)
natpar2tvec(pType, natpar)
tvec2natpar(pType, Rvec)
\method{natpar2Rvec}{real}(pType, natpar)
\method{Rvec2natpar}{real}(pType, Rvec)
\method{natpar2tvec}{real}(pType, natpar)
\method{tvec2natpar}{real}(pType, Rvec)
\method{natpar2Rvec}{pos}(pType, natpar)
\method{Rvec2natpar}{pos}(pType, Rvec)
\method{natpar2tvec}{pos}(pType, natpar)
\method{tvec2natpar}{pos}(pType, Rvec)
\method{natpar2Rvec}{unit}(pType, natpar)
\method{Rvec2natpar}{unit}(pType, Rvec)
\method{natpar2tvec}{unit}(pType, natpar)
\method{tvec2natpar}{unit}(pType, Rvec)
\method{natpar2Rvec}{pVec}(pType, natpar)
\method{Rvec2natpar}{pVec}(pType, Rvec)
\method{natpar2tvec}{pVec}(pType, natpar)
\method{tvec2natpar}{pVec}(pType, Rvec)
\method{natpar2Rvec}{cpMat}(pType, natpar)
\method{Rvec2natpar}{cpMat}(pType, Rvec)
\method{natpar2tvec}{cpMat}(pType, natpar)
\method{tvec2natpar}{cpMat}(pType, Rvec)
\method{natpar2Rvec}{const}(pType, natpar)
\method{Rvec2natpar}{const}(pType, Rvec)
\method{natpar2tvec}{const}(pType, natpar)
\method{tvec2natpar}{const}(pType, Rvec)
\method{natpar2Rvec}{incrK}(pType, natpar)
\method{Rvec2natpar}{incrK}(pType, Rvec)
\method{natpar2tvec}{incrK}(pType, natpar)
\method{tvec2natpar}{incrK}(pType, Rvec)
}
\arguments{
  \item{pType}{An object of class \code{\linkS4class{PType}} which
    describes the parameter class.}
  \item{natpar}{A numeric object (\code{natpar2Rvec}) or tensor
    (\code{natpar2tvec}) giving the natural parameter.}
  \item{Rvec}{An R vector (\code{Rvec2natpar}) or tensor (\code{tvec2natpar})
    in transformed form.}
}
\details{

 The function \code{natpar2Rvec} (and \code{natpar2tvec}) first calls
 \code{\link{pMat2pVec}} (\code{\link{pMat2pVec10}}) to remove unused
 (see \code{\link{whichUsed}}) parameters and turn it into a vector,
 then applies a transformation which varies by the type of the \code{pType}
 argument.  The function {\code{Rvec2natpar}} ({\code{tvec2natpar}}) applies the inverse
 transformation and \code{\link{pVec2pMat}} (\code{\link{pVec2pMat10}}) to
 restore the shape.


  These functions dispatch on the type of the \code{pType} (see
  \code{\linkS4class{PType}}) to determine the transformation.

  \describe{
    \item{real}{These parameters are not transformed, so the (inverse)
      transformation is the identity.}
    \item{pos}{The natural parameter is positive, the transformation
      is the log; inverse is the exponential.}
    \item{unit}{The natural parameter is between 0 and 1.  The
      transformation is the logistic (\code{\link{logit}})
      and the transformation is the \code{\link{invlogit}}. }
    \item{pVec}{This is a vector which sums to one.  The inverse
      transformation is \code{\link{softmax}} and the transformation is
      log.}
    \item{cpMat}{Each row of the matrix sums to one.  It uses log/softmax
      on each row.  It uses \code{\link{rowlist2pMat}} and
      \code{\link{vec2rowlist}} to operate row-by-row.}
    \item{const}{These return null as there is no parameter.}
    \item{incrK}{Here the natural parameter is increasing in columns, so the
      transformation is \code{\link{ecusum}}.  It uses \code{\link{collist2pMat}} and
      \code{\link{vec2collist}} to operate column-by-column.}
  }

  In the \code{\linkS4class{CombinationRule}}, these functions are used to
  translate between \code{$aMat} and \code{$bMat} (natural parameters) and
  \code{$aVec} and \code{$bVec} (real vector parameters).

}
\value{

  The function \code{natpar2Rvec} returns a vector of real numbers.

  The function \code{Rvec2natpar} returns the natural parameter, often as a matrix.

  The functions \code{natpar2tvec} and \code{tvec2natpar} return tensors.
}
\author{Russell Almond}
\seealso{

  \code{\linkS4class{PType}}, \code{\linkS4class{CombinationRule}},
  \code{\linkS4class{CPT_Link}}.

}
\examples{

  pt <- setpTypeDim(PType("pos"),K=3,J=2)
  pv <- (-2:3)+.1
  pm <- matrix(exp(pv),3,2)
  natpar2Rvec(pt,pm)
  Rvec2natpar(pt,pv)

  pt1 <- PType("real")
  pt1 <- setpTypeDim(pt1,K=3,J=2)
  whichUsed(pt1) <-  matrix(c(TRUE,FALSE,TRUE,FALSE,TRUE,TRUE),3,2)
  natpar2Rvec(pt1,matrix(1:6,3,2))

}
\keyword{ math }
\concept{ transformation }
