\name{defaultParameter}
\alias{defaultParameter}
\alias{defaultParameter.incrK}
\alias{defaultParameter10}
\alias{getZero}
\title{Gets an initial value for a parameter.}
\description{

  This creates a default parameter which follows the rules of the the
  \code{\linkS4class{PType}}.
}
\usage{
defaultParameter(pType)
\method{defaultParameter}{incrK}(pType)
defaultParameter10(pType,device=CPTtorch::CPTtorch_device())
getZero(pType)
}
\arguments{
  \item{pType}{An object of class \code{\linkS4class{PType}} giving
    the type of parameter.}
  \item{device}{The \code{\link[torch]{torch_device}} to use when building
    the tensor.  Defaults to \code{\link{CPTtorch_device}()}.}
}
\details{
  The function \code{getZero()} returns the value which should be taken as
  a natural zero for this parameter type.

  If the shape of the parameter has been established through a call to
  \code{\link{setpTypeDim}}, then \code{defaultParameter} fills an array
  of the proper shape with zeros. The method \code{defaultParameter.incrK}

}
\value{
  The function \code{getZero()} returns a number.
  The function \code{defaultParameter()} returns either a number or
  an array if the dimensions are set; \code{defaultParameter10} returns
  a tensor.

}
\author{Russell Almond}
\seealso{
  \code{\linkS4class{PType}}, \code{\link{checkParam}}
  \code{\link{setpTypeDim}}
}
\examples{

  pt <- PType("real")
  getZero(pt)
  defaultParameter(pt)

  pt <- setpTypeDim(pt,K=3,J=2)
  defaultParameter(pt)

  pt <- setpTypeDim(PType("incrK",dim=c(K,J)),K=3,J=2)
  getZero(pt)
  defaultParameter(pt)


}
\keyword{ attributes }
\concept{ PType }
