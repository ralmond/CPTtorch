\name{setpTypeDim}
\alias{setpTypeDim}
\alias{pTypeDim}
\title{Gets or sets the dimensions of the parameter.}
\description{

  When \code{\linkS4class{PType}}s are declared for
  \code{\linkS4class{CombinationRule}}s, the final dimensions are
  not yet known.  Instead they are declared in terms of variables:
  \var{J} (number of parents), \var{S} (number of parent configurations),
  and \var{K} (number of child states).  The function \code{setpTypeDim}
  resolves the values of the variables and produces the final dimensions,
  which can be retrieved with \code{pTypeDim}.

}
\usage{
pTypeDim(pType)
setpTypeDim(pType, J = 1L, S = 2L^J, K = 1L)
}
\arguments{
  \item{pType}{An object of class \code{\linkS4class{PType}}.}
  \item{J}{A numeric value to be substituted for the number of parent variables.}
  \item{S}{A numeric value to be substitued for the number of parent configurations.}
  \item{K}{A numeric value to be substituted for the number of child states.  This
    is generally found through a call to \code{\linkS4class{CPT_Link}$etWidth()}.}
}
\details{

  The needed dimensions of the parameter are not known until the parent variables.
  In the \code{\linkS4class{PType}} constructor, the \code{dim} argument is not
  evaluated, so it can contain free variables, \var{J}, \var{S} and \var{K}.
  The function \code{setpTypeDim} fixes the value of the free variables, so that
  the \code{$dim} field is set.

  The meaning and default values of the free variables is as follows:

  \describe{
    \item{J}{The number of parent variables, this generally comes from the
      length of the \code{Tvallist} of the \code{\linkS4class{CombinationRule}}
      class.}
    \item{S}{The number of parent configurations, this generally comes from the
      expanding the grid of the \code{Tvallist} of the \code{\linkS4class{CombinationRule}}
      class.}
    \item{K}{This depends on both the number of child states and the specific
      link function.  Normally, this is number of child states minus 1 (for
      normalization), but it varies.  The method \code{\linkS4class{CPT_Link}$etWidth()}
      gives the proper default value for a given link function.}
  }
}
\value{

  The function \code{pTypeDim} returns a vector of dimensions, or NULL if
  it has not been set yet.

  The function \code{setpTypeDim} returns a modified copy of the \code{pType}
  argument.  Following the usual R copy on modify convention, this often needs
  to replace the original argument.
}
\author{Russell Almond}
\seealso{
  \code{\linkS4class{PType}}, \code{\linkS4class{CombinationRule}},
  \code{\linkS4class{CPT_Link}}
}
\examples{

  pt <- PType("real",dim=c(K,J))
  pTypeDim(pt)

  pt1 <- setpTypeDim(pt,K=3,J=2)
  pTypeDim(pt1)
  pTypeDim(pt)

}
\keyword{ attributes }
\concept{ PType }
