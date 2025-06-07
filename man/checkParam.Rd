\name{checkParam}
\alias{checkParam}
\alias{checkParam.default}
\alias{checkParam.PType}
\title{Check to make sure parameter value is legal for a PType}
\description{

  This checks to make sure that the parameter value, vector, or matrix
  fits the restrictions defined in the  \code{\linkS4class{PType}}.  If
  all check pass, then it returns \code{TRUE}, otherwise it returns a
  character scaler describing the failed check.

}
\usage{
checkParam(pType, par)
}
\arguments{
  \item{pType}{An object of class \code{\linkS4class{PType}}.}
  \item{par}{A matrix or vector of parameter values.}
}
\details{

  The method \code{checkParam.PType} (which is called by other methods)
  checks to make sure that the dimensions of the parameter matrix match
  the value of \code{\link{pTypeDim}(pType)}.  Note that if this has not
  been finalized, this will be \code{NULL} and the parameter must be a
  vector without dimensions for the check to pass.  Once the dimensions
  have been set, then the expectation is that it will be a matrix.
}
\value{

  Returns \code{TRUE} if the parameter passes all checks.

  Returns a character value describing a failed check if one check
  fails.  If a logical value is needed, use \code{isTRUE(checkParam())}.

}
\author{Russell Almond}
\section{Writing Methods for New Subclasses.}{

  The implementation the inheritance system of method dispatch to
  partition task of checking the parameter.  Recall that the class of a
  \code{\linkS4class{PType}} is a vector
  \code{c(\var{subclass},"PType")}.  So the method hierarchy will be
  \code{checkParam.\var{subclass}}, \code{checkParam.PType}, and
  \code{checkParam.default}.  The default method (always last in the
  sequence), returns \code{TRUE}.  The other methods should either
  return an error message (character value) or call \code{NextMethod()}
  to run the next set of checks in sequence.  The
  \code{checkParam.PType} method takes care of checking the dimensions
  of the parameter matrix, and is supplied by the package.  So only the
  \code{checkParam.\var{subclass}} needs to be supplied for new
  parameter types.

  If \code{\link{whichUsed}} is set to something other than \code{TRUE},
  the function \code{\link{pMat2pVec}} should be called to eliminate the unused
  cells of the parameter matrix.  If there are constraints on the rows
  or columns they may be of uneven length, often the best strategy is to
  use \code{pMat2rowlist} or \code{pMat2collist} and then iterate the
  checks over rows and columns.  If any check fails, the method should
  return an error message.  If all check succeed, the method should call
  \code{NextMethod()} to pass control to \code{checkParam.PType}.

}
\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or standard data sets, see data().

## The function is currently defined as
function (pType, par)
UseMethod("checkParam")
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
