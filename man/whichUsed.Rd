\name{whichUsed}
\alias{whichUsed}
\alias{whichUsed<-}
\title{Get set matrix of used parameter elements.}
\description{

  Depending on the configuration, only certain cells of the parameter
  matrix are used.  This can be specified with a logical matrix \eqn{Q},
  which is the same dimension as the parameter, \eqn{A}.  Then
  \eqn{a_{k,j}} is used if and only if \eqn{q_{k,j}} is true.  In place
  of a matrix, the scalar value \code{TRUE} can be used to indicate that
  all parameters are used.

}
\usage{
whichUsed(pType)
whichUsed(pType) <- value
}
\arguments{
  \item{pType}{A \code{\linkS4class{P4Type}} object describing the paramter.}
  \item{value}{A logical matrix of the same dimensions as the parameter
    or the scalar value \code{TRUE}.}
}
\details{

  Consider a math problem with two parts that is graded on a partial
  credit scale, so that the three possible scores are 0, 1 and 2 (no,
  full and partial credit) depending on how many parts are completed.
  The first part requires Skills 1 and 2, and the second part Skills 2
  and 3.  Let

  \deqn{Q = \begin{matrix}F, T, T\cr T, T, F\end{matrix}}

  With the partial credit link function, the first column of the
  effective theta matrix is the probability of solving the second part
  correctly, given that the first part was solved correctly, and the
  second column is the probability of solving the first part correctly.
  Assume further, that we are using the compensatory combination rule.
  The value of the output matrix of the combination rule is then:

  \deqn{\tilde{\theta}_{s,k} = \sum_{j : q_{k,j}} a_{k,j} \theta_{s,j} +
    b_{k,1} \ .}

  Note that the function \code{\link{genMMtQ}} does this using arbitrary
  operators, and takes the value of \code{whichUsed(pType)} as the
  \code{QQ} argument.

  Note that generally if a particular parameter cell is unused, it
  should not be passed to the optimizer or sampler for model fitting.
  For that reason, the \code{\link{natpar2Rvec}} method eliminates the
  elements corresponding to the false elements of are eliminated from
  the vector version.  Its inverse, \code{\link{Rvec2natpar}} rebuilds
  the matrix view leaving the unused values undefined (or set as NAs).

  When implementing these functions, the helpers function
  \code{\link{pMat2pVec}} removes the unused value from the vector, so
  the length of the resulting vector is \code{sum(whichUsed(pType))}.
  The converse function \code{\link{pVec2pMat}} first fills a matrix of
  the correct dimensions, and then sets only the cells correponding to
  true values of \code{whichUsed(pType)}.

}
\value{

  A matrix of dimensions \code{\link{pTypeDim}(pType)} of logical values
  or the scalar value \code{TRUE} (indicating that all elements of the
  matrix should be considered true).

}
\references{

  This is related to the idea of the \dQuote{Inner Q-matrix} appears in
  a conference talk:

  Almond, R. G. (2018). Inner and Outer Q-matrixes for Multi-step
  Problems. \emph{Annual Meeting of the American Educational Research
    Association.}

  Unfortunately, while slides exist for this talk, there is no written
  paper.

  The ideas of that presentation were built into the \code{CPTtools}
  package, and the vignettes there are possibly the best explanation
  currently avaiable.

}
\author{Russell Almond}
\seealso{

  \code{\linkS4class{PType}}, \code{\link{natpar2Rvec}},
  \code{\link{pMat2pVec}}

}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or standard data sets, see data().

## The function is currently defined as
function (pType)
{
    return(pType$used)
  }
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
