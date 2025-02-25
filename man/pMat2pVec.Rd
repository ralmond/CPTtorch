\name{pMat2pVec}
\alias{pMat2pVec}
\alias{pMat2rowlist}
\alias{pMat2collist}
\alias{list2vec}
\alias{pVec2pMat}
\alias{vec2rowlist}
\alias{vec2collist}
\alias{list2vec}
\title{Transform matrixes to vectors and back.}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
pMat2pVec(pType, pMat)
pMat2rowlist(pType, pMat)
pMat2collist(pType, pMat)
list2vec(plist)
pVec2pMat(pType,pVec)
vec2rowlist(pType,pVec)
vec2rowlist(pType,pVec)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{pType}{
%%     ~~Describe \code{pType} here~~
}
  \item{pMat}{
%%     ~~Describe \code{pMat} here~~
}
  \item{pVec}{
%%     ~~Describe \code{pMat} here~~
}
  \item{plist}{
%%     ~~Describe \code{pMat} here~~
}

}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or standard data sets, see data().

## The function is currently defined as
function (pType, pMat) 
{
    if (isTRUE(whichUsed(pType)) || is.null(pTypeDim(pType))) 
        as.vector(pMat)
    else as.vector(pMat[whichUsed(pType)])
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
