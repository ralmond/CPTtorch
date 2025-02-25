\name{setpTypeDim}
\alias{setpTypeDim}
\alias{pTypeDim}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Gets or sets the dimensions of the parameter.}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
pTypeDim(pType)
setpTypeDim(pType, J = 1L, S = 2L^J, K = 1L)
}
\arguments{
  \item{pType}{
%%     ~~Describe \code{pType} here~~
}
  \item{J}{
%%     ~~Describe \code{J} here~~
}
  \item{S}{
%%     ~~Describe \code{S} here~~
}
  \item{K}{
%%     ~~Describe \code{K} here~~
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
function (pType, J = 1L, S = 2L^J, K = 1L) 
{
    pType$dim <- eval(rlang::inject(pType$dimexpr))
    PType
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
