\name{Transformations}
\alias{transformations}
\alias{logit}
\alias{invlogit}
\alias{cloglog}
\alias{invcloglog}
\alias{ldiff}
\alias{lldiff}
\alias{ecusum}
\alias{eecusum}
\alias{stickbreak}
\alias{invstickbreak}
\alias{softmax}
\title{Transformations to make Parameter Spaces Span Real Numbers}
\description{

  Optimizers and simulation algorithms (e.g., Hamiltonian Monte Carlo)
  often behave better if the parameters are defined over
  \eqn{\mathbb{R}^n} instead of over a subset (say
  \eqn{(\mathbb{R}^{+})^n} or \eqn{\mathbb{1}^n}).  These
  transformations are designed to move back and forth from the
  restricted natural parametner space and \eqn{\mathbb{R}^n}.

}
\usage{
logit(p)
invlogit(x)
cloglog(p)
invcloglog(x)
ldiff(v)
ecusum(vv)
lldiff(v)
eecusum(vv)
stickbreak(p)
invstickbreak(q)
softmax(x)
}
\arguments{
  \item{p}{A vector of probabilities (between 0 and 1)}
  \item{x}{A vector of real numbers}
  \item{v}{A vector of increasing numbers}
  \item{vv}{A vector of the initial number and log differences.}
  \item{q}{A vector of break proportions.}
}
\details{

  The idea is to create pairs of transformations one of which goes from
  the natural parameter space to \eqn{\mathbb{R}^n} and the other
  reverses the transformation.

  \em{Real Parameters.}  \code{identity()}--\code{identity()}.
  If the natural space of the parameter is the
  real line, the correct transformation and its inverse are the
  \code{\link[base]{identity}()} function.

  \em{Positive Parameters.}  \code{log()}--\code{exp()}.  If the
  parameter must be strictly greater than zero, then logs and antilogs
  satisfy the criteria.  Again, the functions from the `base` package
  are satisfactory.

  \em{Unit Parameters (1).} \code{logit()}--\code{invlogit()}.  There are
  several possible transformations that fit the bill.  The logistic
  function and its inverse are commonly used.  [Note that the 1.7 scale
  factor is not included here.]

  \deqn{\mathrm{logit} (p) = \log\left(\frac{p}{1-p}\right)}

  \deqn{\mathrm{invlogit} (x) = \frac{1}{1+e^{x}}}

  \em{Unit Parameters (2).} \code{cloglog()}--\code{invcloglog()}.
  Another possible transformation is the complementary log-log.

  \deqn{\mathrm{cloglog} (p) = \log\left(-\log(1-p)\right)}

  \deqn{\mathrm{invcloglog} (x) = 1-\exp(-\exp(x))}

  \em{Increasing Real Numbers}.  \code{ldiff()}--\code{ecusum()}.  If
  the series is strictly increasing, the differences must be strictly
  positive.  So the transformed parameter is the first element, followed
  by the log of the differences.  The inverse takes antilogs and the
  calculates the cumulative sum.

  \em{Increasing Positive Real Numbers}.
  \code{lldiff()}--\code{eecusum()}. 
  The difference between this pair and the previous one is that
  the log of the initial value is also taken so that the it will be
  strictly positive as well.

  \em{Unit Simplex (1).}
  \code{logit(stickbreak(p))}--\code{invstickbreak(invlogit(x))}.
  A unit simplex is a vector of non-negative numbers such that
  \eqn{\sum_k p_k =1}.  Start with a stick of unit length and at each
  stage, break off a piece of length \eqn{p_k}, and let \eqn{q_k} be the
  remaining fraction of the stick at this point in time (note, that other
  definitions of the stick breaking process use \eqn{1-q_k} as their
  parameter.  As \eqn{q_K} is always zero, the resulting vector has one
  length of 1 less than the input vector.

  \deqn{p_{k}=(1-q_k) \prod_{kk < k} q_{kk} ;}

  with \eqn{p_K = 1-\prod q_k}.  Note that the \eqn{q} values are
  constrained to the unit interval, so an additional logit is need to
  get to the real numbers.

  \em{Unit Simplex(2)}. \code{log(p)}--\code{softmax(x}}.  The
  cannoncial distribution over a simplex is the Dirichlet distribution.
  One of the better ways of generating a Dirichlet is to generate
  \eqn{K} gamma variates with the same scale parameter (and the shape
  paremeter equal to the corresponding Dirichlet parameter) and then
  normalize the vector by dividing by the sum.
  
}
\value{

  The following transformations go from a restricted parameter space to
  the real numbers: \code{identity()}, \code{log()}, \code{logit()},
  \code{cloglog()}, \code{ldiff()}, and \code{\lldif()}.


  The domains of the inverse transformations are as follows:
  \describe{
    \item{\code{identity()}.}{Real numbers.}
    \item{\code{exp()}.}{Positive real numbers.}
    \item{\code{invlogit()}.}{Unit interval.}
    \item{\code{invcloglog()}.}{Unit Interval.}
    \item{\code{ecumsum()}.}{Increasing vector of real numbers.}
    \item{\code{eecumsum()}.}{Increasing vector of positive real
      numbers.}
    \item{\code{softmax()}.}{Unit simplex.}

    The function \code{stickbreak()} takes values on the unit simplex
    and returns a vector with one few elements of values on the unit
    interval.  The funciton \code{invstickbreak()} does the reverse.
    

}
\references{
  Documentation for the stick break algorithm can be found, among other
  places, in the manual for the `nimble` package at
  \url{https://r-nimble.org}. 
}
\author{Russell Almond}
\note{

  The operators which work on vectors do not handle missing values
  gracefully.  These need to be purged before being put to work.

}
\seealso{
  \code{\link{pMat2pVec}}, \code{\link{natpar2Rvec}}, \code{\link{Rvec2natpar}}
}
\examples{

logit(.7)
cloglog(.7)
invlogit(1.5)
invcloglog(1.5)

softmax(log(1:3))

ldiff(1:4)
lldiff(1:4)
ecusum(rep(0,4))
eecusum(rep(0,4))

stickbreak((1:3)/6)
invstickbreak(c(5/6,3/5))

}
\keyword{ math }
\concept{ transformation }
