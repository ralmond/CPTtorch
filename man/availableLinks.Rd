\name{availableLinks}
\alias{availableLinks}
\alias{getLink}
\alias{setLink}
\alias{DifferenceLink}
\alias{GaussianLink}
\alias{GradedResponseLink}
\alias{PartialCreditLink}
\alias{PotentialLink}
\alias{SlipLink}
\alias{SoftmaxLink}
\alias{StepProbsLink}
\title{Available CPT_Links and their Names}
\description{
  The function \code{availableLinks} lists the available
  link functions.  The function \code{getLink} fetches the CPT_Link
  object by name, and \code{setLink} registers a new link function.
}
\usage{
availableLinks()
getLink(linkname)
setLink(linkname,value)
}
\arguments{
  \item{linkname}{A character scalar giving the name of the link function.}
  \item{value}{A \code{\linkS4class{CPT_Link}} object
    to be registered.}
}

\details{

  Link functions translate the output of the
  \code{\linkS4class{CombinationRule}}, the effective theta,
  into a conditional probability distribution.  The target width of
  the effective is controlled by the \code{$etWidth} function
  of the link which is a function of \eqn{K}, the number of child states.

  The following link functions are available by default:

  \describe{
    \item{Potential"}{The effective theta is the conditional
      probability distribution.  The \code{$etWidth} is \eqn{K}.
      The only transformation is
      ensuring that the rows sum to one (see
      \code{\link{torch_simplexify}}). There is no link scale
      parameter.}
    \item{StepProbs"}{The effective theta is the probability of
      moving from one step to the next.  The \code{$etWidth} is
      \eqn{K-1}.  The transformation looks at the cumulative product
      of probabilities.  There is no link scale parameter.}
    \item{Difference"}{The effective theta is the probability of being
      at or above a certain state.  The \code{$etWidth} is \eqn{K-1}.
      The transformation is \code{\link{cuts2simplex}}.  There is no link
      scale parameter.}
    \item{"Softmax"}{The effective theta is the log of the probability
      of being in each category.  The \code{$etWidth} is \eqn{K}.
      The transformation is \code{\link{softmax}} and uses the constant \eqn{D=1.7}.
      There is no link scale parameter.}
    \item{GradedResponse"}{The effective theta is the logit of the probability
      of being in that category or above.  The \code{$etWidth} is \eqn{K-1}.
      The transformation is \code{\link{cuts2simplex}} applied to
      the sigmod transformation of the effective theta, and uses the
      constant \var{D=1.7}.  There is no link scale parameter.}
    \item{PartialCredit"}{The effective theta is the logit of the step
      transition probabilities, it uses \eqn{D=1.7}.  The \code{$etWidth}
      is \eqn{K-1}.  There is no log scale parameter.}
    \item{Normal","Gaussian"}{The transformation uses cuts from a normal
      distribution with mean given by the effective theta and standard
      deviation given by the link scale parameter.  The \code{$etWidth}
      is 1.}
  }

  Note that only the \dQuote{Normal} (or \dQuote{Gaussian}) link
  function uses a link scale parameter.

  The logistic links, \dQuote{Softmax}, \dQuote{GradedResponse},
  and \dQuote{PartialCredit}, used the constant \eqn{D=1.7} to make
  the logistic function close to the normal ogive.  This can be set to
  other values (i.e., 1 to use the raw logistic function).

  All of the \code{\linkS4class{CPT_Link}} have slip and guess parameter
  so can serve as a noisy link.
}
\value{
  The function \code{availableLink()} returns a list of registered
  link names.

  The function \code{getLink(name)} returns a
  \code{\linkS4class{CPT_Link}}

  The function \code{setLink} is used for its side effects and
  its return value should not be used.
}
\references{

  Almond, R. (2015). An IRT-based Parameterization for Conditional
  Probability Tables. In J. M. Agosta & R. N. Carvalho (Eds.),
  \emph{Proceedings of the Twelfth UAI Bayesian Modeling Applications
  Workshop (BMAW 2015)} (Vol. 1565, pp. 14–23).
  \url{http://ceur-ws.org/Vol-1565/bmaw2015_paper4.pdf}

  Almond, R. G., Mislevy, R. J., Steinberg, L. S., Williamson, D. M.,
  & Yan, D. (2015). \emph{Bayesian Networks in Educational Assessment.}
  Springer. \url{http://www.springer.com/statistics/social+sciences+\%26+law/book/978-1-4939-2124-9}


}
\author{Russell Almond}
\seealso{

  \code{\link{cuts2simplex}}, \code{\link{softmax}}

  \code{\linkS4class{CPT_Link}},
  \code{\linkS4class{CombinationRule}}

}
\examples{

  availableLinks()
  gl <- getLink("Gaussian")$new(3)

}
\keyword{ classes }
\concept{ CPT_Link }

