\name{fit2table}
\alias{fit2table}
\title{Fits a CPT_Model to a Table of Data}
\description{

  This runs the m-step of an EM-algorithm, using an
  optimizer to adjust the model parameters of a
  \code{\linkS4class{CPT_Model}} to make the
  closest fit to the data table.

}
\usage{
fit2table(model, dattab, maxit = 100L, tolerance = 1e-04, stepit = 1L, log = NULL)
}
\arguments{
  \item{model}{A \code{\linkS4class{CPT_Model}} whose
    parameters are to be adjusted.  Note:  this argument
    is destructively modified.}
  \item{dattab}{A table of data of the same size as
    the CPT to which the model is to be fit.}
  \item{maxit}{The maximum number of optimizer cycles
    to run.}
  \item{tolerance}{Test criteria for convergence.}
  \item{stepit}{The number of steps to take between
    logging cycles.}
  \item{log}{What, if anything to log.  Could be \dQuote{cpt}
    for the CPT, or one or more of the parameters \dQuote{aVec},
    \dQuote{bVec}, \dQuote{sVec}, \dQuote{sP}, or \dQuote{gP}.
    If \code{NULL}, no logging is done.}
}
\details{

  This function runs the model optimizer to fit a
  \code{\linkS4class{CPT_Model}} to a \code{datatab}.

  The optimizer runs, taking \code{stepit} cycles
  at each step until either the change in deviance is
  less than \code{tolerance} or \code{maxit} cycles
  have been reached.

  If logging is requested, it is done every \code{stepit}
  cycles.  The logging will trace any combination of the cpt and
  any of the parameters.

}
\value{

  Returns a logical value, \code{TRUE} if convergence was
  reached and \code{FALSE} if \code{maxit} cycles
  elapsed before convergence.

  Note:  Destructively modifies the \code{model} argument.

}
\references{

  Almond, R. (2015). An IRT-based Parameterization
  for Conditional Probability Tables. In J. M. Agosta &
  R. N. Carvalho (Eds.), \emph{Proceedings of the Twelfth
  UAI Bayesian Modeling Applications Workshop (BMAW 2015)}
  (Vol. 1565, pp. 14–23).
  \url{http://ceur-ws.org/Vol-1565/bmaw2015_paper4.pdf}

}
\author{Russell Almond}
\seealso{
  \code{\linkS4class{CPT_Model}},
  \code{\link{deviance_loss}}
}
\examples{

\dontrun{
  mod0 <- CPT_Model$new("Compensatory","PartialCredit",
                        list('Par1'=c('P1_low', 'P1_hi'), 'Par2'=c('P2_low', 'P2_hi')),
                        c("A","B","C"))
  mod0$penalties <- list("bVec"=1,"sVec"=1)
  mod1 <- CPT_Model$new("Compensatory","PartialCredit",
                        list('Par1'=c('P1_low', 'P1_hi'), 'Par2'=c('P2_low', 'P2_hi')),
                        c("A","B","C"))
  mod1$bMat <- matrix(c(0,.5),2,1)

  cpt1 <- mod1$getCPT()
  dattab <- torch_mul(cpt1,1000)

  conv <- fit2table(mod0,dattab,stepit=5,log=c("bVec","sVec","cpt"), maxit=100L)
}
}
\keyword{ optimize }
\concept{ CPT }

