\name{deviance_loss}
\alias{deviance_loss}
\alias{penalty_fun}
\title{Components of the Loss Function for CPTs}
\description{

  The function \code{deviance_loss} calculates the
  deviance of a data table from data generated
  from the CPT.  The \code{penalty_fun} adds a penalty
  for non-zero parameters.
}
\usage{
deviance_loss(datatab, cpt, ccbias = 0, bin_eps = 2^-35)
penalty_fun(params,which,bias,device=CPTtorch::CPTtorch_device())
}
\arguments{
  \item{datatab}{A table of data the same shape as the
    \code{cpt} which nominally came from the cpt.}
  \item{cpt}{The conditional probability distribution.}
  \item{ccbias}{The quantity \code{ccbias*cpt} is added
    to \code{datatab} to add a small bias towards the
    prior.}
  \item{bin_eps}{A value that is added to \code{cpt}
    to avoid taking log of zero.}
  \item{params}{A list of parameter vectors}
  \item{which}{An index into the list of parameter vectors
    telling which one to use.  Should be one of
    \dQuote{aVec}, \dQuote{bVec},
  \dQuote{sVec}, \dQuote{sP}, or \dQuote{gP}.}
  \item{bias}{The penalty applied to non-zero parameter.}
  \item{device}{A \code{\link[torch]{torch_device}}
    telling where to build the tensor if parameters
    are null.}
}
\details{

  Let \eqn{X} represent \code{datatab+ccbias*cpt}
  and \eqn{P} represent \code{cpt}.  Then the
  \code{deviance_loss} is given by
  \deqn{-2 \sum_{i,j} X_{i,j} \log P_{i,j} .}

  The \code{penalty_fun} is
  \code{bias*sum(params[[which]]^2)}.

  To request penalties for a CPT, use an expression like
  \code{model$penalties <- list("bVec"=1,"sVec"=1)}, where
  the names should be in the set \dQuote{aVec}, \dQuote{bVec},
  \dQuote{sVec}, \dQuote{sP}, \dQuote{gP}.


  A \code{\linkS4class{CPT_Model}} uses the sum
  of the deviance loss and optionally penalties for
  the various parameters as the loss function when
  fitting a CPT to a table (\code{\link{fit2table}}).
}
\value{

  A real value tensor giving the loss/penalty.

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
  \code{\linkS4class{CPT_Model}}, \code{\link{fit2table}}
}
\examples{

  cpt3 <- CPT_Model$new("Compensatory","PartialCredit",
                        list(A=c("A1","A2","A3"),B=c("B1","B2","B3")),
                        c("C1","C2","C3"))
  aa <- matrix(c(1.1,.9,.9,1.1),2,2)
  cpt3$aMat <- aa
  bb <- matrix(c(-.5,.5),2,1)
  cpt3$bMat <- bb

  cpt <- cpt3$forward()
  dattab <- torch_ones_like(cpt)
  deviance_loss(dattab,cpt,1)
  cpt3$deviance(dattab)

  penalty_fun(cpt3$params(),1,10)

}
\keyword{ optimize }
\concept{ deviance }
\concept{ penalty }
\concept{ loss function }

