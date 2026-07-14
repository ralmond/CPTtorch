\name{availableRules}
\alias{availableRules}
\alias{getRule}
\alias{setRule}
\alias{CompensatoryRule}
\alias{CompensatoryGRRule}
\alias{ConjunctiveRule}
\alias{DisjunctiveRule}
\alias{DirichletRule}
\alias{NoisyAndRule}
\alias{NoisyOrRule}
\alias{CenterRule}
\title{Available Combination Rules and their Names}
\description{

  The function \code{availableRules} lists the available rules.
  The function \code{getRule} fetches the rule object by name,
  and \code{setRule} registers a new rule.

}
\usage{
availableRules()
getRule(name)
setRule(name,value)
}
\arguments{
  \item{name}{A character scalar giving the name of the rule.}
  \item{value}{A \code{\linkS4class{CombinationRule}} object
    to be registered.}
}
\details{

  The currently available rules are:

  \describe{
    \item{CompensatoryRule}{Effective theta is a weighted
      average of the inputs.  Rule has a matrix of weights and a
      column of offsets.}
    \item{CompensatoryGRRule}{Similar to the CompensatoryRule
      except the weights are the same for each row.}
    \item{ConjunctiveRule}{Effective theta is the minimum
      across columns.  Uses a matrix of offsets and a vector of
      weights.}
    \item{DisjunctiveRule}{Effective theta is the maximum
      across columns.  Uses a matrix of offsets and a vector of
      weights.}
    \item{NoisyAndRule}{Each cell becomes a logical value and the
      summary is the product.  Uses a matrix of thresholds and a
      matrix of weights.}
    \item{NoisyOrRule}{Each cell becomes a logical value and the
      summary is the 1- the product of 1-p.  Uses a matrix of thresholds and a
      matrix of weights.}
    \item{DirichletRule}{Each row is a Dirichlet distribution.}
    \item{CenterRule}{Each row is a different value.}
  }

  In general the \var{A} parameter gives weights and \var{B}
  gives offsets.  The rule can combine \var{A} first, \var{B} first
  and do the summary step in the middle or the last.  (See
  \code{\linkS4class{CombinationRule}}.)  The following table
  lays out how the predefined rules operate.



  \tabular{llllccl}{
    Name           \tab  Inherit \tab aType \tab bType \tab aop \tab bop \tab summary \cr
    Compensatory   \tab  ASB \tab pos(K,J) \tab real(K,1) \tab * \tab - \tab sum \cr
    CompensatoryGR \tab  ASB \tab pos(1,J) \tab incrK(K,1) \tab * \tab - \tab sum \cr
    Conjunctive    \tab  BSA \tab pos(K,1) \tab real(K,J) \tab * \tab - \tab min \cr
    Disjunctive    \tab  BSA \tab pos(K,1) \tab real(K,J) \tab * \tab - \tab max \cr
    NoisyAnd       \tab  BAS \tab unit(K,J) \tab real(K,J) \tab * \tab > \tab product \cr
    NoisyOr        \tab  BAS \tab unit(K,J) \tab real(K,J) \tab * \tab <= \tab prod_1 \cr
    Center         \tab  ConstB \tab NULL \tab real(S,K) \tab  \tab  \tab mean \cr
    Dirichlet      \tab  ConstB \tab NULL \tab real(S,K) \tab  \tab  \tab mean \cr
  }

}
\value{

  The function \code{availableRules()} returns a list of registered
  rule names.

  The function \code{getRule(name)} returns a
  \code{\linkS4class{CombinationRule}}

  The function \code{setRule} is used for its side effects and
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
  \code{\linkS4class{CombinationRule}},
  \code{\linkS4class{PType}}
}
\examples{

availableRules()
cr <- getRule("CompensatoryGR")$new(as_Tvallist(list(A=c(0,1),B=c(0,1))),2)

}
\keyword{ classes }
\concept{ CombinationRule }
