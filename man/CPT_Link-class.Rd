\name{CPT_Link}
\Rdversion{1.1}
\docType{class}
\alias{CPT_Link-class}
\title{Class \code{"CPT_Link"} -- Calculates CPT from Effective Thetas}
\description{

  This function takes the effective theta tensor, the output of a
  \code{\link{CombinationRule}} and converts it into a conditional
  probability table.  This may require an additional \code{linkScale}
  parameter. 

}
\section{Public Fields}{

  \describe{
    \item{link}{This should be a funciton with signature which takes a
      single tensor as input and produces a tensor as output.}
    \item{sVec}{If the transformation requires and an additional scale
      parameter, this is a \code{\link[torch]{nn_parameter}} which
      contains the vectorized parameter.  Generally, it should not be
      set directly, but rather through the \code{$linkScale} active
      field.}
    \item{D}{Used with the logistic link function.  It should be either
      \code{torch_tensor(1.7)} or \code{torch_tensor(1.0)}.}
  }

  The value of \eqn{D=1.7} is used with the \code{SoftmaxLink}, the
  \code{GradedResponseLink} and the \code{PartialCreditLink}.  This is
  often used in item response theory, as \code{invlogit(1.7*x)} is
  approximately equal to \code{invprobit(x)}.  Different schools of
  thought exist on the wisdom of using the 1.7 constant (I recommend it
  for use with the normal quantile mapping of variable states to real
  values, but setting \code{\var{link}$D <- torch_tensor(1.0)} will
  eleiminate it for compatability with the other common
  parameterization.) 

}
\section{Active Fields}{
  \describe{
    \item{K}{This is an integer value giving the number of columns in
      the outcome variable, which is the same as the number of states of
      the child varaible.}
    \item{sType}{This is either NULL, if there is no scale parameter
      required, or a \code{\link{PType}} describing the scale parameter.}
    \item{linkScale}{This is the link scale parameter in its natural
      representation.}
  }

}
\section{Private Fields}{

  \describe{
    \item{k}{Internal value of \code{$K}, should be set through the
      active field.}
    \item{stype}{The internal value of the \code{$sType} field.  This
      should not be set directly, but can be overridden in subclasses.}
  }

}
\section{Methods}{

  \describe{
    \item{etWidth}{\code{signature()} Returns the expected number of
      columns in the effective theta table.  }
    \item{initialize}{\code{signature(\var{nstates}:int)} initializes
      the link. }
    \item{forward}{\code{signature(\var{input}:torch_tensor)} this does
      the calculation.  The default method calls the \code{$link} function. }
  }

}
\section{Scale Parameter}{

  Some link functions (\code{PotentialLink}, \code{DifferenceLink},
  \code{StepProbLin},  \code{GradedRepsonseLink}, 
  \code{PartialCreditLink}, and \code{SoftmaxLink}) do not need any
  additional information besides what is in the input effective theta
  matrix.  For these link functions, the \code{$sType}, \code{$sVec},
  and \code{$linkScale} are used to describe the extra parameter.
  
  Others do require an additional parameter.  The \code{GaussianLink}
  requires a residual standard deviation.  The \code{SlipLink}, 
  \code{GuessLink} and \code{GuessSlipLink} have a slipping, guessing
  and both parameters.  In this case, the \code{$sType} field is set to
  a \code{\link{PType}} object describing the expected domain and length
  of the scale parameter.

  As with the \code{\linkS4class{CombinationRule}} there are two
  versions the parameter, \code{$linkScale} which is the parameter on the
  natural scale, and \code{$sVec} which is transformed so that the
  domain is the real line.  For the \code{GaussianLink}, the \code{$linkScale} is
  the residual standard deviation on the natural scale (positive real
  numbers) and \code{$sVec} is the log of the standard deviation.
  Generally, users should set the natural version of the parameter.


}
\section{Effective Theta Dimensions}{

  In a conditional probability tensor, the last dimension must sum to
  1.0, that is it must be a unit simplex. Let the number of states in
  the last dimension of the output tensor be \eqn{K}.  There are
  currently three different strategies for ensuring the final CPT is a
  simplex, each of which have a different expected size for the input
  effective theta tensor.

  \emph{Normalization} (used by \code{PotentialLink} and
  \code{SoftmaxLink}).  Each cell is divided by the sum across the last
  dimension (see \code{\link{torch_simplexify}}).  For this strategy, the
  number of columns of the input effective theta tensor should be
  \eqn{K}.

  \emph{Drop one element}  (used by \code{StepProbsLink},
  \code{DifferenceLink}, \code{GradedResponseLink},
  \code{PartialCreditLink}, \code{SlipLink}, \code{GuessLink}, and
  \code{GuessSlipLink}).  In this strategy the input effective theta
  tensor has one fewer columns than the output tensor, so the dimension
  is \eqn{K-1}.

  \emph{Something Else} (used by \code{GaussianLink}).  The Gaussian
  model generates the complete probability simplex from a single mean
  (and the link scale parameter), so the dimension of the effective
  theta tensor is \eqn{1}.

  The method \code{\var{link}$etWidth{}} returns the required second
  dimension of the link function.

}
\section{Algorithms}{

  The basic alogirthm calls the \code{$link} function which should be a
  function of a tensor of size \eqn{S \times K'}, and returns a tensor
  of size \eqn{S \times K}, where \eqn{S} is the number of parent
  configurations, \eqn{K} is the number of states of the child variable,
  and \eqn{K'} is the required width of the effective theta
  (\code{\var{link}$etWidth()}).

  The \code{$forward(\var{input})} method calls this link function.

}
\section{Constructor}{

  The normal way to create a new link function object is to fetch the
  link function from the library with the function
  \code{\link{getLink}(\var{name})$new(\var{nstates})}, where
  \code{nstates} is the number of states in in the child variable.
  The function \code{\link{availableLinks}()} lists currently registered
  link functions.

  The currently available link functions are:

  \tabular{llrlll}{
    Name \tab Link function \tab D \tab Scale Parameter \tab Type \tab etWidth \cr
    Potential \tab simplexify \tab \tab -- \tab -- \tab K \cr
    StepProds \tab cumprod \tab \tab -- \tab -- \tab K-1 \cr
    Difference \tab diff \tab \tab -- \tab -- \tab K-1 \cr
    Softmax \tab softmax \tab 1.7 \tab -- \tab -- \tab K \cr
    PartialCredit \tab partialCredit \tab 1.7 \tab -- \tab -- \tab K-1 \cr
    GradedResponse \tab graded Response \tab 1.7 \tab -- \tab -- \tab K-1 \cr
    Gaussian \tab probit \tab \tab residual std \tab pos dim=c(1) \tab 1 \cr
    Slip \tab diff \%*\% slipmat \tab \tab slipping \tab unit dim=c(1) \tab K-1 \cr
    Guess \tab diff \%*\% guessmat \tab \tab guessing \tab unit dim=c(1) \tab K-1 \cr
    GuessSlip \tab diff \%*\% guessmat \tab \tab guessing \tab unit dim=c(2) \tab K-1 \cr
    \tab \%*\% slipmat \tab \tab slipping \tab \tab \cr
  }

  Note that the \code{SoftmaxLink}, \code{PartialCreditLink} and
  \code{GradedResponseLink} all use the \eqn{D=1.7} with the logistic
  (or exponential) function.

}
\section{Superclasses and Subclasses}{

  The \code{CombinationRule} is a subclass of
  \code{\link[torch]{nn_module}}.  The two most
  important inherited methods are \code{$forward()} which calculates the
  CPT from the effective theta tensor and \code{$parameters} which returns
  \code{$sVec} (or NULL).  The autograd function built into
  \pkg{torch} with calculate the gradient of the \code{$forward}
  function wrt the vector valued parameters.

}
\section{Creating Subclasses}{

  A subclass of \code{CombinationRule} can be created using the
  \code{\link[torch]{nn_module}} function, and having it inherit from
  the \code{CPT_Link} class.  The \code{$link} field should be set to a
  function which takes the effective theta tensor and returns the
  conditional probability tensor.  The \code{$etWidth} field should be
  set to the desired width of the input tensor.
  If a link scale parameter is needed, the private field \code{$stype}
  can be set to a \code{\linkS4class{PType}} class with the appropriate
  dimensions.  Note that the dimensions can be a function of \code{$K}
  the number of states in the output variable.  If needed the value of
  \code{$D} can be set to a scalar tensor.

  As an example here is the Softmax link.

  \preformatted{
    SoftmaxLink <- nn_module(
      classname="SoftmaxLink",
      inherit=CPT_Link,
      scale=NULL,
      D=torch_tensor(1.7),
      etWidth=function() {self$K},
      link=function(et) {
        nnf_softmax(et$mul_(self$D),2)
      },
      private=list(
        stype=NULL
      )
    )
}

}
\author{Russell Almond}
\seealso{

}
\keyword{classes}
