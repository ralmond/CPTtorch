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
    \item{slipP,guessP}{This is either a \code{nn_parameter()}
      containing the logit of the slipping/guessing parameter, or
      \code{NULL} inidicating that it not use.}
    \item{high2low}{A logical value.  If true, then the columns are
      reversed in the final graph.}
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
    \item{slip,guess}{These set the slipping or guessing parameter.  It
      can be set to either a numeric value to set the slipping/guessing
      parameter or to \code{NA} or \code{NULL} to disable this feature.}
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
    \item{initialize}{\code{signature(\var{nstates}:int,\var{guess}=NA,
      \var{slip}=NA, \var{high2low}=FALSE)} initializes
      the link. }
    \item{leakmat}{\code{signature()}{Either returns a leaking mattrix
	if the slip or guess parameter is set, or returns \code{NULL} if
	neither were set.}
    \item{forward}{\code{signature(\var{input}:torch_tensor)} this does
      the calculation.  The default method calls the \code{$link}
      function, multiplies it by the \code{$leakmat} matrix and then
      reverses the columns if \code{$reversed} is true. }
  }

}
\section{Scale Parameter}{

  Although most link functions do not require additional parameters, the
  \code{GaussianLink} requires a residual standard deviation.
  \code{$sType} field is set to a \code{\link{PType}} object describing 
  the expected domain and length of the scale parameter.

  As with the \code{\linkS4class{CombinationRule}} parameters there are two
  versions the parameter, \code{$linkScale} which is the parameter on the
  natural scale, and \code{$sVec} which is transformed so that the
  domain is the real line.  For the \code{GaussianLink}, the
  \code{$linkScale} is the residual standard deviation on the natural
  scale (positive real numbers) and \code{$sVec} is the log of the
  standard deviation. Generally, users should set the natural version of
  the parameter. 

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
  \code{PartialCreditLink} tensor has one fewer columns than the output
  tensor, so the dimension is \eqn{K-1}.

  \emph{Something Else} (used by \code{GaussianLink}).  The Gaussian
  model generates the complete probability simplex from a single mean
  (and the link scale parameter), so the dimension of the effective
  theta tensor is \eqn{1}.

  The method \code{\var{link}$etWidth{}} returns the required second
  dimension of the link function.

}
\section{Slipping and Guessing}{

  With logical models, sometimes a leak or noise propability is added to
  flip the output.  In the classic DINA (deterministic input noisy-and)
  model, a _slipping_ parameter, $e$, is added which is the probability
  that the child variable takes on the zero state even if all of the
  inputs take on the one state.  In the DINO (deterministic input
  noisy-or) model, the noise parameter is a _guessing_ parameter, $g$,
  the probability of getting a 1 output even with a zero input.

  To extend this to non-binary child variables define the guessing
  parameter $g$, as the probability that the output will be one level
  higher than specified by the model.  The probability it will be two
  levels higher is $g^2$, and three levels higher $g^3$.  This produces
  a guessing matrix:

  \deqn{\mathbf{G} =\matrix{cccc}{
      * \tab g \tab g^2 \tab g^3 \cr
      0 \tab * \tab g   \tab g^2 \cr
      0 \tab 0 \tab *   \tab g   \cr
      0 \tab 0 \tab 0   \tab 1   \cr
    }\ ,}
  where * is the value need to make the row sum to one.  Note, that to
  ensure that the sum is not greater than one, $g$ is restricted to be
  less than 1/2.

  Similarly, the slipping matrix is:
  \deqn{\mathbf{E} =\matrix{cccc}{
      1   \tab 0   \tab 0 \tab 0 \cr
      e   \tab *   \tab 0 \tab 0 \cr
      e^2 \tab e   \tab * \tab 0 \cr
      e^3 \tab e^2 \tab e \tab * \cr
    }\ .}

  If both guessing and slipping parameters are applied, the resulting
  matrix is \eqn{\mathbf{G}\mathbf{E}}, that is first the guessing logic
  is applied and then the slipping.
  
  This covers the most common cases while using only one parameter for
  guessing and one for slipping.  These are the \code{link$guess} and
  \code{link$slip} parameters, which should have the value \code{NA} if
  guessing or slipping is not used.

  The functions \code{\link{guessmat}} and \code{\link{slipmat}} build
  these matrixes.  The method \code{$leakmat()} returns \code{NULL},
  \eqn{\mathbf{G}}, \eqn{\mathbf{E}} or \eqn{\mathbf{G}\mathbf{E}}
  depending on which of the guessing and/or slipping parameters are set.
  [Note, mathematically, \code{$leakmat()} could return the identity
  matrix if no parameters are set, but skipping the multiplication
  should be faster.]
  
  As with the scale parameter, the \code{$slip} and \code{$guess} are
  active fields with the true \code{\link[torch]{nn_parameter}} objects
  in the \code{$slipP} and \code{$guessP} fields.  As the parameters are
  constrained to be in the interval $[0,.5]$, the actual values stored are
  $\mathord{logit](2g)$ and $\mathord{logit](2e)$.

}
\section{Algorithms}{

  The basic alogirthm calls the \code{$link} function which should be a
  function of a tensor of size \eqn{S \times K'}, and returns a tensor
  of size \eqn{S \times K}, where \eqn{S} is the number of parent
  configurations, \eqn{K} is the number of states of the child variable,
  and \eqn{K'} is the required width of the effective theta
  (\code{\var{link}$etWidth()}).

  The \code{$forward(\var{input})} method calls this link function. If
  the \code{$guess} or \code{$slip} have been specified, then the result
  of the link function are multiplied by the \code{$leakmat()}.

  All link functions are written as if the the states are ordered from
  lowest to highest in the output matrix.  If the \code{$high2low} field
  is set, then the columns are reversed to change the ordering.

}
\section{Constructor}{

  The normal way to create a new link function object is to fetch the
  link function from the library with the function
  \code{\link{getLink}(\var{name})$new(\var{nstates},\var{guess},
  \var{slip},\var{high2low})}, where
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
  }

  Note that the \code{SoftmaxLink}, \code{PartialCreditLink} and
  \code{GradedResponseLink} all use the \eqn{D=1.7} with the logistic
  (or exponential) function.

  The \code{guess} and \code{slip} arguments can be used to adjust the
  corresponding fields.  If left at the default value of \code{NA}, then
  the slipping and guessing adjustments will not be made.

  If the \code{high2low} flag is set, then the columns of the final CPT
  will be reversed to make the states go from highest to lowest.

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
