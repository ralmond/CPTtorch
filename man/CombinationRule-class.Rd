\name{CombinationRule}
\Rdversion{1.1}
\docType{class}
\alias{CombinationRule-class}
\alias{RuleASB}
\alias{RuleBSA}
\alias{RuleBAS}
\alias{RuleConstA}
\alias{RuleConstB}
\title{Class \code{"CombinationRule"} -- Calculates Effective Thetas}
\description{

  The first step in the DiBello model for conditional probability tables
  is to take an \emph{parent theta matrix} (a table of combinations
  of the parent variablee states mapped to real values) and condense
  them down to a set of \emph{effective thetas} for each parent
  combination.  In general the number of columns in the effective theta
  matrix is related to the number of states in the child variable with
  the exact definition given by the \link{CPT_Link} function used.

}
\value{

  The \code{$forward()} method (which takes no arguments) calculates an
  effective theta tensor.  This is a \eqn{S \times K} tensor where
  \eqn{S} is the number of possible configurations of the parent
  variables, and \eqn{K} is the number passed via the \code{nstates}
  argument to the constructor.  (The value of \eqn{K} is based on the
  number of states of the output variable, but the value is generally
  found by calling \code{\var{link}$etWidth()} on the link function to
  be used with the combination rule.

}
\section{Public Fields}{

  \describe{
    \item{aVec,bVec}{These are the A (discrimination, weight) and B
      (difficulty, bias) parameters transformed onto the real line,
      \eqn{\mathbb{R}}.  Generally the value should be a subclass of
      \code{\link[torch]{nn_parameter}}.  Typically, these are not set
      directly but set through the \code{$aMat} and \code{$bMat} active
      values.}
    \item{aop,bop}{These are the combination functions used with the A
      and B paramters (see Algorithm).  They should have
      \code{signature(e1:torch_tensor, e2:torch_tensor)}.}
    \item{summary}{This is the summary function used in the algorithm
      (see Algorithm).  It should have
      \code{signature(self:torch_tensor, dim:integer)}.}
    \item{pTheta}{This is the calculated Parent Theta Matrix.  The
      recommended way to set this field is through the
      \code{$setParents()} method.}
  }

}
\section{Active Fields}{

  \describe{
    \item{aType}{The type of the A parameter.  This can be set to a
      \code{\linkS4class{PType} object.}
    \item{bType}{Type type of the B parameter.  This can be set to a
      \code{\linkS4class{PType} object. }
    \item{aMat}{The A parameter as a tensor (either \eqn{K \times J} or
      \eqn{K \times 1} depending on the rule type.  Note that this is in
      the natural scale.}
    \item{bMat}{The B parameter as a tensor.}
    \item{QQ}{This should either be a logical tensor the same size as
      the A/B tensor, or the scalar value \code{TRUE} (which is
      equivalent to a tensor with all values true, but faster.) }
    \item{et}{This returns the value of the effective theta tensor,
      returning a cached value if available.}
    \item{et_p}{If called without arguments it returns a logical value
      indiciating whether or not a cached effective theta tensor is
      available.  If set to FALSE, it clears the cache.}
  }
}
\section{Public Methods}{
  \describe{
    \item{setParents}{\code{signature(parents)}. Parents should be a
      list of numeric vectors giving the mapped value for each parent
      (see Parent Theta Matrix).  Note that setting this value can reset
    the parameter values.}
    \item{setDim}{\code{signature(S=1L,J=1L,K=1L)}. This function
      adjusts the dimensions of the parameters.  (See Parameter
      Dimensions).  Note that setting this value can reset the parameter
      values.}
    \item{initialize}{\code{signature(parents,nstates,QQ=true,...)}. This
      initializes a new Rule object.  See Initialization.}
    \item{forward}{\code{signature()}. This function calculates the
      effective theta table from the parent theta matrix.}
  }
}
\section{Private Fields}{

  These should only be manipulated when building subclasses.

  \describe{
    \item{atype}{The initial type of the A parameter.}
    \item{btype}{The initial type of the B parameter.}
    \item{SJK}{A list with three dimensions, \code{S} -- parent
      configuations, \code{J} -- number of parents, and \code{K} --
      number of columns needed in the effective theta table. }
    \item{cache}{Either a cached effective theta table or \code{NULL}.}
  }
}
\section{Parent Thetas Tensor}{

  Assume that there a \eqn{J} parent varaibles and each parent variable
  has \eqn{M_j} states.  Each state of each parent variable is mapped to
  a value on the real line.  As the scale of this mapping is
  undetermined, the convention is to map the points onto quantiles of a
  standard normal distribution, so that the median is at 0 and 95
  percent of the values are between -2 and 2.  The function
  \code{effectiveTheta} produces equally spaced quantiles of the normal
  distribution which can be used for this mapping.

  The \code{parents} argument to the \code{$setParents} method should be
  a list of length $J$ with each entry corresponding to a parent
  variable.  Each value should be a numeric vector of length \eqn{M_j}
  giving the assigned real values of the parent states.

  This is passed to \code{\link[torch]{torch_cartisian_product}} (see
  also \code{\link[base]{expand.grid}}) to produce an \eqn{S \times J}
  matrix of all possible combinations of the parent states.  Here
  \eqn{S = \prod_j M_j}; that is each row corresponds to a possible combination
  of the number of parent states.

  Note that setting the parents will potentially change the required
  size of the parameters, and hence cause them to be replaced by the
  default value.

}
\section{Algorithms}{

 Let \eqn{\Theta} be the \eqn{S \times J} tensor of parent
 thetas. Let \bold{A} be the \eqn{K \times J} discrimination or
 weight tensor and let \bold{B} be the \eqn{K \times 1} difficulty or
 bias tensor.  Then the compensatory effective theta is given by

 \deqn{\tilde{\theta} = \Theta \bold{A}^T - \bold{B}^T}, where the
 \code{B} tensor is replicated along the first dimension (broadcast) as
 part of the addition.

 The matrix multiplication can be replaced by a generalized matrix
 multiplication (\code{\link{genMMt}}).  Let ⓐ be the A-operator (the
 \code{$aop} field) , ⓑ be the B-operator (the \code{$bop} field), and
 \eqn{\bigoplus} be the summary operator (the \code{$summary} field).  In the
 compensatory model, ⓐ is multiplication, \eqn{\bigoplus} is summation (over the
 second dimension) and ⓑ is subtraction.  Project the \eqn{Theta} tensor
 to have dimension \eqn{S \times J \times 1} and Project the \bold{A}
 tensor to have dimension \eqn{1 \times J \times K}, and the generalized
 matrix multiplication can be rewritten as:

\deqn{\tilde{\theta} = \bigoplus_{J} (\Theta \mathbin{ⓐ} \bold{A})  \mathbin{ⓑ} \bold{B}.}

In some cases, it is advantageous to perform these operations in a
different order.  Currently, classes for three supported orders are
available.

\code{ASBRule}.

\deqn{\tilde{\theta} = \bigoplus_{J} (\Theta \mathbin{ⓐ} \bold{A})  \mathbin{ⓑ} \bold{B}.}
 
This is the standard order.  \bold{A} should have dimension \eqn{J
  \times K} and \bold{B} should have dimension \eqn{K \times 1}.  The
  \code{CompensatoryRule} is an example of this type of rule.

\code{BSARule}.

\deqn{\tilde{\theta} = \bigoplus_{J} (\Theta \mathbin{ⓑ} \bold{B}) \mathbin{ⓐ} \bold{A} .}
 
This is a reversed (bias first) order.  \bold{A} should have dimension \eqn{K
  \times 1} and \bold{B} should have dimension \eqn{J \times K}. The
  \code{ConjunctiveRule} and \code{DisjunctiveRule} are examples, with ⓑ
  as subtraction, ⓐ multiplication and \eqn{\bigoplus} minimum and maximum
  respectively.


\code{BASRule}.

\deqn{\tilde{\theta} = \bigoplus_{J} ((\Theta \mathbin{ⓑ} \bold{B}) \mathbin{ⓐ} \bold{A}) .}
 
  In this rule, the summation step is doen last.  Both \bold{A} and
  \bold{B} should have dimensions \eqn{K \times J}.  The
  \code{NoisyAndRule} and \code{NoisyOrRule} rules are examples with with ⓑ
  as greater-than, ⓐ multiplication and \eqn{\bigoplus} as the product (or
  complementary product).

The default \code{$forward()} methods for the \code{ASBRule} (which is
the same as the \code{CombinationRule}), \code{BSARule} and
\code{BASRule} implement these three operations using the operations in the
\code{$aop}, \code{$bop} and \code{$summary} fields.  The forward method
can be overridden either to create another kind of rule (e.g., the
constant rules) or to produce a more efficient implementation (see the
\code{CompensatoryRule}). 

}
\section{Parameters}{

  There are two parameters A and B, both of which are tensor-valued.

  \describe{
    \item{A}{The A paramter is used as a discrimination, weight or
      slope.}
    \item{B}{The B parameter is used as a difficulty, bias or
      intercept.}
  }

  Each parameter has a corresponding type: \code{$aType} and
  \code{$bType} respectively.  These govern what are legal values.
  Often A-parameters are constrained to be positive and B-parameters are
  not.

  There are two views of each parameter: the matrix view (\code{$aMat}
  and \code{$bMat}, which are two-dimensional tensors) and the vector
  view (\code{$aVec} and \code{$bVec} which are one dimensional
  tensors).  In addition to reshaping (note that \code{torch_tensors}
  are stored in row-major order), if the parameter space is constrained,
  then the vector version is transformed to an unconstrained space
  (e.g., if the A parameter is positive, then the \code{$aVec} is
  \eqn{\log(A)}).  The conversion between the natural (constrained)
  parameter space of the matrix view and the unconstrained (\eqn{R})
  space is goverened by the \code{$natpar2Rvec()} and
  \code{$Rvec2natpar()} methods of the corresponding
  \code{\linkS4class{PType}}. In addition, if there is an inner Q matrix
  (that is the \code{$QQ} value has false values) then the unused
  elements are removed from the vector view.

  In general, users should set the parameters using the matrix views
  (\code{$aMat} and \code{$bMat}).  On the other hand, the formal
  \code{\link[torch]{nn_parameter}} objects are the vector views
  (\code{$aVec} and \code{$bVec}).  These are the versions which are
  optimized or should be randomly sampled in a Monte Carlo algorithm.

}
\section{Parameter Dimensions}{

  The dimnesions of both the A and B parameter can be expressed in terms
  of three constants:
  \describe{
    \item{S}{The number of configurations of the parent variable.  This
      is \code{nrow($pTheta)}.  If there are no parents, this should be
      \code{1L}.}
    \item{J}{The number of parents.  This is \code{ncol($pTheta)}.  Note
      that if there are no parents, this should be \code{1L}.}
    \item{K}{This is the number of columns in the output effective theta
      matrix.  This depends on the link function
      (\code{\linkS4class{CPT_Link}$etWidth()}).  In many cases this is
      the number of states in the child variable minus 1 (for
      normalization), although the number of states, and just one are
      also possible.}
  }

  The dimensions of the \code{$aType} and \code{$bType} are expressions,
  which can contain the constants \code{S}, \code{J} and \code{K}.

  The \code{$setDim()} function updates the values of the three
  constants.  If the size of the parameter changes, it is reset to its
  default value.  Note that \code{$setParents()} will also call
  \code{$setDim()}. 

  Note that dimensions \code{J} and \code{K} can be replaced with
  \code{1}.  The tensor will the be broadcast (or replicated over the
  appropriate dimension).  For example suppose that the A parameter has
  dimension \code{(c(K,J)}, meaning a different set of discriminations
  for each column of the output.  If instead a parameter with dimension
  \code{c(1,J)} is used, then the same discrimination parameters will be
  used for each row of the output.
  
}
\section{Inner Q-matrix}{

  Each column of the output matrix is associated with a different state
  of the child variable, a boundary between two states, or a transition
  between two states.  A variable is a parent if it affects any of the
  columns, but it may be that certain parent variables are not relevant
  for certain states or transitions between states.

  The \code{QQ} matrix \eqn{K \times J} logical matrix where
  \code{QQ[k,j]} is false if Parent \eqn{j} is not relevant to output
  column \eqn{k}.

  For a ASB type rule, the \code{$QQ} field is stored with the
  \code{whichUsed} property of the \code{aType}.  And the effective
  forward equation becomes:

  \deqn{\tilde{\theta}_{s,k} = \bigoplus_{j: qq_{k,j}} (\Theta_{s,j} \mathbin{ⓐ} a_{k,j})  \mathbin{ⓑ} b_{k,1}.}

  For a BSA type rule, the \code{$QQ} field is stored with the
  \code{whichUsed} property of the \code{bType}.  And the effective
  forward equation becomes:

  \deqn{\tilde{\theta}_{s,k} = \bigoplus_{j: qq_{k,j}} (\Theta_{s,j} \mathbin{ⓑ}
  b_{k,j}) \mathbin{ⓐ} a_{k,1}.}

  For BAS type rules, the \code{$QQ} field is stored with the
  both types and the forward equation is:

  \deqn{\tilde{\theta}_{s,k} = \bigoplus_{j:qq_{k,j}} ((\Theta_{s,j} \mathbin{ⓑ} b_{k,j}) \mathbin{ⓐ} a_{k,j}) .}

  Note that values of the A (for ASB and BAS rules) and B (for BSA and
  BAS rules) tensors corresponding to false values of \code{QQ} do not
  influence the output, therefore they are removed from the vector
  versions of the parameters.

  In many cases, the inner Q-matrix is not needed, or should be a tensor
  filled with \code{TRUE}.  In this case, the the value \code{TRUE} can
  be used to indicate that all elements of the parameters are used.
  This results in a slightly more efficient version of the algorithm
  being used.  (Note that if the \pkg{torch} tracing facility has been run, it
  needs to be rerun after changing the value of \code{$QQ} as the
  branching in the R code will not be traced).

}
\section{Constructor}{

  The normal construction proceedure is to first look up a subclass of
  \code{CombinationRule} using the \command{\link{getRule}(\var{name})}
  function.  (A list of currently available rules can be found with
  \command{avilableRules()}.)  Then the \command{$new(\var{parents},
    \var{nstates}, \var{QQ}, ...)} should be invoked on the result.
  The arguments are as follows:

  \describe{
    \item{\code{parents}}{This should be a list of length \eqn{J} where
      each element is a vector of real values associated with the
      parents.  The function \command{\link{as_Tvallist}()} will
      create appropriate parent value lists from abbreviated formats.}
    \item{\code{nstates}}{This is the number of columns in the output
      tensor.  Generally, it determined by the value of
      \command{\linkS4class{CPT_Link}$etWidth()}.}
    \item{\code{QQ}}{If supplied this should be a logical tensor with
      dimensions \eqn{K \times J} (see Inner Q-matrix above).  If
      ommitted, it defaults to \code{TRUE}.}
    \item{\dots}{This is supplied so that subclasses can add additional
      arguments.}
  }

  The \code{$aMat} and \code{$bMat} are initialized to default values,
  but can be set to new values as desired.

  Here is an example for creating a compensatory distribution.
  \preformatted{
    rule1 <- getRule("Compensatory")$new(
      as_Tvallist(list(S1=c("High","Med","Low"),
                       S2=("Master","Non-Master"))),
      2)
    rule1$aMat <- torch_tensor(matrix(c(.8,1.1),1,2))
    rule1$bMat <- torch_tensor(matrix(c(1,-1),1,2))
  }
  
}
\section{Superclasses and Subclasses}{

  The \code{CombinationRule} is a subclass of
  \code{\link[torch]{nn_module}} and has direct subclasses
  \code{RuleASB}, \code{RuleBSA} and \code{RuleBAS}, \code{RuleConstA}
  and \code{RuleConstB}.  The two most
  important inherited methods are \code{$forward()} which calculates the
  effective theta tensor and \code{$parameters} which returns a list of
  the \code{$aVec} and \code{$bVec}.  The autograd function built into
  \pkg{torch} with calculate the gradient of the \code{$forward}
  function wrt the vector valued parameters.

  Three direct subclasses, \code{ASBRule} (which is currently an
  alias for \code{CompensatoryRule}), \code{BSARule} and \code{BASRule},
  are abstract classes which differ by the order of operations in
  calculating the final rule.  The \code{RuleConstA} and
  \code{RuleConstB} are special rules which ignore the parents and
  simply return the parameter in question.

  \tabular{lllclll}{
    Rule \tab Algorithm \tab Summary \tab AB \tab Op \tab pType \tab dim \cr
    Compensatory \tab ASB \tab sumrootj \tab A \tab mul \tab pos \tab c(K,J) \cr
                 \tab     \tab          \tab B \tab sub \tab real \tab c(K,1) \cr
    Compensatory1 \tab ASB \tab sumrootj \tab A \tab mul \tab pos \tab c(1,J) \cr
                  \tab     \tab          \tab B \tab sub \tab real \tab c(K,1) \cr
    Conjunctive \tab BSA \tab min \tab B \tab sub \tab real \tab c(K,J) \cr
                \tab     \tab    \tab A \tab mul \tab pos \tab c(K,1) \cr
    Disjunctive \tab BSA \tab max \tab B \tab sub \tab real \tab c(K,J) \cr
                \tab     \tab    \tab A \tab mul \tab pos \tab c(K,1) \cr
    NoisyAnd \tab BAS \tab prod \tab B \tab gt \tab real \tab c(K,J) \cr
             \tab     \tab      \tab A \tab mul \tab pos \tab c(K,J) \cr
    NoisyOr \tab BAS \tab 1-prod \tab B \tab gt \tab real \tab c(K,J) \cr
             \tab     \tab      \tab A \tab mul \tab pos \tab c(K,J) \cr
    Center \tab ConstB \tab --  \tab B \tab --  \tab real \tab c(S,K) \cr
    Dirichlet \tab ConstB \tab --  \tab B \tab --  \tab pos \tab c(S,K) \cr
  }

The \code{CenterRule} and \code{DirichletRule} both have a single
parameter, but are meant to use with different links.  The
\code{CenterRule} is meant to use with the Gaussian link, where
generally \eqn{K=1} and the parameter is the mean.  The
\code{DirichletRule} is meant to use with the potential link,
where \eqn{K} is the number of states and the link function simply
normalizes each row.  So this is used to implement the non-parametric
hyper-Dirichlet model.


}
\section{Creating Subclasses}{

  A subclass of \code{CombinationRule} can be created using the
  \code{\link[torch]{nn_module}} function.  Generally this will inherit
  from one of the subclasses:  \code{RuleASB}, \code{RuleBSA},
  \code{RuleBAS}, or \code{RuleConstB}, giving the basic framework of
  the rule.  Next, the \code{$aop}, \code{$summary} and \code{$bop}
  functions need to be set to the appropriate \pkg{torch} functions.
  Finally, the private fields \code{$atype} and \code{$btype} should be
  set to the parameter type matching the operations.

  Additionally, the \code{$forward()} method can be overridden to either
  provide a for a new algorithm or a more efficient implmentation.

  Finally, the new class can be registered using the
  \code{\link{setRule}()} function.

  As an example, here is the implementation for the
  \code{ConjunctiveRule}. 

  \preformatted{
    ConjunctiveRule <- nn_module(
    classname="ConjunctiveRule",
    inherit = RuleBSA,
    bop = torch_sub,
    summary = torch_min,
    aop = torch_mul,
    private=list(
        atype=PType("pos",c(K,1)),
        btype=PType("real",c(K,J))
     )
  }
  

}
\author{Russell Almond}
\seealso{

}
\references{

}
\keyword{classes}
