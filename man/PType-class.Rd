\name{PType}
\Rdversion{1.1}
\docType{class}
\alias{PType-class}
\alias{availablePTypes}
\alias{PType}
\alias{isPtype}
\title{Class \code{"PType"} -- a parameter description}
\description{

  This is an abstract desription of a parameter for a
  \code{\linkS4class{CombinationRule}} or
  \code{\linkS4class{LinkFunction}}.  Note that the dimensions are
  saved as an expression so that later the number of parent variables,
  \eqn{J}, the number of parent state combinations, \eqn{S}, and the
  number of states for the child variable, \eqn{K} can be subsituted in
  as needed.

}
\usage{
PTypefunction(pType,dim=c(K,J), zero=NULL, used=TRUE)
availablePTypes()
isPtype(obj)
}
\arguments{
  \item{pType}{A character value giving the subtype of the parameter
    type.  This should generally be one of the values returned by
    \code{availablePTypes()}.}
  \item{dim}{An expression giving the dimensions of the parameter.
    Note:  this is not evaluated on object creation, so can include
    references to \code{J} number of parents, \code{K} number of child
    states, or \code{S} number of possible combinations of the parents.
    See \code{\link{setpTypeDim}}.}
  \item{zero}{If supplied, a number to be used in creating default
    values.  If left null, the function \code{\link{getZero}} will be
    used instead.}
  \item{used}{This should be either \code{TRUE} or a logical matrix as
    the same dimension as the parameter.  Cells of the parameter matrix
    corresponding to \code{FALSE} are ignored.  See \code{\link{whichUsed}}.}
  \item{obj}{An object to be tested if it follows the \code{PType} protocol.}

}
\section{Extends and Extensions}{

  The PType class is implemented as an S3 class which class
  \code{c(pType, "PType")} were \code{pType} is one of the subclass types
  listed in the \squote{Available Subclasses} section listed below (or
  returned by \code{availablePTypes()}.  While the \dquote{PType} class
  takes care of the common methods (particularly dealing with
  dimensions, and used values) the subclasses take care of checking for
  used values, transformations between natural and real parameters and
  initial values.

}
\section{Fields}{
  \describe{
    \item{\code{dimexpr}:}{This is the dimension expression passed in
      when the PType object is first created.  It is an unevaluated
      expression and can use the constants \code{J}, \code{S} and \code{K}.}
    \item{\code{dim}:}{This is the integer vector returned by calling
      \code{\link{setpTypeDim}} on the PType object}
    \item{\code{zero}:}{If supplied, this is the value to the parameter
      with in the initial value.  If this is null, the the
      \code{\link{getZero}} function}
    \item{\code{used}:}{This is either \code{TRUE} (indicating that all
      values in the matrix are used parameters), or a logical matrix
      the same size of the parameter indicating which values of the
      matrix are useful parameters and which are ignored. See
      \code{\link{pMat2pVec}}.} 
  }

  Note that the class attribute is compound, with the first entry being
  the subtype.
}
\section{Generic Functions}{

  These are generic functions for which the methods are generally
  provided by the \dQuote{PType} class.

  \describe{
    \item{isPtype}{\code{signature(obj)}: Returns true if the object is
      a PType.  The default method checks that the subclass is in
      \code{availablePTypes()}, so must be overridden by new subclasses.}
    \item{\link{pTypeDim}}{\code{signature(pType)}: This returns the
      value of the \code{dim} field, after the dimensions have been
      set. }
    \item{\link{setpTypeDim}}{\code{signature(pType,J=1L,S=2L^J,K=1L)}:
      This function evaluates the \code{exprdim} expression,
      substituting in the supplied values for \code{J}, \code{S} and
      \code{K}. 
    }
  }

  This next collection of generic functions is generally implemented by
  the subclass, so will have slightly different behavior.
  
  \describe{
   \item{\link{checkParam}}{\code{signature(pType,par)}: Returns true if
     \code{par} is a legal value for the parameter.}
   \item{\link{natpar2Rvec}}{\code{signature(pType,natpar)}: Converts
     the parameter to a vector of real numbers.}
   \item{\link{Rvec2natpar}}{\code{signature(pType,Rvec)}: Inverse of
     the above transformation, takes a vector of real values and returns
     a legal parameter value of the correct shape.}
   \item{\link{getZero}}{\code{signature(pType)}: Returns a value to be
     used as the default value.}
   \item{\link{defaultParameter}}{\code{signature(pType)}: Fill the
     parameter with default values.}
 }
}
\details{

  The subclass determines what are valid values for the parameter.  For
  example, any number is valid for \dQuote{real}, only positive numbers
  for \dQuote{pos} and only numbers between 0 and 1 for \dQuote{unit}.
  Some of the other parameter types have more complex constraints, for
  example the values must be increasing for \dquote{incrK} parameters
  and must sum to 1 for \dquote{pVec} parameters.  The
  \code{\link{checkParam}} function checks if a value is valid.

  Let \eqn{J} be the number of parent variables, \eqn{S} the number of
  possible combinations of the parent states (the number of rows of the
  final CPT) and \eqn{K} equal the number of states of the child
  variable (the number of columns of the final matrix).  A compensatory 
  \code{\linkS4class{CombinationRule}} takes a \eqn{S \times J} matrix of
  effective thetas, \eqn{\Theta}, a \eqn{K'' \times J} matrix of slopes
  (discriminations), \eqn{A}, and a \eqn{K' \times 1} column vector of
  difficulties (negative intercepts).  The final effective theta output
  is a \eqn{S \times K'} matrix, computed by \eqn{\Theta A^{T} - B},
  where the dimensions are replicated as necessary.  The value of
  \eqn{K'} is determined by the \code{\linkS4class{LinkFunction}} and
  is generally \eqn{K-1} for the graded response and partial credit link
  functions, and 1 for the normal link function.  For the graded
  response link function, the \eqn{A} parameters are typically restricted
  to be the same for each output state, so the correct dimension is
  \eqn{K''=1}.

  When the parameter is first declared, the values of \eqn{J} and
  \eqn{K} (and \eqn{S}) may not been known, so the \code{dim} argument
  to \code{PType()} is not evaluated when the parameter type object is
  created.  The \code{exprdim} field is an unevaluated expression with
  the free variables \code{J} and \code{K} allowed.  In the example
  above, the dimensions of the \eqn{A} parameter would be declared as
  \code{c(K,J)}.  Then when the parameter is actually used, a call to
  \code{\link{setpTypeDim}} with te desired values of \code{J} and
  \code{K} fixed the final dimensions.

  When finding the optimal value of a parameter, e.g., with the
  \code{\link[stats]{optim}} function, transforming the parameter so
  that all real values are legal means that unconstrained optimization
  can be used.  Furthermore \code{optim} expects a parameter vector
  instead of matrix.  The generic functions \code{\link{natpar2Rvec}}
  and \code{\link{Rvec2natpar}} take care of this transformation.

  In some cases, not all of the cells in the parameter matrix are
  relevant to the correct computation.  In particular, with the graded
  response link function, the values along the \eqn{K} dimension
  represent transitions between states, and not all parent variables
  might be relevent for each transition.  If the value of
  \code{\link{whichUsed}} is a matrix of the same size of the parameter
  which is true if that cell is used in the computations and false if
  not.  The \code{\link{natpar2Rvec}} code removes unused cells from the
  parameter vector and \code{\link{Rvec2natpar}} restores the matrix to
  the right shape (with arbitrary values in the unused cells).

  Finally, the function \code{\link{defaultValues}}} creates a parameter
  object of the necessary shape (assuming te dimensions have been
  finalized) which has a valid value.  The function
  \code{\link{getZero}} returns a default value, and the default method
  for \code{defaultValues} fills the matrix with that value.

}
\section{Available Subclasses}{

  The following subclasses are avaiable.

  \describe{
    \item{\dquote{real}}{:  Values are any real number.  Uses the
      identity transformation in calculating the real vector.}
    \item{\dquote{pos}}{:  Values are any positive number.  Uses the log
      transformation in calculating the real vector.}
    \item{\dquote{unit}}{:  Values are between 0 and 1.  Uses the logit
      transformation in calculating the real vector.}
    \item{\dquote{pVec}}{:  This is a vector representing a probability
      distribution, with all values non-negative and the sum equals 1.  It
      uses the log transformation in going to real values and te
    \code{\link{softmax}} in going back to the natural parameter space.}
    \item{\dquote{cpMat}}{: A conditional probability matrix in which
      each row must sum to one.  It applies the softmax to the rows on
      the inverse transformation.}
    \item{\dquote{incrK}}{:  A matrix whose values are increasing in the
      columns.  This is particularly useful for situation in which
      difficulty parameters are expected to be increasing.  It uses the
      \code{\link{ldiff}} transformation to get to a real vector.}
    \item{\dquote{const}}{:  A special case in which the parameter is to
      be considered fixed wrt optimization.  It allows any value and
      and both \code{natpar2Rvec.const} and \code{Rvec2natpar.const}
      return a zero length vector.}
  }

}
\section{New Subclasses}{

  In order to add a new subclass, \dQuote{newclass} the following S3
  methods must be implemented.

  The method \code{isPType.newclass} must return true (as the new class
  will not be in the list of built-in classes.

  A method for \code{checkParam.newclass} is needed.  It should
  generally return \code{FALSE} if unsuccessful and call
  \code{NextMethod()} if successful as the \code{checkParam.PType}
  method checks the dimensions of the parameter.

  Methods for \code{natpar2Rvec.newclass} and
  \code{Rvec2natpar.newclass} which make the usual transformations.
  These often call \code{\link{pMat2pVec}} and \code{pVec2pMat}
  respectively to take care of unused parameters.

  A method for \code{getZero.newclass} and possibly
  \code{defaultValue.newclass}.   The default method fills the parameter
  matrix with the zero value, which may suffice.

}
\author{Russell Almond}
\seealso{

  \code{\linkS4class{CombinationRule}},
  \code{\linkS4class{LinkFunction}},

  \code{\link{setpTypeDim}}, \code{\link{whichUsed}},
  \code{\link{pMat2pVec}}, \code{\link{checkParam}},
  \code{\link{natpar2Rvec}}, \code{\link{defaultParameter}}

}
\examples{
pt <- PType("pos")
pt <- setpTypeDim(pt,K=3,J=2)
af <- matrix(1:6+.1,3,2)
checkParam(pt,af)
af[2,2] <- -3
checkParam(pt,af)

pv <- (-2:3)+.1
pm <- matrix(exp(pv),3,2)
natpar2Rvec(pt,pm)
Rvec2natpar(pt,pv)

qq <- matrix(c(T,F,T,F,T,T),3,2)
whichUsed(pt) <- qq
natpar2Rvec(pt,pm)


}
\keyword{classes}
