\name{as_Tvallist}
\alias{as_Tvallist}
\alias{effectiveTheta}
\alias{buildeTheat10}
\alias{cartesian_prod}
\title{Build a Mapping from Variable States to Theta Values.
}
\description{

  The first step in building a CPT is to assign real values to the
  parent states.  This expands various abbreviated specifications to
  provide a list of state value maps.

}
\usage{
as_Tvallist(parents, parentprefix="P", stateprefix="S", high2low=TRUE)
effectiveTheta(nlevels,high2low=TRUE)
buildeTheta10(Tvallist)
cartesian_prod{list_o_vecs}
}
\arguments{
  \item{parents}{A possibly abbreviated form of the parent
    specification.  This could be a list of numeric vectors giving the
    parent state values, a list of character vectors giving the parent
    state names, or a vector giving the number of states in each parent
    variable. }
  \item{parentprefix,stateprefix}{A string to prepend to parent (state)
    names when they are not supplied.}
  \item{high2low}{A logical value or vector.  Are states ordered high to
    low (default) or low to high (FALSE).  For \code{as_Tvallist} can
    have unique value for each parent.}
  \item{nlevels}{The number of levels of the variable.}
  \item{Tvallist}{A named list of named numeric vectors, i.e., the
    output of \code{as_Tvallist()}.}
  \item{list_o_vecs}{A list of vectors.}
}
\value{

  The value of \code{as_Tvallist} is a list of the form:
  \code{(list
    \var{parent1}=c(\var{state1_1}=\code{val1_1}, \var{state1_2}=\code{val1_2}, ...),
    \var{parent2}=c(\var{state2_1}=\code{val2_1}, ...),
    ...)}

  Here \var{parent1}, 2, ... are the names of the parent variables,
  The values \var{state1_1} is the name of the first state of the first
  parent, and \var{val1_1} is the value of the first state of the first
  parent.

  If there are no parents, this is an empty list.

  The output of \code{effectiveTheta} is a numeric vector of length
  \code{nlevels}.

  The output of \code{buildeTheta()} is a tensor with columns
  corresponding to the parent values and rows corresponding to
  configurations. The output of \code{cartesian_prod} is a data frame 
  with similar rows and columns.
  
}
\details{

  The function \code{effectiveTheta(M)} produces \eqn{M} equally spaced
  normal quantiles, that is \code{qnorm(((1:\var{M})-.5)/\var{M})}.
  These are a ordered from highest to lowest or lowest to highest
  depending on the value of \code{high2low}.


  The final product of \code{as_Tvallist()} is a named list of named
  numeric vectors.  The values are used to calculate the parent theta
  matrix, and the names are used to label the resulting tensor.  This
  function will impute values based on the number of states.

  In particular, the most common values to set for the parent states are
  the values returned by
  \code{effectiveTheta(length(parents[i])}.  Thus, the actual
  values aren't necessary, just the names of the states.  Thus, there
  are three possible ways that the \code{parents} variable can be
  abbreviated:

  \describe{
    \item{List of character vectors.}{These are the names of each
      parent states.  These are converted to numeric vectors (with the
      supplied character vectors as names) using the
      \code{effectiveThetas()} function to supply the values.}
    \item{Vector of integers.}{This is converted into a vector of
      names (using the \code{stateprefix} and number) and then the
      previous rule is applied.}
    \item{List of numeric vectors.}{No conversion is necessary, but if
      no names are supplied, names are created using \code{stateprefix}.}
  }

  If parent names are not supplied, then they are created using
  \code{parentprefix}. The \code{high2low} argument is passed to the
  calls to \code{effectiveTheta}, it is replicated to the length of the
  parents.

  The function \code{buildeTheta10()} takes the output of
  \code{as_Tvallist} and turns it into a tensor with columns
  corresponding to the parent values and rows to configurations of the
  parents.  It is basically a wrapper for
  \code{\link[torch]{torch_cartesian_prod}}.  This function is similar,
  but not quite the same as \code{\link[base]{expand.grid}} (the first
  variable varies fastest with \code{expand.grid} and slowest with
  \code{cartesian.prod}.)  The function \code{cartesian_prod()} is the
  equivalent of \code{torch_cartesian_prod} only working with a list of
  R objects.
  
}
\author{Russell Almond}
\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{

pvals <- list(skill1=c(1,0,-1),skill2=c(1,-1))
as_Tvallist(pvals)
buildeTheta10(as_Tvallist(pvals))

effectiveTheta(5)
effectiveTheta(3,FALSE)


pnames <- list(var1=c("High","Med","Low"),var2=c("Yes","No"))
as_Tvallist(pnames)
cartestian_prod(pnames)
buildeTheta10(as_Tvallist(pnames))

as_Tvallist(c(3,2))


as_Tvallist(list())
buildeTheta10(as_Tvallist(list()))

}
% \keyword{ ~kwd1 }
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }

