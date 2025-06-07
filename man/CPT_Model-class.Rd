\name{CPT_Model}
\Rdversion{1.1}
\docType{class}
\alias{CPT-class}
\alias{CPT_Model-class}
\title{Class \code{"CPT_Model"} -- Model for a conditional probability tensor.}
\description{

  This class binds together a \code{\linkS4class{CombinationRule}} and a
  \code{\linkS4class{CPT_Link}} to produce a model for a conditional
  probability tensor.  It can generate the CPT from the current
  parameter values or perform a gradient descent search to find
  parameters which most closely match a given data set.
}
\section{Public Fields}{

  \describe{
    \item{rule}{The \code{\linkS4class{CombinationRule}} in the model.}
    \item{link}{The \code{\linkS4class{CPT_Link}} in the model.}
  }


}
\section{Active Fields}{

  \describe{
    \item{aMat}{The A parameter of the combination rule.  This is a
      synonym for \code{self$rule$aMat}.}
    \item{bMat}{The B parameter of the combination rule.  This is a
      synonym for \code{self$rule$bMat}.}
    \item{linkScale}{The scale parameter of the link funciton.  This is
      a synonym for \code{self$link$linkScale}.}
    \item{parentVals}{This is a list of numeric vectors giving the real
      value associated with each parent state.}
    \item{parentNames}{This is a list of character vectors giving the
      names of the parent states.}
    \item{stateNames}{This is the list of names of the output variables.}
    \item{QQ}{A logical tensor (inner-Q matrix) or \code{TRUE}.  This is
      a synonym for \code{self$rule$QQ}.}
  }

}
\section{Private Fields}{

  \describe{
    \item{parents}{The state values of the parents.  This should be a
      named list of named numeric vectors.}
    \item{states}{A character vector containing the names of the output
      variable.}
    \item{shape}{The shape of the complete tensor.}
    \item{cpt}{If the CPT has not changed since the last time the
      parameters were updated, this is the cached value.}
  }


}
\section{Methods}{

  \describe{
    \item{initialzie}{\code{signature(\var{ruletype}:char,
	\var{linktype}:char,
	\var{parents}=list(),\var{states}=character(),
        \var{QQ}=TRUE)}.  Creates a new CPT model.  See Construtor
      section below.
    }
    \item{forward}{\code{signature()}. Calculates the CPT from the
    current parameters.  Calling this function forces a recalculation.}
    \item{lscore}{\code{signature(\code{datatab}:tensor)}. Calculates
      the log probability of the data given in \code{datatab}. See
      calculating the score below. }
    \item{getCPT}{\code{signature()}. Gets the CPT, using a cached value
      if possible.  This is a \eqn{J+1} dimensional tensor where \eqn{J}
      is the number of parents.}
    \item{getCPF}{\code{signature()}. Gets the conditional proability
      frame, which is the CPT converted into a data frame with addition
      information about the parent configurations.}
    \item{getETFrame}{\code{signature()}. Get the effective theta table
      from the \code{$rule} and displays it as a data frame with
      additional information about the parent states.}

  }
}
\section{Parents and Parent Value Matrix}{


}
\section{Constructing the Conditional Probability Tensor}{


}
\section{Calculating the Score}{



}
\section{Parameters}{


}
\section{Matching Parameters to Data}{

}
\section{Constructor}{

}
\section{Superclasses and Subclasses}{


}

\author{Russell Almond}
\seealso{

}
\keyword{classes}
