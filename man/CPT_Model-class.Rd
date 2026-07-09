\name{CPT_Model}
\Rdversion{1.1}
\docType{class}
\alias{CPT-class}
\alias{CPT_Model-class}
\alias{CPT_Model}
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
    \item{high2low}{A logical value.  If true (default), then the states are
      ordered from highest to lowest.}
    \item{device}{The \code{\link[torch]{torch_device}} that should be used
      for building tensors.}
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
      \var{QQ}=TRUE),\var{guess}=NA,\var{slip}=NA,\var{high2low}=TRUE,
      \var{device}=CPTtorch_device()}.  Creates a new CPT model.  See Construtor
      section below.
    }
    \item{forward}{\code{signature()}. Calculates the CPT from the
      current parameters.  Calling this function forces a recalculation.}
    \item{lscore}{\code{signature(datatab:tensor)}. Calculates
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

  The third argument to the constructor is a list of parents.
  This should be a named list, with the names corresponding to the names of the
  parent variables.  The values should be a numeric vector whose length is
  related to the number of the states of each parent.  The vector should be named as
  well, giving the state names.  Finally, it should respect the value of the
  \code{high2low} flag.  The function \code{\link{as_Tvallist}} generates this list
  from more compact notation.


}
\section{Constructing the Conditional Probability Tensor}{

  The \code{$forward()} method calculates the conditional probability tensor.
  In particular, it calls the \code{rule$forward()} method to calculate the
  effective theta tensor which is then passed to the \code{link$forward()}
  method to calculate the CPT.  It uses the parameters which are stored in the
  \code{rule} and \code{link} objects.

  The method \code{$getCPT()} gets either a cached value of the CPT or
  recalculates it if there has been a parametere change since the last
  time it was calculated.  The method \code{$cptBuiltp()} returns a logical value
  describing whether or not there is a cached value of the CPT.  The method \code{$getCPTframe()} embeds the CPT in
  a data frame with the rows labeled according to the configurations of the parent values.

  The method \code{$getETFrame()} gets just the effective theta values (the
  output of the \code{rule} and displays them in a data frame.

}
\section{Calculating the Score}{

  The score is a measure of how well the CPT fits with a particular set of data.
  The method \code{$lscore(dattab)} calculates the log probability of observing
  the data in the \code{dattab} tensor given the current CPT.  The \code{dattab}
  argument should have the same size and shape as the CPT.

}
\section{Parameters}{

  The parameters reside in the \code{rule} and \code{link} parts of the model.
  From the \code{\linkS4class{CombinationRule}}, it gathers the \code{aVec} and \code{bVec}
  parameters.  From the \code{\linkS4class{CPT_Link}}, it gathers the \code{sVec},
  \code{guessP} and \code{slipP} parameters.  If any of these parameters are null,
  then they are dropped from the list returned by \code{$params()}.


}
\section{Matching Parameters to Data}{

  The method \code{$step(datatab,r)} runs an optimizer for the parameters for
  \eqn{r} steps to match the parameters to the data.  The method
  \code{$buildOptimizer()} constructs the optimizer; it is called if needed by
  \code{$step()}.

}
\section{Constructor}{

  \code{CPT_Model} objects are constructed by a call to
    \code{CPT_Model$new(ruletype,linktype,parents=list(),states=character(),
        QQ=TRUE,guess=NA,slip=NA,high2low=TRUE,device=CPTtorch_device())}.

  \describe{
    \item{ruletype}{A character scalar giving the name of the combination rule.
      See \code{\link{getRule}}.}
    \item{linktype}{A character scalar giving the name of the link.  See
      \code{\link{getLink}}.}
    \item{parents}{A list of numeric vectors giving numeric values for the parent states.
      See Parents section above, and \code{\link{as_Tvallist}}.}
    \item{states}{A character vector giving the names of the output states.}
    \item{QQ}{An inner Q-matrix.  Either a logical matrix showing which elements
      of the appropriate parameter are used and unused, or \code{TRUE} indicating all
      parameters are used.  See \code{\linkS4class{CombinationRule}}.}
    \item{guess,slip}{Numeric values giving the guessing and slipping parameters (see
      \code{linkS4class{CPT_Link}}) or \code{NA} indicating that they are not used.}
    \item{high2low}{A logical value indicating if states are ordered highest-to-lowest (true)
      or lowest-to-highest (false).}
    \item{device}{This should be either \code{\link[torch]{torch_device}("cpu")} or
      \code{\link[torch]{torch_device}("gpu")} indicates where the tensors
      should be created.}
  }

}
\section{Superclasses and Subclasses}{

  The \code{CPT_Model} is a subclass of \code{torch::\link[torch]{nn_module}}.
  In particular, the \code{$forward()} method calculates the CPT.

  There are no subclasses, however, it incorporates \code{\linkS4class{CombinationRule}}
  and \code{\linkS4class{CPT_Link}} by containment.

}
\author{Russell Almond}
\seealso{
  \code{\link[torch]{nn_module}}, \code{\link{as_Tvallist}},
  \code{\linkS4class{CombinationRule}}, \code{\linkS4class{CPT_Link}}.
}
\keyword{classes}
