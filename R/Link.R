CPT_Link <- nn_module(
    classname="CPT_Link",
    link=identity,
    scale=NULL,
    etWidth=function() {self$K-1},
    private=list(
      k=NA,
      stype=NULL,
    ),
    initialize=function(states) {
      if (is.character(states))
        private$k <- length(states)
      else private$k <- states
      if (!is.null(private$stype)) {
        private$stype <- setpTypeDim(private$stype,K=private$k)
        self$linkScale <- torch_tensor(defaultParameter(private$stype))
      }
    },
    forward=function(input) {
      self$link(input)
    },
   active=list(
       K=function(){private$k},
       sType=function(value) {
          if (missing(value)) return (private$stype)
          if (!is(value,"PType"))
            abort("The scale type field must be a PType object.")
          private$stype <- value
          if (!is.na(private$K)) {
            olddim <- pTypeDim(private$stype)
            private$stype <- setpTypeDim(private$stype,K=private$k)
            if (!isTRUE(all.equal(olddim, pTypeDim(private$stype))))
              self$linkScale <- torch_tensor(defaultParameter(private$stype))
          }
          invisible(self)
       },
       linkScale=function(value) {
          if (missing(value)) {
            if (is.null(self$scale)) return (NULL)
            return (tvec2natpar(private$stype,self$scale))
          }
          pcheck <- checkParam(private$sType,value)
          if (!isTRUE(pcheck))
            abort("Illegal link scale parameter value,",pcheck,".")
          self$sVec <- nn_parameter(pMat2tvec10(private$sType,value))
          invisible(self)
        }
    )
}



LinkedList <- new.env()
getLink <- function (linkname) {
  LinkedList(linkname)
}
setLink <- function (linkname,value) {
  if (!is.null(value) || !is(ptype,"CPT_Link"))
    stop("Second argument must be a CPT_Link or NULL.")
  LinkedList[[linkname]] <- value
}
availableLinks <- function() {
  names(LinkedList)
}


### Link Functions
### These are a function of three arguments
### et -- a table of effective thetas, one for row for each
###       configuration of parent variables, and one column for
###       each child state except for the last.
### k -- the number of states of the child variable.
### linkScale --- a scale parameter used by some link functions.

### Zero variants, do not have the obsLevel, as it is generally not
### needed and may speed up some loops.

### obsLevel --- a list of names of the observables, assume to be
### sorted from highest to lowest.

### It returns a conditional probability table.

PotentialLink <- nn_module(
    classname="PotentialLink",
    inherits=CPT_Link,
    scale=NULL,
    etWidth=function() {self$K},
    link=function(et) {
      torch_2simplex(et)
    },
    private=list(
      stype=NULL,
      )
)

StepProbsLink <- nn_module(
    classname="StepProbsLink",
    inherits=CPT_Link,
    scale=NULL,
    etWidth=function() {self$K-1},
    link=function(et) {
      torch_2simplex(
          torch_hstack(list(et,torch_ones(ncol(et),1)))$
          fliplr_()$cumprod_(2)$fliplr_())
    },
    private=list(
      stype=NULL,
      )
)

DifferenceLink <- nn_module(
    classname="DifferenceLink",
    inherits=CPT_Link,
    scale=NULL,
    etWidth=function() {self$K-1},
    link=function(et) {
      torch_hstack(list(torch_zeros(ncol(et),1),et,torch_ones(ncol(et),1)))$
        cummax_(2)$diff_()$clip_(0,1)
    },
    private=list(
      stype=NULL,
      )
)

SoftmaxLink <- nn_module(
    classname="SoftmaxLink",
    inherits=CPT_Link,
    scale=NULL,
    D=torch_tensor(1.7)
    etWidth=function() {self$K},
    link=function(et) {
      nnf_softmax(et$mul_(self$D),2)
    },
    private=list(
      stype=NULL,
      )
)


GradedResponseLink <- nn_module(
    classname="GradedResponseLink",
    inherits=DifferenceLink,
    D=torch_tensor(1.7)
    scale=NULL,
    etWidth=function() {self$K-1},
    link=function(et) {
      torch_hstack(list(torch_zeros(ncol(et),1),nnf_sigmoid(et$mul_(self$D)),
                        torch_ones(ncol(et),1)))$
        cummax_(2)$diff_()$clip_(0,1)
    },
    private=list(
      stype=NULL,
      )
)

PartialCreditLink <- nn_module(
    classname="PartialCreditLink",
    inherits=StepProbsLink,
    scale=NULL,
    etWidth=function() {self$K-1},
    link=function(et) {
      torch_2simplex(
          torch_hstack(list(et,torch_zeroes(ncol(et),1)))$
          fliplr_()$cumsum_(2)$fliplr_()$
          mul_(self$D)$nnf_sigmoid())
    },
    private=list(
      stype=NULL,
      )
)



GaussianLink <- nn_module(
    classname="GaussianLink",
    inherits=CPT_Link,
    scale=NULL,
    etWidth=function() {1},
    link=function(et) {
      pt <- torch_pnorm(self$cuts$sub_(et)$div_(self$linkScale))
      torch_hstack(list(torch_zeros(ncol(et),1),pt,torch_ones(ncol(et),1)))$
        diff_(2)
    },
    private=list(
        stype=setpTypeDim(PType("pos",1))
        cuts=NULL,
        ),
    active=list(
        Cuts = function() {
          if (is.null(private$cuts)) {
            private$cuts <-torch_tensor(matrix(qnorm(((self$k-1L):1L)/self$k),
                                               1L,self$k))
          }
          private$cuts
        })
)

NoisyLink <- nn_module(
    classname="NoisyLink",
    inherits=CPT_Link,
    scale=NULL,
    etWidth=function() {K-1},
    link=function(et) {
      torch_hstack(et,torch_sum(et,2)$neg_()$add_(1))$
        matmul_(self$linkScale)
    },
    private=list(
        stype=PType("cpMat",c(K,K))
    )
)

SlipLink <- nn_module(
    classname="SlipLink",
    inherits=CPT_Link,
    scale=NULL,
    etWidth=function() {K-1},
    link=function(et) {
      torch_hstack(et,torch_sum(et,2)$neg_()$add_(1))$
        matmul_(torch_slipmat(self$K,self$linkScale))
    },
    private=list(
        stype=PType("unit",c(1))
    )
)

GuessLink <- nn_module(
    classname="GuessLink",
    inherits=CPT_Link,
    scale=NULL,
    etWidth=function() {K-1},
    link=function(et) {
      torch_hstack(et,torch_sum(et,2)$neg_()$add_(1))$
        matmul_(torch_guessmat(self$K,self$linkScale))
    },
    private=list(
        stype=PType("unit",c(1))
    )
)

GuessSlipLink <- nn_module(
    classname="GuessSlipLink",
    inherits=CPT_Link,
    scale=NULL,
    etWidth=function() {K-1},
    link=function(et) {
      torch_hstack(et,torch_sum(et,2)$neg_()$add_(1))$
        matmul_(torch_guessmat(self$K,self$linkScale[1]))$
        matmul_(torch_slipmat(self$K,self$linkScale[2]))
    },
    private=list(
        stype=PType("unit",c(2))
    )
)



setLink("Potential",PotentialLink)
setLink("StepProbs",StepProbsLink)
setLink("Difference",DifferenceLink)
setLink("GradedResponse",GradedResponseLink)
setLink("PartialCredit",PartialCreditLink)
setLink("Normal",GaussianLink)
setLink("Gaussian",GaussianLink)
setLink("Softmax",SoftmaxLink)
setLink("Noisy",NoisyLink)
setLink("Slip",SlipLink)
setLink("Guess",GuessLink)
setLink("GuessSlip",GuessSlipLink)



