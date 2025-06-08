# !diagnostics suppress=self,private,super


CPT_Link <- torch::nn_module(
    classname="CPT_Link",
    link=identity,
    sVec=NULL,
    slipP=NULL,
    guessP=NULL,
    etWidth=function() {self$K-1},
    high2low=FALSE,
    private=list(
      k=NA,
      stype=NULL
    ),
    initialize=function(nstates,guess=NA,slip=NA,high2low=FALSE,...) {
      self$K <- nstates
      self$guess <- guess
      self$slip <- slip
      self$high2low <- high2low
      if (!is.null(private$stype)) {
        self$linkscale <- defaultParameter10(private$stype)
      }
    },
    leakmat=function() {
      if (is.null(self$guessP) && is.null(self$slipP)) return (NULL)
      result <- torch_eye(self$K)
      if (!is.null(self$guessP)) {
        gm <- torch_guessmat(self$K,self$guess)
        if (self$high2low) gm <- gm$flip_lr()
        result <- result$matmul_(gm)
      }
      if (!is.null(self$slipP)) {
        sm <- torch_slipmat(self$K,self$slip)
        if (self$high2low) sm <- sm$flip_lr()
        result <- result$matmul_(sm)
      }
      result
    },
    forward=function(et) {
      cpt <- self$link(et)

      leakmat <- self$leakmat()
      if (is.null(leakmat)) {
        cpt <- cpt$matmul_(leakmat)
      }

      if (isTRUE(self$high2low))
        cpt <- cpt$fliplr_()

      cpt

    },
   active=list(
       K=function(value){
         if (missing(value)) return (private$k)
         if (!is.null(private$stype)) {
           olddim <- pTypeDim(private$stype)
           private$stype <- setpTypeDim(private$stype,K=private$k)
           if (!isTRUE(all.equal(olddim, pTypeDim(private$stype))))
             self$linkScale <- defaultParameter10(private$stype)
         }
       },
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
          self$sVec <- nn_parameter(natpar2tvec(private$sType,
                                                as_torch_tensor(value)))
          invisible(self)
       },
       guess=function(value) {
         if (missing(value))
           if (is.null(self$guessP)) return(NA)
           else return (torch_sigmoid(self$guessP)$div_(2))
         if (is.na(value) || is.null(value) || isFALSE(value))
           self$guessP <- NULL
         else {
           if (as_array(value) < 0 || as_array(value)>.5)
             abort("Guessing paramter must be between 0 and .5.")
           self$guessP <- nn_parameter(as_torch_tensor(value)$mul_(2)$logit_())
         }
       },
       slip=function(value) {
         if (missing(value))
           if (is.null(self$slipP)) return(NA)
           else return (torch_sigmoid(self$slipP)$div_(2))
         if (is.na(value) || is.null(value) || isFALSE(value))
           self$slipP <- NULL
         else {
           if (as_array(value) < 0 || as_array(value)>.5)
             abort("Slipping paramter must be between 0 and .5.")
           self$slipP <- nn_parameter(as_torch_tensor(value)$mul_(2)$logit_())
         }
       }
    )
)



LinkedList <- new.env()
getLink <- function (linkname) {
  if (is(linkname,"CPT_Link")) return(linkname)
  LinkedList[[linkname]]
}
setLink <- function (linkname,value) {
  if (!is.null(value) && !is(value,"CPT_Link"))
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

PotentialLink <- torch::nn_module(
    classname="PotentialLink",
    inherit=CPT_Link,
    scale=NULL,
    etWidth=function() {self$K},
    link=function(et) {
      torch_simplexify(et)
    },
    private=list(
      stype=NULL
    )
)

StepProbsLink <- torch::nn_module(
    classname="StepProbsLink",
    inherit=CPT_Link,
    scale=NULL,
    etWidth=function() {self$K-1},
    link=function(et) {
      torch_simplexify_(
          torch_hstack(list(et,torch_ones(ncol(et),1)))$
          cumprod_(2))
    },
    private=list(
      stype=NULL
    )
)

cuts2simplex <- function (et)
  torch_hstack(list(torch_zeros(ncol(et),1),et,torch_ones(ncol(et),1)))$
    cummax_(2)$diff_()$clip_(0,1)


DifferenceLink <- torch::nn_module(
    classname="DifferenceLink",
    inherit=CPT_Link,
    scale=NULL,
    link=cuts2simplex,
    etWidth=function() {self$K-1},
    private=list(
      stype=NULL
    )
)

SoftmaxLink <- torch::nn_module(
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


GradedResponseLink <- torch::nn_module(
    classname="GradedResponseLink",
    inherit=DifferenceLink,
    D=torch_tensor(1.7),
    scale=NULL,
    etWidth=function() {self$K-1},
    link=function(et) {
      torch_hstack(list(torch_zeros(ncol(et),1),nnf_sigmoid(et$mul_(self$D)),
                        torch_ones(ncol(et),1)))$
        cummax_(2)$diff_()$clip_(0,1)
    },
    private=list(
      stype=NULL
    )
)

PartialCreditLink <- torch::nn_module(
    classname="PartialCreditLink",
    inherit=StepProbsLink,
    scale=NULL,
    etWidth=function() {self$K-1},
    link=function(et) {
      torch_simplexify(
          torch_hstack(list(et,torch_zeros(ncol(et),1)))$
          cumsum_(2)$mul_(self$D)$nnf_sigmoid())
    },
    private=list(
      stype=NULL
    )
)



GaussianLink <- torch::nn_module(
    classname="GaussianLink",
    inherit=CPT_Link,
    scale=NULL,
    etWidth=function() {1},
    link=function(et) {
      pt <- torch_pnorm(self$cuts$sub_(et)$div_(self$linkScale))
      torch_hstack(list(torch_zeros(ncol(et),1),pt,torch_ones(ncol(et),1)))$
        diff_(2)
    },
    private=list(
        stype=setpTypeDim(PType("pos",1)),
        cuts=NULL
    ),
    active=list(
        Cuts = function() {
          if (is.null(private$cuts)) {
            private$cuts <-torch_tensor(matrix(qnorm((1L:(self$k-1L))/self$k),
                                               1L,self$k))
          }
          private$cuts
        }
    )
)

## NoisyLink <- nn_module(
##     classname="NoisyLink",
##     inherit=CPT_Link,
##     scale=NULL,
##     etWidth=function() {K-1},
##     link=function(et) {
##       torch_hstack(et,torch_sum(et,2)$neg_()$add_(1))$
##         matmul_(self$linkScale)
##     },
##     private=list(
##         stype=PType("cpMat",c(K,K))
##     )
## )

addcolk <- function (et)
  torch_hstack(et,torch_sum(et,2)$neg_()$add_(1))

SlipLink <- torch::nn_module(
    classname="SlipLink",
    inherit=CPT_Link,
    scale=NULL,
    etWidth=function() {self$K-1},
    link=function(et) {
      cuts2simplex(et)$matmul_(torch_slipmat(self$K,self$linkScale))
    },
    private=list(
        stype=PType("unit",c(1))
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



