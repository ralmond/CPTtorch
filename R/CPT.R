CPT_Model <-
  torch::nn_module(
    classname="CPT_Model",
#    inherit=nn_Module,
    rule=NULL,
    link=NULL,
    ccbias=10,
    abias=0,
    bbias=0,
    optimizer=NULL,
    lossfn=NULL,
    initialize = function(ruletype,linktype,parents=list(),states=character(),
                          QQ=TRUE,guess=NA,slip=NA,high2low=FALSE) {
      self$parentVals <- parents
      self$stateNames <- states
      link <- getLink(linktype)
      if (is.null(link)) abort("Unknown link type",lintype)
      self$link <- link$new(length(states),guess,slip,high2low)

      rule <- getLink(ruletype)
      if (is.null(rule)) abort("Unknown rule type",lintype)
      self$rule <- rule$new(self$parentVals,
                            self$link$etWidth(),
                            QQ,high2low)
    },
    forward = function () {
      private$cpt <- self$link$forward(self$rule$forward())
      private$cpt
    },
    deviance = function (datatab) {
      cpt <- self$forward()
      datatab <- datatab$reshape_(dim(cpt))$add_(cpt,self$ccbias)
      score <- cpt$log()$mul_(datatab)$sum_(1:2)$neg_()
      if (self$abias >0)
        score <- score$add_(self$rule$aVec$square()$sum_(1)$mul_(self$abias))
      if (self$bbias >0)
        score <- score$add_(self$rule$bVec$square()$sum_(1)$mul_(self$bbias))
      score
    },
    numparams = function () {
      length(self$rule$aVec) + length(self$rule$bVec) +
        length(self$link$sVec) + length(self$link$guessP) +
        length(self$link$slipP)
    },
    AIC = function(datatab) {
      as_array(self$deviance(datatab)) + 2*self$numparams()
    },
    getCPT = function () {
      if (is.null(private$cpt)) self$forward(NULL)
      private$cpt$reshape_(private$shape)
    },
    getCPFrame = function () {
      if (is.null(private$cpt)) self$forward(NULL)
      frame <- data.frame(cartesian_prod(self$parentNames),
                          as_array(private$cpt))
      names(frame) <- c(names(self$parentNames),self$stateNames)
      frame
    },
    getETframe = function () {
      if (is.null(private$rule)) return(NULL)
      frame <- getETframe()
      names(frame) <- c(names(self$parentNames),self$stateNames[1L:ncol(et)])
      frame
    },
    buildOptimizer = function(constructor=optim_adam, ...) {
      self$cache <- NULL
      self$lossfn <- jit_trace(self$deviance,torch_ones(private$shape))
      self$optimizer <- exec(constructor,self$parameters(),!!!...)
      self$optimizer
    },
    step = function (datatab,r=1L) {
      if (is.null(self$optimizer)) self$buildOptimizer()
      if (is.null(self$lossfn)) {
        self$cache <- NULL
        self$lossfn <- jit_trace(self$deviance,datatab)
      }
      for (rr in 1:r) {
        self$optimizer$zero_grad()
        self$lossfn(datatab)$backward()
        self$optimizer$step()
      }
      self$cache <- NULL
      self$deviance(datatab)
    },
    private=list(
        parents=list(),
        states=character(),
        shape=c(1L,1L),
        cpt=NULL
    ),
    active=list(
        aMat = function (value) {
          if (missing(value)) return(self$rule$aMat)
          self$rule$aMat <- value
          private$cpt <- NULL
          invisible(self)
        },
        bMat = function (value) {
          if (missing(value)) return(self$rule$bMat)
          self$rule$bMat <- value
          private$cpt <- NULL
          invisible(self)
        },
        linkScale = function(value) {
          if (missing(value)) return (self$link$linkScale)
          self$rule$linkScale <- value
          private$cpt <- NULL
          invisible(self)
        },
        slip = function(value) {
          if (missing(value)) return (self$link$slip)
          if (xor(is.na(value),is.na(self$rule$slip))) {
            self$optimizer <- NULL
            self$lossfn <- NULL
          }
          self$rule$slip <- value
          private$cpt <- NULL
          invisible(self)
        },
        guess = function(value) {
          if (missing(value)) return (self$link$guess)
          if (xor(is.na(value),is.na(self$rule$guess))) {
            self$optimizer <- NULL
            self$lossfn <- NULL
          }
          self$rule$guess <- value
          private$cpt <- NULL
          invisible(self)
        },
        parentVals = function (value) {
          if (missing(value)) private$parents
          private$parents <- as_Tvallist(value)
          private$shape <- c(sapply(private$parents,length),
                             private$shape[length(private$shape)])
          if (!is.null(self$rule)) self$rule$setParents(private$parents)
          self$lossfn <- NULL
          self$optimizer <- NULL
          private$cpt <- NULL
          invisible(self)
        },
        parentNames = function () {
          lapply(private$parents,names)
        },
        stateNames = function (value) {
          if (missing(value)) private$states
          if (length(private$states) != length(value)) {
            self$optimizer <- NULL
            self$lossfn <- NULL
          }
          private$states <- value

          private$shape[length(private$shape)] <- length(value)
          if (!is.null(self$link)) {
            self$link$K <- length(states)
            if (!is.null(self$rule))
              self$rule$setDim(K=self$rule$etWidth())
          }
          private$cpt <- NULL
          invisible(self)
        },
        QQ = function (value) {
          if (is.null(self$rule)) return(NULL)
          if (missing(value)) return(self$rule$QQ)
          self$rule$QQ <- value
          self$lossfn <- NULL
          self$optimizer <- NULL
          private$cpt <- NULL
          invisible(self)
        },
        high2low = function(value) {
          if (missing(value)) return(self$link$high2low)
          if (!is.logical(high2low))
            abort("High2low field must be a logical value.")
          self$optimizer <- NULL
          self$lossfn <- NULL
          private$cpt <- NULL
          self$link$high2low <- value
          self$rule$high2low <- value
        }
    )
)
