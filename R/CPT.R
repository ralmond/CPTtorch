# !diagnostics suppress=self,private,super

deviance_loss <- function(datatab,cpt,ccbias=0) {
  datatab <- torch_reshape(datatab,dim(cpt))$add(cpt,ccbias)
  cpt$log()$mul_(datatab)$sum()$neg_()
}
penalty_fun = function(params,which,bias) {
  if (!is.null(params[[which]]))
    params[[which]]$square()$sum()$mul_(bias)
  else
    torch_tensor(0,torch_double())
}

build_loss_fun <- function (ccbias,penalties) {
  function(dattab,cpt,params) {
    result <- deviance_loss(dattab,cpt,ccbias)
    for (ipar in names(penalties)) {
      result <- result$add_(
        penalty_fun(params,ipar,penalties[[ipar]])
      )
    }
    result
  }
}

CPT_Model <-
  torch::nn_module(
    classname="CPT_Model",
#    inherit=nn_Module,
    rule=NULL,
    link=NULL,
    ccbias=10,
    optimizer=NULL,
    oconstructor="optim_adam",
    oparams=list(lr=.1),
    lossfn=NULL,
    initialize = function(ruletype,linktype,parents=list(),states=character(),
                          QQ=TRUE,guess=NA,slip=NA,high2low=FALSE) {
      self$parentVals <- parents
      self$stateNames <- states
      link <- getLink(linktype)
      if (is.null(link)) abort("Unknown link type",linktype)
      self$link <- link$new(length(states),guess,slip,high2low)

      rule <- getRule(ruletype)
      if (is.null(rule)) abort("Unknown rule type",ruletype)
      self$rule <- rule$new(self$parentVals,
                            self$link$etWidth(),
                            QQ,high2low)
    },
    forward = function () {
      private$cpt <- self$link$forward(self$rule$forward())
      private$cpt
    },
    numparams = function () {
      length(self$rule$aVec) + length(self$rule$bVec) +
        length(self$link$sVec) + length(self$link$guessP) +
        length(self$link$slipP)
    },
    params = function() {
      plist <- list(
      aVec=self$rule$aVec,bVec=self$rule$bVec,
      sVec=self$link$sVec,gP=self$link$guessP,
      sP=self$link$sP)
      plist[!sapply(plist,is.null)]
    },
    AIC = function(datatab) {
      as_array(self$deviance(datatab)) + 2*self$numparams()
    },
    getCPT = function () {
      if (is.null(private$cpt)) self$forward()
      private$cpt$reshape(private$shape)
    },
    cptBuiltp = function () {
      !is.null(private$cpt)
    },
    getCPTframe = function () {
      if (is.null(private$cpt)) self$forward()
      frame <- data.frame(cartesian_prod(self$parentStates),
                          as_array(private$cpt))
      names(frame) <- c(names(self$parentNames),self$stateNames)
      frame
    },
    getETframe = function () {
      if (is.null(self$rule)) return(NULL)
      frame <- self$rule$getETframe()
      names(frame) <- c(names(self$parentNames),self$stateNames[1L:self$link$etWidth()])
      frame
    },
    deviance=function(dattab) {
      deviance_loss(dattab,self$forward(),self$ccbias)
    },
    buildOptimizer = function() {
      self$cache <- NULL
      self$lossfn <-
        jit_trace(build_loss_fun(self$ccbias,
                                 self$penalities),
          torch_ones(self$shp),
          self$forward(),
          self$params())
      self$optimizer <-
        do.call(self$oconstructor,
                c(list(self$params()),self$oparams))
      self$optimizer
    },
    step = function (datatab,r=1L) {
      if (is.null(self$optimizer)) self$buildOptimizer()
      if (is.null(self$lossfn)) {
        self$cache <- NULL
        self$lossfn <-
        jit_trace(build_loss_fun(self$ccbias,self$penalties),
          datatab,self$forward(),self$params())
      }
      for (rr in 1:r) {
        self$optimizer$zero_grad()
        self$lossfn(dattab=datatab,self$forward(),
                    self$params())$backward(retain_graph=TRUE)
        self$optimizer$step()
      }
      self$cache <- NULL
      self$deviance(datatab)
    },
    private=list(
        parents=list(),
        states=character(),
        shape=c(1L,1L),
        cpt=NULL,
        pbiases=list(aVec=NULL,bVec=NULL,sVec=NULL,
                     gP=NULL,sP=NULL)
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
          self$link$linkScale <- value
          private$cpt <- NULL
          invisible(self)
        },
        slip = function(value) {
          if (missing(value)) return (self$link$slip)
          if (xor(is.na(value),is.null(self$link$slip))) {
            self$optimizer <- NULL
            self$lossfn <- NULL
          }
          self$link$slip <- value
          private$cpt <- NULL
          invisible(self)
        },
        guess = function(value) {
          if (missing(value)) return (self$link$guess)
          if (xor(is.na(value),is.null(self$link$guess))) {
            self$optimizer <- NULL
            self$lossfn <- NULL
          }
          self$link$guess <- value
          private$cpt <- NULL
          invisible(self)
        },
        parentVals = function (value) {
          if (missing(value)) return(private$parents)
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
          names(private$parents)
        },
        parentStates = function() {
          lapply(private$parents,names)
        },
        stateNames = function (value) {
          if (missing(value)) return(private$states)
          if (length(private$states) != length(value)) {
            self$optimizer <- NULL
            self$lossfn <- NULL
          }
          private$states <- value

          private$shape[length(private$shape)] <- length(value)
          if (!is.null(self$link)) {
            self$link$K <- length(value)
            if (!is.null(self$rule))
              self$rule$setDim(K=self$link$etWidth())
          }
          private$cpt <- NULL
          invisible(self)
        },
        shp = function() {
          as.numeric(private$shape)
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
          if (!is.logical(value))
            abort("High2low field must be a logical value.")
          self$optimizer <- NULL
          self$lossfn <- NULL
          private$cpt <- NULL
          self$link$high2low <- value
          self$rule$high2low <- value
        },
        penalties=function(value) {
          if (missing(value))
            return(private$pbias[!is.null(private$pbias)])
          if (!is.list(value))
            stop("Value must be a list.")
          private$pbias <- value
        }
    )
)

fit2table<- function(model,dattab,
                     maxit=100L,tolerance=.0001,
                     stepit=1L,log=NULL) {
  rit <- 0L
  model$buildOptimizer()
  dev <- as.numeric(model$deviance(dattab))
  if (!is.null(log)) {
    cat("Cycle:",rit,"Deviance:",dev,"\n")
    for (pname in log) {
      print(pname)
      print(model$params()[[pname]])
    }
  }
  while (rit < maxit) {
    olddev <- dev
    dev <- as.numeric(model$step(dattab,stepit))
    rit <- rit+stepit
    if (!is.null(log)) {
      cat("Cycle:",rit,"Deviance:",dev,"\n")
      for (pname in log) {
        print(pname)
        if (pname=="cpt")
          print(model$getCPT())
        else
          print(model$params()[[pname]])
      }
    }
    if (abs(olddev-dev) < tolerance) break
  }
  return (rit < maxit)
}

