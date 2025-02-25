

### Calculates the effective theta values for a skill variable
### with the argument number of levels
effectiveTheta10 <- function (nlevels) {
  torch_tensor(rev(qnorm((2*(1:nlevels)-1)/(2*nlevels),0,1)))
}

eThetaFrame <- function (skillLevels, lnAlphas, beta, rule="Compensatory") {
  pdims <- sapply(skillLevels,length)
  tvals <- lapply(pdims,effectiveThetas)
  thetas <- do.call("expand.grid",tvals)
  names(thetas)<- paste(names(skillLevels),"theta",sep=".")
  etheta <- do.call(rule,list(thetas,exp(lnAlphas),beta))
  data.frame(expand.grid(skillLevels),thetas, Effective.theta=etheta)
}

eTheta10 <- function (dims) {
  torch_cartisian_product(lapply(dims, \(d) effectiveTheta10(d)))
}

buildeTheta10 <- function(parents) {
  if (is.numeric(parents)) return(eTheta10(parents))
  if (all(sapply(parents,is.character)))
    return(eTheta10(sapply(parents,length)))
  if (all(sapply(parents,is.numeric)))
    return(torch_cartisian_product(parents))
  abort("Input to buildeTheta10 must be a vector of dims, a list of state names or a list of state values.")
}


############
## New OO version

RuleASB <- nn_module(
    classname="RuleASB",
    inherit = nnModule,
    prescale <- torch_mul,
    summary <- torch_sum,
    postscale <- torch_add,
    initialize = function (parents, nstates, QQ=true, ...) {
      self$eTheta <- buildeTheta10(parents)
      private$SJK <- c(as.list(self$eTheta$shape()),list(nstates))
      names(private$SJK) <- c("S","J","K")
      private$atype <- exec(setpTypeDim,private$atype,!!!private$SJK)
      private$btype <- exec(setpTypeDim,private$btype,!!!private$SJK)
      self$QQ <- QQ
      self$aMat <- torch_tensor(defaultParameter(private$atype))
      self$bMat <- torch_tensor(defaultParameter(private$btype))
    },
    forward = function() {
      if (!isTRUE(self$QQ)) {
        exec(self$postscale,
             genMMt(self$eTheta,self$aMat,self$prescale,self$summary),
             self$bMat)
      } else {
        exec(self$postscale,
             genMMt(self$eTheta,self$aMat,self$prescale,self$summary),
             self$bMat)
      }
    },
    private=list(
        atype=PType("real",c(K,J)),
        btype=PType("real",c(K,1)),
        SJK=NULL,
        cache$NULL
    ),
    active = list(
        aType=function(value) {
          if (missing(value)) return (private$atype)
          if (!is(value,"PType"))
            abort("The aType field must be a PType object.")
          private$atype <- value
          if (!is.null(private$SJK)) {
            exec(setpTypeDim,private$atype,!!!private$SJK)
            self$aMat <- torch_tensor(defaultParam(private$atype))
          }
        },
        bType=function(value) {
          if (missing(value)) return (private$btype)
          if (!is(value,"PType"))
            abort("The bType field must be a PType object.")
          private$btype <- value
          if (!is.null(private$SJK)) {
            exec(setpTypeDim,private$btype,!!!private$SJK)
            self$bMat <- torch_tensor(defaultParam(private$btype))
          }
        },
        QQ=function(value) {
          if (missing(value)) return (whichUsed(private$atype))
          if (!is.logical)
            abort("The QQ field must be a logical matrix or TRUE")
          whichUsed(private$aType) <- value
        },
        aMat=function(value) {
          if (missing(value)) {
            if (is.null(self$aVec)) return (NULL)
            return (tvect2pMat10(private$aType,self$aVec))
          }
          pcheck <- checkParam(private$aType,value)
          if (!isTRUE(pcheck))
            abort("Illegal A parameter value,",pcheck,".")
          private$cache <- NULL
          self$aVec <- nn_parameter(pMat2tvec10(private$aType,value))
        },
        bMat=function(value) {
          if (missing(value)) {
            if (is.null(self$bVec)) return (NULL)
            return (tvect2pMat10(private$bType,self$bVec))
          }
          pcheck <- checkParam(private$bType,value)
          if (!isTRUE(pcheck))
            abort("Illegal A parameter value,",pcheck,".")
          private$cache <- NULL
          self$bVec <- nn_parameter(pMat2tvec10(private$bType,value))
        },
        cpt = function() {
          if (is.null(private$cache))
            private$cache <- self$forward()
          private$cache
        },
        paramsupdated = function() { is.null(private$cache)}
    )
}


RuleBSA <- nn_module(
    classname="RuleBSA",
    inherit = RuleASB,
    prescale <- torch_sum,
    summary <- torch_max,
    postscale <- torch_mul,
    forward = function() {
      if (!isTRUE(self$QQ)) {
        exec(self$postscale,
             genMMt(self$eTheta,self$bMat,self$prescale,self$summary),
             self$aMat)
      } else {
        exec(self$postscale,
             genMMt(self$eTheta,self$bMat,self$prescale,self$summary),
             self$aMat)
      }
    },
    private=list(
        atype=PType("real",c(K,1)),
        btype=PType("real",c(K,J)),
        ),
    active = list(
        QQ=function(value) {
          if (missing(value)) return (whichUsed(private$btype))
          if (!is.logical)
            abort("The QQ field must be a logical matrix or TRUE")
          whichUsed(private$bType) <- value
        }
    )
}

RuleBAS <- nn_module(
    classname="RuleBAS",
    inherit = RuleASB,
    prescale <- torch_sum,
    summary <- torch_max,
    postscale <- torch_mul,
    forward = function() {
      if (!isTRUE(self$QQ)) {
        genMMt(exec(self$prescale,self$eTheta,self$bMat,self$prescale),
               self$aMat,self$postscale,self$summary)
      } else {
        exec(self$postscale,
             genMMt(self$eTheta,self$bMat,self$prescale,self$summary),
             self$aMat)
      }
    },
    private=list(
        atype=PType("real",c(K,J)),
        btype=PType("real",c(K,J)),
        ),
    active = list(
        QQ=function(value) {
          if (missing(value)) return (whichUsed(private$btype))
          if (!is.logical)
            abort("The QQ field must be a logical matrix or TRUE")
          whichUsed(private$bType) <- value
        }
    )
}

RuleConstB <- nn_module(
    classname="RuleConstB"
    inherit = RuleASB,
    forward = function() {self$bMat},
    private=list(
        atype=PType("const",c(K,J)),
        btype=PType("const",c(K,J)),
        ),
    active = list(
        QQ=function(value) {
          if (missing(value)) return (whichUsed(private$btype))
          if (!is.logical)
            abort("The QQ field must be a logical matrix or TRUE")
          whichUsed(private$bType) <- value
        }
    )
}

RuleConstA <- nn_module(
    classname="RuleConstA",
    inherit = RuleASB,
    forward = function() {self$aMat},
    private=list(
        atype=PType("const",c(K,J)),
        btype=PType("const",c(K,J)),
        ),
    active = list(
        QQ=function(value) {
          if (missing(value)) return (whichUsed(private$atype))
          if (!is.logical)
            abort("The QQ field must be a logical matrix or TRUE")
          whichUsed(private$aType) <- value
        }
    )
}


RuleSet <- new.env();

getRule <- function(name) RuleSet[[name]]
setRule <- function(name,value) {
  if (!is(value,"CombinationRule"))
    stop("Value must be a CombinationRule")
  env_poke(RuleSet,name,value)
}



CompensatoryRule <- nn_module(
    classname="CompensatoryRule",
    inherit = RuleASB,
    prescale <- torch_sum,
    summary <- torch_rootk,
    postscale <- torch_mul,
    forward = function() {
      if (!isTRUE(self$QQ)) {
        exec(self$postscale,
             genMMt(self$eTheta,self$bMat,self$prescale,self$summary),
             self$aMat)
      } else {
        exec(self$postscale,
             genMMt(self$eTheta,self$bMat,self$prescale,self$summary),
             self$aMat)
      }
    },
    active = list(
        QQ=function(value) {
          if (missing(value)) return (whichUsed(private$btype))
          if (!is.logical)
            abort("The QQ field must be a logical matrix or TRUE")
          whichUsed(private$bType) <- value
        }
    )
}


setRule("Compensatory",
  CombinationRule("Compensatory","ASBRule","*",sumrootk,"-",
                  PType("pos",c(K,J)),PType("real",c(K,1))))
setRule("Conjunctive",
  CombinationRule("Conjunctive","BSARule","-",min,"*",
                  PType("pos",c(K,1)),PType("real",c(K,J))))
setRule("Disjunctive",
  CombinationRule("Disjunctive","BSARule","-",max,"*",preparam="B",
                  PType("pos",c(K,1)),PType("real",c(K,J))))
setRule("NoisyAnd",
  CombinationRule("NoisyAnd","BASRule",">",prod,"*",preparam="AB",
                  PType("unit",c(K,J)),PType("real",c(K,J))))
setRule("NoisyAnd",
  CombinationRule("NoisyOr","BASRule",">",prodq,"*",
                  PType("unit",c(K,J)),PType("real",c(K,J))))

setRule("constB1",
  CombinationRule("constB1","constB",identity,mean,identity,
                  PType("real",c(0,0)),PType("real",c(1,1))))
setRule("DirichletRule",
  CombinationRule("Dirichlet","constB",identity,mean,identity,
                  PType("real",c(0,0)),PType("pvec",c(1,K))))
setRule("hyperDirichletRule",
  CombinationRule("hyperDirichlet","constB",identity,mean,identity,
                  PType("real",c(0,0)),PType("cpMat",c(S,K))))



