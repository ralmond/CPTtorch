

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
      if (!is.null(private$atype))
        private$atype <- exec(setpTypeDim,private$atype,!!!private$SJK)
      if (!is.null(private$btype))
        private$btype <- exec(setpTypeDim,private$btype,!!!private$SJK)

      self$QQ <- QQ
      if (!is.null(private$atype))
        self$aMat <- torch_tensor(defaultParameter(private$atype))
      if (!is.null(private$btype))
        self$bMat <- torch_tensor(defaultParameter(private$btype))
    },
    forward = function(input) {
      if (!isTRUE(self$QQ)) {
        self$postscale(
                 genMMtQ(self$eTheta,self$aMat,self$QQ,
                         self$prescale,self$summary),
                 self$bMat)
      } else {
        self$postscale(
                 genMMt(self$eTheta,self$aMat,
                        self$prescale,self$summary),
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
            private$atype <- exec(setpTypeDim,private$atype,!!!private$SJK)
            self$aMat <- torch_tensor(defaultParam(private$atype))
          }
          invisible(self)
        },
        bType=function(value) {
          if (missing(value)) return (private$btype)
          if (!is(value,"PType"))
            abort("The bType field must be a PType object.")
          private$btype <- value
          if (!is.null(private$SJK)) {
            private$btype <- exec(setpTypeDim,private$btype,!!!private$SJK)
            self$bMat <- torch_tensor(defaultParam(private$btype))
          invisible(self)
          }
        },
        QQ=function(value) {
          if (missing(value)) return (whichUsed(private$atype))
          if (!is.logical)
            abort("The QQ field must be a logical matrix or TRUE")
          whichUsed(private$atype) <- value
          invisible(self)
        },
        aMat=function(value) {
          if (missing(value)) {
            if (is.null(self$aVec)) return (NULL)
            return (tvect2pMat10(private$atype,self$aVec))
          }
          pcheck <- checkParam(private$atype,value)
          if (!isTRUE(pcheck))
            abort("Illegal A parameter value,",pcheck,".")
          private$cache <- NULL
          self$aVec <- nn_parameter(pMat2tvec10(private$atype,value))
          invisible(self)
        },
        bMat=function(value) {
          if (missing(value)) {
            if (is.null(self$bVec)) return (NULL)
            return (tvect2pMat10(private$btype,self$bVec))
          }
          pcheck <- checkParam(private$btype,value)
          if (!isTRUE(pcheck))
            abort("Illegal A parameter value,",pcheck,".")
          private$cache <- NULL
          self$bVec <- nn_parameter(pMat2tvec10(private$btype,value))
          invisible(self)
        },
        et = function() {
          if (is.null(private$cache))
            private$cache <- self$forward()
          private$cache
        },
        et_p = function(value) {
          if (missing(value)) return(is.null(private$cache))
          if (isFALSE(value)) private$cache <- NULL
          inivisble(self)
    )
}


RuleBSA <- nn_module(
    classname="RuleBSA",
    inherit = RuleASB,
    prescale <- torch_sum,
    summary <- torch_max,
    postscale <- torch_mul,
    forward = function(input) {
      if (!isTRUE(self$QQ)) {
        self$postscale(
                 genMMtQ(self$eTheta,self$bMat,self$QQ,
                         self$prescale,self$summary),
                 self$aMat)
      } else {
        self$postscale(
                 genMMt(self$eTheta,self$bMat,
                        self$prescale,self$summary),
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
          whichUsed(private$btype) <- value
          invisible(self)
        }
    )
}

RuleBAS <- nn_module(
    classname="RuleBAS",
    inherit = RuleASB,
    prescale <- torch_sum,
    summary <- torch_max,
    postscale <- torch_mul,
    forward = function(input) {
      if (!isTRUE(self$QQ)) {
        tmp <- self$postscale(
                        self$prescale(self$eTheta,
                                      self$bMat$reshape(1,dim(self$bMat))),
                        self$aMat$reshape(1,dim(self$aMat))),
        result <- torch_empty(private$SJK[c("S","K")])
        for (kk in 1L:private$SJK["K"])
          result[,kk] <- self$summary(tmp[,which(self$QQ[kk,]),kk],2)
        result
      } else {
        self$summary(
                 self$postscale(
                          self$prescale(self$eTheta,
                                        self$bMat$reshape(1,dim(self$bMat))),
                          self$aMat$reshape(1,dim(self$aMat))),
                 2)
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
          whichUsed(private$btype) <- value
          invisible(self)
        }
    )
}




RuleConstB <- nn_module(
    classname="RuleConstB"
    inherit = RuleASB,
    forward = function(input) {self$bMat},
    private=list(
        atype=NULL,
        btype=PType("const",c(K,J)),
        ),
    active = list(
        aType=function(value) {
          if (missing(value)) return (private$atype)
          warning("A Type is ignored in ConstB Rules.")
          invisible(self)
        },
        aMat=function(value) {
          if (missing(value)) {
            if (is.null(self$aVec)) return (NULL)
            return (tvect2pMat10(private$atype,self$aVec))
          }
          warning("A parameter is ignored in ConstB Rules.")
          invisible(self)
        },
        QQ=function(value) {
          if (missing(value)) return (whichUsed(private$btype))
          if (!is.logical)
            abort("The QQ field must be a logical matrix or TRUE")
          whichUsed(private$btype) <- value
          invisible(self)
        }
    )
}

RuleConstA <- nn_module(
    classname="RuleConstA",
    inherit = RuleASB,
    forward = function(input) {self$aMat},
    private=list(
        atype=PType("const",c(K,J)),
        btype=NULL,
        ),
    active = list(
        bType=function(value) {
          if (missing(value)) return (private$btype)
          warning("B Type is ignored in ConstA Rules.")
          invisible(self)
        },
        bMat=function(value) {
          if (missing(value)) {
            if (is.null(self$bVec)) return (NULL)
            return (tvect2pMat10(private$btype,self$aVec))
          }
          warning("B parameter is ignored in ConstB Rules.")
          invisible(self)
        },
        QQ=function(value) {
          if (missing(value)) return (whichUsed(private$atype))
          if (!is.logical)
            abort("The QQ field must be a logical matrix or TRUE")
          whichUsed(private$atype) <- value
          invisible(self)
        }
    )
}


CompensatoryRule <- nn_module(
    classname="CompensatoryRule",
    inherit = RuleASB,
    prescale <- torch_sum,
    summary <- torch_rootk,
    postscale <- torch_mul,
    private=list(
        atype=PType("pos",c(K,J)),
        btype=PType("real",c(K,1)),
        rootk=NA
    ),
    initialize=function(parents, nstates, QQ=true, ...) {
      super$initialize(parents,nstates,QQ,...)
      private$rootk <- torch_tensor(1/sqrt(private$SJK$K))
    },
    forward = function(input) {
      if (!isTRUE(self$QQ)) {
        self$postscale(
                 genMMtQ(self$eTheta,self$aMat,self$QQ,
                         self$prescale,self$summary),
                 self$bMat)
      } else {
        ## Using built-in matrix multiplication should be faster
        torch_matmul(self$eTheta, self$aMat$t_())$
          mul_(private$rootk)$
          add_(self$bMat)
      }
    }
    )
}

ConjunctiveRule <- nn_module(
    classname="ConjunctiveRule",
    inherit = RuleBSA,
    prescale <- torch_sub,
    summary <- torch_min,
    postscale <- torch_mul,
    private=list(
        atype=PType("pos",c(K,1)),
        btype=PType("real",c(K,J))
     )
}

DisjunctiveRule <- nn_module(
    classname="DisjunctiveRule",
    inherit = RuleBSA,
    prescale <- torch_sub,
    summary <- torch_max,
    postscale <- torch_mul,
    private=list(
        atype=PType("pos",c(K,1)),
        btype=PType("real",c(K,J))
     )
}


NoisyAndRule <- nn_module(
    classname="NoisyAndRule",
    inherit = RuleBAS,
    prescale <- torch_gt,
    summary <- torch_prod,
    postscale <- torch_mul,
    private=list(
        atype=PType("pos",c(K,J)),
        btype=PType("real",c(K,J))
     )
}

NoisyOrRule <- nn_module(
    classname="NoisyAndRule",
    inherit = RuleBAS,
    prescale <- torch_gt,
    summary <- torch_prodq,
    postscale <- torch_mul,
    private=list(
        atype=PType("pos",c(K,J)),
        btype=PType("real",c(K,J))
     )
}






CenterRule <- nn_module(
    classname="CenterRule",
    inherit = RuleConstB,
    prescale <- identity,
    summary <- torch_mean,
    postscale <- identity,
    private=list(
        atype=NULL,
        btype=PType("real",c(S,K))
     )
}

DirichletRule <- nn_module(
    classname="DirichletRule",
    inherit = RuleConstB,
    prescale <- identity,
    summary <- torch_mean,
    postscale <- identity,
    private=list(
        atype=NULL,
        btype=PType("real",c(S,K))
     )
}


RuleSet <- new.env();

getRule <- function(name) RuleSet[[name]]
setRule <- function(name,value) {
  if (!is(value,"CombinationRule"))
    stop("Value must be a CombinationRule")
  env_poke(RuleSet,name,value)
}
availableRules <- function() names(RuleSet)

setRule("Compensatory",CompensatoryRule)
setRule("Conjunctive",ConjunctiveRule)
setRule("Disjunctive",DisjunctiveRule)
setRule("NoisyAnd",NoisyAndRule)
setRule("NoisyOr",NoisyOrRule)
setRule("Center",CenterRule)
setRule("Dirichlet",DirichletRule)


