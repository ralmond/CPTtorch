

### Calculates the effective theta values for a skill variable
### with the argument number of levels
effectiveTheta10 <- function (nlevels) {
  torch_tensor(rev(qnorm((2*(1:nlevels)-1)/(2*nlevels),0,1)))
}

effectiveTheta <- function (nlevels,high2low=TRUE) {
  et <- qnorm((2*(1:nlevels)-1)/(2*nlevels))
  if (high2low) rev(et)
  else et
}

eTheta10 <- function (dims) {
  torch_cartisian_product(lapply(dims, \(d) effectiveTheta10(d)))
}

buildeTheta10 <- function(Tvallist) {
  if (length(Tvallist)==0L) torch_tensor(0.0)
  return(torch_cartesian_prod(lapply(Tvallist,as_torch_tensor)))
}

cartesian_prod <- function (list_o_vecs)
  rev(expand.grid(rev(list_o_vecs)))

as_Tvallist <- function (parents, parentprefix="P", stateprefix="S",high2low=TRUE) {
  pnames <- names(parents)
  if (length(parents) > 0L && length(pnames) == 0L) {
    pnames <- paste0(parentprefix,1L:length(parents))
  }
  if (is.numeric(parents)) {
    parents <- lapply(parents,\(p) paste0(stateprefix,1L:p))
  }
  parents <- map(parents, \(p,h2l) {
    if (is.character(p)) {
      pp <- effectiveTheta(length(p),h2l)
      names(pp) <- p
      pp
    } else {
      if (length(names(p)) == 0L) {
        names(p) <- paste0(stateprefix,1L:length(p))
      }
      p
    }
  }, parents, rep_len(high2low,length(parents))
  names(parents) <- pnames
  parents
}



############
## New OO version

CombinationRule <- nn_module(
    classname="CombinationRule",
#    inherit = nn_Module,
    aop = torch_mul,
    summary = torch_sum,
    bop = torch_add,
    aVec = NULL,
    bVec = NULL,
    setParents = function(parents) {
      self$pTheta <- buildeTheta10(parents)
      self$setDim(S=nrow(self$pTheta),J=ncol(self$pTheta))
    },
    setDim = function (S=1L,J=1L,K=1L) {
      if (!missing(S)) private$SJK$S <- S
      if (!missing(J)) private$SJK$S <- J
      if (!missing(K)) private$SJK$S <- K
      if (!is.null(private$atype)) {
        adim <- pTypeDim(private$atype)
        private$atype <- exec(setpTypeDim,private$atype,!!!private$SJK)
        if (!isTRUE(all.equal(adim,pTypeDim(private$atype))))
           self$aMat <- torch_tensor(defaultParameter(private$atype))
      }
      if (!is.null(private$btype)) {
        bdim <- pTypeDim(private$btype)
        private$btype <- exec(setpTypeDim,private$btype,!!!private$SJK)
        if (!isTRUE(all.equal(bdim,pTypeDim(private$btype))))
           self$bMat <- torch_tensor(defaultParameter(private$btype))
      }
      invisible(self)
    },
    initialize = function (parents, nstates, QQ=TRUE, ...) {
      private$SJK$K <- nstates
      self$setParent(parents)
      self$QQ <- QQ
    },
    forward = function() {
      if (!isTRUE(self$QQ)) {
        self$bop(
                 genMMtQ(self$pTheta,self$aMat,self$QQ,
                         self$aop,self$summary),
                 self$bMat$t_())
      } else {
        self$bop(
                 genMMt(self$pTheta,self$aMat,
                        self$aop,self$summary),
                 self$bMat$t_())
      }
    },
    private=list(
        atype=PType("real",c(K,J)),
        btype=PType("real",c(K,1L)),
        SJK=list(S=1L,J=1L,K=1L),
        cache=NULL
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
          self$aVec <- nn_parameter(pMat2tvec10(private$atype,
                                                as_torch_tensor(value)))
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
          self$bVec <- nn_parameter(pMat2tvec10(private$btype,
                                                as_torch_tensor(value)))
          invisible(self)
        },
        et = function() {
          if (is.null(private$cache))
            private$cache <- self$forward()
          private$cache
        },
        et_p = function(value) {
          if (missing(value))
            return(is.null(private$cache))
          if (isFALSE(value))
            private$cache <- NULL
          inivisble(self)
        }
    )
)

RuleASB <- CombinationRule

RuleBSA <- nn_module(
    classname="RuleBSA",
    inherit = CombinationRule,
    aop =torch_mul,
    summary =torch_max,
    bop = torch_sub,
    forward = function() {
      if (!isTRUE(self$QQ)) {
        self$aop(
                 genMMtQ(self$pTheta,self$bMat,self$QQ,
                         self$bop,self$summary),
                 self$aMat$t_())
      } else {
        self$aop(
                 genMMt(self$pTheta,self$bMat,
                        self$bop,self$summary),
                 self$aMat$t_())
      }
    },
    private=list(
        atype=PType("real",c(K,1L)),
        btype=PType("real",c(K,J))
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
)

RuleBAS <- nn_module(
    classname="RuleBAS",
    inherit = CombinationRule,
    aop = torch_mul,
    summary = torch_max,
    bop =  torch_sub,
    forward = function() {
      if (!isTRUE(self$QQ)) {
        tmp <- self$aop(self$bop(self$pTheta,
                                 self$bMat$reshape(1,dim(self$bMat))),
                        self$aMat$reshape(1,dim(self$aMat)))
        result <- torch_empty(private$SJK[c("S","K")])
        for (kk in 1L:private$SJK["K"])
          result[,kk] <- self$summary(tmp[,which(self$QQ[kk,]),kk],2)
        result
      } else {
        self$summary(
                 self$aop(
                          self$bop(self$pTheta,
                                        self$bMat$reshape(1,dim(self$bMat))),
                          self$aMat$reshape(1,dim(self$aMat))),
                 2)
      }
    },
    private=list(
        atype=PType("real",c(K,J)),
        btype=PType("real",c(K,J))
    ),
    active = list(
        QQ=function(value) {
          if (missing(value)) return (whichUsed(private$btype))
          if (!is.logical)
            abort("The QQ field must be a logical matrix or TRUE")
          whichUsed(private$btype) <- value
          whichUsed(private$atype) <- value
          invisible(self)
        }
    )
)




RuleConstB <- nn_module(
    classname="RuleConstB",
    inherit = CombinationRule,
    forward = function() {self$bMat},
    private=list(
        atype=NULL,
        btype=PType("const",c(K,J))
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
)

RuleConstA <- nn_module(
    classname="RuleConstA",
    inherit = CombinationRule,
    forward = function(input) {self$aMat},
    private=list(
        atype=PType("const",c(K,J)),
        btype=NULL
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
)


CompensatoryRule <- nn_module(
    classname="CompensatoryRule",
    inherit = RuleASB,
    aop = torch_mul,
    summary = torch_sumrootk,
    bop = torch_sub,
    private=list(
        atype=PType("pos",c(K,J)),
        btype=PType("real",c(K,1)),
        rootj=NA
    ),
    initialize=function(parents, nstates, QQ=true, ...) {
      super$initialize(parents,nstates,QQ,...)
      private$rootj <- 1/sqrt(private$SJK$J)
    },
    setParents = function(parents) {
      super$setParents(parents)
      private$rootj <- 1/sqrt(private$SJK$J)
    },
    forward = function() {
      if (!isTRUE(self$QQ)) {
        self$bop(
                 genMMtQ(self$pTheta,self$aMat,self$QQ,
                         self$aop,self$summary),
                 self$bMat$t_())
      } else {
        ## Using built-in matrix multiplication should be faster
        torch_addmm(self$bMat$neg_()$t_(),self$pTheta,self$aMat$t_(),alpha=private$rootj)
      }
    }
)


CompensatoryRule1 <- nn_module(
    classname="CompensatoryRule1",
    inherit = CompensatoryRule,
    private=list(
        atype=PType("pos",c(K,1L))
    )
)


ConjunctiveRule <- nn_module(
    classname="ConjunctiveRule",
    inherit = RuleBSA,
    bop = torch_sub,
    summary = torch_min,
    aop = torch_mul,
    private=list(
        atype=PType("pos",c(K,1)),
        btype=PType("real",c(K,J))
     )
)

DisjunctiveRule <- nn_module(
    classname="DisjunctiveRule",
    inherit = RuleBSA,
    bop = torch_sub,
    summary = torch_max,
    aop = torch_mul,
    private=list(
        atype=PType("pos",c(K,1)),
        btype=PType("real",c(K,J))
     )
)


NoisyAndRule <- nn_module(
    classname="NoisyAndRule",
    inherit = RuleBAS,
    bop = torch_gt,
    summary = torch_prod,
    aop = \(e1,e2) torch_pow(e2,e1),
    private=list(
        atype=PType("unit",c(K,J)),
        btype=PType("real",c(K,J))
     )
)

NoisyOrRule <- nn_module(
    classname="NoisyAndRule",
    inherit = RuleBAS,
    bop = torch_lte,
    summary = torch_prod_1,
    aop = \(e1,e2) torch_pow(e2,e1),
    private=list(
        atype=PType("unit",c(K,J)),
        btype=PType("real",c(K,J))
     )
)






CenterRule <- nn_module(
    classname="CenterRule",
    inherit = RuleConstB,
    prescale = identity,
    summary = torch_mean,
    postscale = identity,
    private=list(
        atype=NULL,
        btype=PType("real",c(S,K))
     )
)

DirichletRule <- nn_module(
    classname="DirichletRule",
    inherit = RuleConstB,
    prescale = identity,
    summary = torch_mean,
    postscale = identity,
    private=list(
        atype=NULL,
        btype=PType("real",c(S,K))
     )
)


RuleSet <- new.env();

getRule <- function(name) RuleSet[[name]]
setRule <- function(name,value) {
  if (!is(value,"CombinationRule"))
    stop("Value must be a CombinationRule")
  env_poke(RuleSet,name,value)
}
availableRules <- function() names(RuleSet)

setRule("Compensatory",CompensatoryRule)
setRule("Compensatory1",CompensatoryRule1)
setRule("Conjunctive",ConjunctiveRule)
setRule("Disjunctive",DisjunctiveRule)
setRule("NoisyAnd",NoisyAndRule)
setRule("NoisyOr",NoisyOrRule)
setRule("Center",CenterRule)
setRule("Dirichlet",DirichletRule)


