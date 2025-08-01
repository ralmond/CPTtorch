# !diagnostics suppress=self,private,super

### Calculates the effective theta values for a skill variable
### with the argument number of levels

effectiveTheta <- function (nlevels,high2low=FALSE) {
  et <- qnorm((2*(1:nlevels)-1)/(2*nlevels))
  if (high2low) rev(et)
  else et
}
effectiveTheta10 <- function (nlevels,high2low=FALSE) {
  torch_tensor(effectiveTheta(nlevels,high2low))
}


buildpTheta10 <- function(Tvallist) {
  if (length(Tvallist)==0L) return(torch_tensor(0.0))
  if (length(Tvallist)==1L) return(torch_reshape(torch_tensor(Tvallist[[1]]),c(-1,1)))
  return(torch_cartesian_prod(lapply(Tvallist,as_torch_tensor)))
}

cartesian_prod <- function (list_o_vecs)
  rev(expand.grid(rev(list_o_vecs)))

as_Tvallist <- function (parents, parentprefix="P", stateprefix="S",high2low=FALSE) {
  pnames <- names(parents)
  if (length(parents) > 0L && length(pnames) == 0L) {
    pnames <- paste0(parentprefix,1L:length(parents))
  }
  if (is.numeric(parents)) {
    parents <- lapply(parents,\(p) paste0(stateprefix,1L:p))
  }
  parents <- purrr::map2(parents, rep_len(high2low,length(parents)),
                         \(p,h2l) {
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
  })
  names(parents) <- pnames
  parents
}



############
## New OO version

CombinationRule <- torch::nn_module(
    classname="CombinationRule",
#    inherit = nn_Module,
    aop = "torch_mul",
    summary = "torch_sum",
    bop = "torch_add",
    aVec = NULL,
    bVec = NULL,
    setParents = function(parents) {
      self$pTheta <- buildpTheta10(parents)
      self$pNames <- lapply(parents,names)
      self$setDim(S=nrow(self$pTheta),J=ncol(self$pTheta))
    },
    setDim = function (S=1L,J=1L,K=1L) {
      if (!missing(S)) private$SJK$S <- S
      if (!missing(J)) private$SJK$J <- J
      if (!missing(K)) private$SJK$K <- K
      if (!is.null(private$atype)) {
        adim <- pTypeDim(private$atype)
        private$atype <- exec(setpTypeDim,private$atype,!!!private$SJK)
        if (!isTRUE(all.equal(adim,pTypeDim(private$atype))))
           self$aMat <- defaultParameter10(private$atype)
      }
      if (!is.null(private$btype)) {
        bdim <- pTypeDim(private$btype)
        private$btype <- exec(setpTypeDim,private$btype,!!!private$SJK)
        if (!isTRUE(all.equal(bdim,pTypeDim(private$btype))))
           self$bMat <- defaultParameter10(private$btype)
      }
      invisible(self)
    },
    initialize = function (parents, nstates, QQ=TRUE, high2low=FALSE,...) {
      private$SJK$K <- nstates
      self$setParents(parents)
      self$QQ <- QQ
      self$high2low <- high2low
      if (!is.null(self$aType)) {
        self$aMat <- defaultParameter10(private$atype)
      }
      if (!is.null(self$bType)) {
        self$bMat <- defaultParameter10(private$btype)
      }
    },
    forward = function() {
      amat <- self$aMat
      qmat <- self$QQ
      bmat <- self$bMat
      if (self$high2low) {
        amat <- torch_flipud(amat)
        bmat <- torch_flipud(bmat)
        if (!isTRUE(qmat))
          qmat <- torch_flipud(qmat)
      }
      if (!isTRUE(qmat)) {
        exec(self$bop,
             genMMtQ(self$pTheta,amat,qmat,
                     self$aop,self$summary),
             bmat$t_())
      } else {
        exec(self$bop,
             genMMt(self$pTheta,amat,
                    self$aop,self$summary),
             bmat$t_())
      }
    },
    getETframe = function () {
      data.frame(cartesian_prod(self$pNames),
                 et=as_array(self$et))
    },
    private=list(
        atype=PType("real",c(K,J)),
        btype=PType("real",c(K,1L)),
        SJK=list(S=1L,J=1L,K=1L),
        high_low = FALSE,
        pTheta=NULL,
        cache=NULL
    ),
    active = list(
        high2low = function (value) {
          if (missing(value)) return (private$high_low)
          if (!is.logical(value))
            abort("The high2low field must have a logical value.")
          private$atype$high2low <- value
          private$btype$high2low <- value
          private$high_low <- value
          private$cache <- NULL
        },
        aType=function(value) {
          if (missing(value)) return (private$atype)
          if (!is(value,"PType"))
            abort("The aType field must be a PType object.")
          private$atype <- value
          if (!is.null(private$SJK)) {
            private$atype <- exec(setpTypeDim,private$atype,!!!private$SJK)
            self$aMat <- defaultParameter10(private$atype)
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
            self$bMat <- defaultParameter10(private$btype)
          invisible(self)
          }
        },
        QQ=function(value) {
          if (missing(value)) return (whichUsed(private$atype))
          if (!is.logical(value))
            abort("The QQ field must be a logical matrix or TRUE")
          whichUsed(private$atype) <- value
          invisible(self)
        },
        aMat=function(value) {
          if (missing(value)) {
            if (is.null(self$aVec)) return (NULL)
            amat <- tvec2natpar(private$atype,self$aVec)
            return (amat)
          }
          pcheck <- checkParam(private$atype,value)
          if (!isTRUE(pcheck))
            abort(paste("Illegal A parameter value,",pcheck,"."))
          private$cache <- NULL
          amat <- as_torch_tensor(value)
          self$aVec <- nn_parameter(natpar2tvec(private$atype,amat))
          invisible(self)
        },
        bMat=function(value) {
          if (missing(value)) {
            if (is.null(self$bVec)) return (NULL)
            bmat <- tvec2natpar(private$btype,self$bVec)
            return (bmat)
          }
          pcheck <- checkParam(private$btype,value)
          if (!isTRUE(pcheck))
            abort(paste("Illegal A parameter value,",pcheck,"."))
          private$cache <- NULL
          bmat <- as_torch_tensor(value)
           self$bVec <- nn_parameter(natpar2tvec(private$btype,bmat))
          invisible(self)
        },
        et = function() {
          if (is.null(private$cache))
            private$cache <- self$forward()
          private$cache
        },
        et_p = function(value) {
          if (missing(value))
            return(!is.null(private$cache))
          if (isFALSE(value))
            private$cache <- NULL
          invisible(self)
        }
    )
)

RuleASB <- CombinationRule

RuleBSA <- torch::nn_module(
    classname="RuleBSA",
    inherit = CombinationRule,
    aop ="torch_mul",
    summary ="torch_amax",
    bop = "torch_sub",
    forward = function() {
      amat <- self$aMat
      qmat <- self$QQ
      bmat <- self$bMat
      if (self$high2low) {
        amat <- torch_flipud(amat)
        bmat <- torch_flipud(bmat)
        qq <- torch_flipud(qmat)
      }
      if (!isTRUE(qmat)) {
        exec(self$aop,
             genMMtQ(self$pTheta,bmat,qmat,
                     self$bop,self$summary),
             amat$t_())
      } else {
        exec(self$aop,
             genMMt(self$pTheta,bmat,
                    self$bop,self$summary),
             amat$t_())
      }
    },
    private=list(
        atype=PType("real",c(K,1L)),
        btype=PType("real",c(K,J))
    ),
    active = list(
        QQ=function(value) {
          if (missing(value)) return (whichUsed(private$btype))
          if (!is.logical(value))
            abort("The QQ field must be a logical matrix or TRUE")
          whichUsed(private$btype) <- value
          invisible(self)
        }
    )
)


RuleBAS <- torch::nn_module(
    classname="RuleBAS",
    inherit = CombinationRule,
    aop = "torch_mul",
    summary = "torch_amax",
    bop =  "torch_sub",
    forward = function() {
      amat <- self$aMat
      qmat <- self$QQ
      bmat <- self$bMat
      if (self$high2low) {
        amat <- torch_flipud(amat)
        bmat <- torch_flipud(bmat)
        qq <- torch_flipud(qmat)
      }
      tmp <- exec(self$aop,
                  exec(self$bop,self$pTheta$reshape(c(dim(self$pTheta),1)),
                       bmat$reshape(c(1,dim(bmat)))),
                  amat$reshape(c(1,dim(amat))))
      if (isTRUE(qmat)) {
        return(exec(self$summary,tmp,2))
      } else {
        result <- torch_empty(private$SJK[c("S","K")])
        for (kk in 1L:nrow(result))
          result[,kk] <- exec(self$summary,tmp[,which(qmat[kk,]),kk],2)
        result
      }
    },
    private=list(
        atype=PType("real",c(K,J)),
        btype=PType("real",c(K,J))
    ),
    active = list(
        QQ=function(value) {
          if (missing(value)) return (whichUsed(private$btype))
          if (!is.logical(value))
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
    forward = function() {
      if (self$high2low) return (torch_flipud(self$bMat))
      self$bMat
    },
    private=list(
        atype=NULL,
        btype=PType("const",c(K,J))
    ),
    active = list(
        aType=function(value) {
          if (missing(value)) return (NULL)
          warning("A Type is ignored in ConstB Rules.")
          invisible(self)
        },
        aMat=function(value) {
          if (missing(value)) {
            if (is.null(self$aVec)) return (NULL)
            return (pVec2pMat10(private$atype,self$aVec))
          }
          warning("A parameter is ignored in ConstB Rules.")
          invisible(self)
        },
        QQ=function(value) {
          if (missing(value)) return (whichUsed(private$btype))
          if (!is.logical(value))
            abort("The QQ field must be a logical matrix or TRUE")
          whichUsed(private$btype) <- value
          invisible(self)
        }
    )
)

RuleConstA <- nn_module(
    classname="RuleConstA",
    inherit = CombinationRule,
    forward = function(input) {
      if (self$high2low) return (torch_flipud(self$aMat))
      self$aMat
    },
    private=list(
        atype=PType("const",c(K,J)),
        btype=NULL
    ),
    active = list(
        bType=function(value) {
          if (missing(value)) return (NULL)
          warning("B Type is ignored in ConstA Rules.")
          invisible(self)
        },
        bMat=function(value) {
          if (missing(value)) {
            if (is.null(self$bVec)) return (NULL)
            return (pVec2pMat10(private$btype,self$aVec))
          }
          warning("B parameter is ignored in ConstB Rules.")
          invisible(self)
        },
        QQ=function(value) {
          if (missing(value)) return (whichUsed(private$atype))
          if (!is.logical(value))
            abort("The QQ field must be a logical matrix or TRUE")
          whichUsed(private$atype) <- value
          invisible(self)
        }
    )
)


CompensatoryRule <- torch::nn_module(
    classname="CompensatoryRule",
    inherit = RuleASB,
    aop = "torch_mul",
    summary = "torch_sumrootk",
    bop = "torch_sub",
    private=list(
        atype=PType("pos",c(K,J)),
        btype=PType("real",c(K,1)),
        rootj=NA
    ),
    # initialize=function(parents, nstates, QQ=true, ...) {
    #   super$initialize(parents,nstates,QQ,...)
    #   private$rootj <- 1/sqrt(private$SJK$J)
    # },
    setParents = function(parents) {
      super$setParents(parents)
      private$rootj <- 1/sqrt(private$SJK$J)
    },
    forward = function() {
      amat <- self$aMat
      qmat <- self$QQ
      bmat <- self$bMat
      if (self$high2low) {
        amat <- torch_flipud(amat)
        bmat <- torch_flipud(bmat)
        qq <- torch_flipud(qmat)
      }
      if (!isTRUE(qmat)) {
        exec(self$bop,
             genMMtQ(self$pTheta,amat,qmat,
                     self$aop,self$summary),
             bmat$t_())
      } else {
        ## Using built-in matrix multiplication should be faster
        ##torch_addmm(bmat$neg()$t_(),self$pTheta,amat$t(),alpha=private$rootj)
        torch_add(bmat$neg()$t_(),
                  torch_matmul(self$pTheta,amat$t())$mul_(private$rootj))
      }
    }
)


CompensatoryGRRule <- torch::nn_module(
    classname="CompensatoryGRRule",
    inherit = CompensatoryRule,
    private=list(
        atype=PType("pos",c(1L,J)),
        btype=PType("incrK",c(K,1L))
    )
)


ConjunctiveRule <- torch::nn_module(
    classname="ConjunctiveRule",
    inherit = RuleBSA,
    bop = "torch_sub",
    summary = "torch_amin",
    aop = "torch_mul",
    private=list(
        atype=PType("pos",c(K,1)),
        btype=PType("real",c(K,J))
     )
)

DisjunctiveRule <- torch::nn_module(
    classname="DisjunctiveRule",
    inherit = RuleBSA,
    bop = "torch_sub",
    summary = "torch_amax",
    aop = "torch_mul",
    private=list(
        atype=PType("pos",c(K,1)),
        btype=PType("real",c(K,J))
     )
)


NoisyAndRule <- torch::nn_module(
    classname="NoisyAndRule",
    inherit = RuleBAS,
    bop = "torch_gt",
    summary = "torch_prod",
    aop = "torch_mul",
    private=list(
        atype=PType("unit",c(K,J)),
        btype=PType("real",c(K,J))
     )
)

NoisyOrRule <- torch::nn_module(
    classname="NoisyAndRule",
    inherit = RuleBAS,
    bop = "torch_le",
    summary = "torch_prod_1",
    aop = "torch_mul",
    private=list(
        atype=PType("unit",c(K,J)),
        btype=PType("real",c(K,J))
     )
)






CenterRule <- torch::nn_module(
    classname="CenterRule",
    inherit = RuleConstB,
    prescale = "identity",
    summary = "torch_mean",
    postscale = "identity",
    private=list(
        atype=NULL,
        btype=PType("real",c(S,K))
     )
)

DirichletRule <- torch::nn_module(
    classname="DirichletRule",
    inherit = RuleConstB,
    prescale = "identity",
    summary = "torch_mean",
    postscale = "identity",
    private=list(
        atype=NULL,
        btype=PType("real",c(S,K))
     )
)



RuleSet <- new.env()
getRule <- function(name) {
  if (is(name,"CombinationRule")) return(name)
  RuleSet[[name]]
}
setRule <- function(name,value) {
  if (!is(value,"CombinationRule"))
     stop("Value must be a CombinationRule")
   RuleSet[[name]] <- value
}
availableRules <- function() {
  names(RuleSet)
}

setRule("Compensatory",CompensatoryRule)
setRule("CompensatoryGR",CompensatoryGRRule)
setRule("Conjunctive",ConjunctiveRule)
setRule("Disjunctive",DisjunctiveRule)
setRule("NoisyAnd",NoisyAndRule)
setRule("NoisyOr",NoisyOrRule)
setRule("Center",CenterRule)
setRule("Dirichlet",DirichletRule)



