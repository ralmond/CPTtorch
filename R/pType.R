PType <- function(pType,dim=c(K,J), zero=NULL, used=TRUE, high2low=FALSE) {
  res <- list(dimexpr=substitute(dim),dim=NULL,zero=zero, used=used,
              high2low=high2low)
  class(res) <- c(pType,"PType")
  res
}

PTypeList <- c("real","pos","unit","pVec","cpMat","const","incrK")
availablePTypes <- function() {PTypeList}
isPType <- function (obj) UseMethod("isPType")
isPType.default <- function (obj)
  any(class(obj) %in% PTypeList)


pTypeDim <- function(pType) pType$dim
setpTypeDim <- function(pType,J=1L,S=2L^J,K=1L) {
  pType$dim <- eval(rlang::inject(pType$dimexpr))
  pType
}

whichUsed <- function(pType) {
  return(pType$used)
}
"whichUsed<-" <- function(pType,value) {
  pType$used <- value
  pType
}

pMat2pVec <- function (pType,pMat) {
  if (isTRUE(whichUsed(pType)) || is.null(pTypeDim(pType))) as.vector(pMat)
  else as.vector(pMat[whichUsed(pType)])
}
pMat2pVec10 <- function(pType,pMat) {
  if (isTRUE(whichUsed(pType))) pMat$reshape(-1)
  else pMat[whichUsed(pType)]
}
pMat2rowlist <- function (pType,pMat) {
  if (is.null(pTypeDim(pType))) return(list(pMat))
  if (isTRUE(whichUsed(pType))) return (lapply(1L:nrow(pMat),function(r) pMat[r,]))
  else lapply(1L:nrow(pMat), function(r) (pMat[r,whichUsed(pType)[r,]]))
}
rowlist2pMat <- function(pType,rlist) {
  if (is.null(pTypeDim(pType))) return(list2vec(rlist))
  if (isTRUE(whichUsed(pType))) return(do.call("rbind",rlist))
  else {
    result <- array(NA,pTypeDim(pType))
    for (rr in 1:nrow(result)) {
      result[rr,whichUsed(pType)[rr,]] <- rlist[[rr]]
    }
    result
  }
}
pMat2collist <- function (pType,pMat) {
  if (is.null(pTypeDim(pType))) return(list(pMat))
  if (isTRUE(whichUsed(pType))) return (lapply(1L:ncol(pMat),function(cc) pMat[,cc]))
  else lapply(1L:ncol(pMat), function(cc) (pMat[whichUsed(pType)[,cc],cc]))
}
list2vec <- function (plist) do.call("c",plist)

pVec2pMat <- function (pType,pVec) {
  if (is.null(pTypeDim(pType))) return(pVec)
  if (isTRUE(whichUsed(pType))) {
    dim(pVec) <- pTypeDim(pType)
    return(pVec)
  }
  pMat <- array(NA,pTypeDim(pType))
  pMat[whichUsed(pType)] <- pVec
  pMat
}

pVec2pMat10 <- function (pType,pVec) {
  if (is.null(pTypeDim(pType))) return(pVec)
  if (isTRUE(whichUsed(pType))) {
    return(pVec$reshape(pTypeDim(pType)))
  }
  pMat <- torch_empty(pTypeDim(pType))
  pMat[whichUsed(pType)] <- pVec
  pMat
}
collist2pMat10 <- function (pType,clist) {
  if (is.null(pTypeDim(pType))) return(list2vec(clist))
  usd <- whichUsed(pType)
  if (isTRUE(usd)) usd <- array(TRUE,pTypeDim(pType))
  pMat <- torch_empty(pTypeDim(pType))
  for (cc in 1:ncol(pMat))
    pMat[usd[,cc],cc] <- clist[[cc]]
  pMat
}
vec2rowlist <- function(pType,pVec) {
  dims <- pTypeDim(pType)
  if (is.null(dims)) return(pVec)
  used <- whichUsed(pType)
  if (isTRUE(whichUsed(pType)))
    rowlens <- rep(dims[2],dims[1])
  else
    rowlens <- rowSums(used)
  pos <- c(0,cumsum(rowlens))
  lapply(1L:dims[1], function (r) pVec[(pos[r]+1L):(pos[r+1L])])
}
vec2collist <- function(pType,pVec) {
  dims <- pTypeDim(pType)
  if (is.null(dims)) return(pVec)
  used <- whichUsed(pType)
  if (isTRUE(whichUsed(pType)))
    collens <- rep(dims[1],dims[2])
  else
    collens <- colSums(used)
  pos <- c(0,cumsum(collens))
  lapply(1L:dims[2], function (r) pVec[(pos[r]+1L):(pos[r+1L])])
}


checkParam <- function (pType,par) {UseMethod("checkParam")}
natpar2Rvec <- function(pType,natpar) {UseMethod("natpar2Rvec")}
Rvec2natpar <- function(pType,Rvec) {UseMethod("Rvec2natpar")}
natpar2tvec <- function(pType,natpar) {UseMethod("natpar2tvec")}
tvec2natpar <- function(pType,Rvec) {UseMethod("tvec2natpar")}


getZero <- function(pType) {UseMethod("getZero")}
getZero.character <- function(pType) {
    do.call(getS3method("getZero",pType),list())
}
defaultParameter <- function(pType) {UseMethod("defaultParameter")}
defaultParameter10 <- function(pType) {
  torch_tensor(defaultParameter(pType),dtype=torch_float())
}
defaultParameter.PType <- function(pType) {
  zero <- pType$zero
  if (is.null(zero)) zero <- getZero(pType)
  if (is.numeric(pTypeDim(pType))) array(zero,pTypeDim(pType))
  else zero
}


pastedims <- function (dms) {
  paste("(",paste(dms,collapse=","),")")
}

checkParam.default <- function(pType,par) {TRUE}
checkParam.PType <- function(pType,par) {
  ptd <- pTypeDim(pType)
  if (is.null(ptd))
    warning("Dimension not yet set, not checked.")
  else {
    if (length(ptd) != length(dim(par)) ||
        any(mapply(\(d1,d2) {d1!=d2 && d2!=1L},ptd,dim(par))))
      return(paste("Dimension mis-match, expected",pastedims(ptd),
                    "got", pastedims(dim(par)),"."))
  }
  NextMethod()
}

as.array.numeric <- function (x) {x}


checkParam.real <- function(pType,par) {
  par <- as.array(par)
  if (any(is.na(pMat2pVec(pType,par))))
    return("Unexpected NAs in parameter.")
  NextMethod()
}
natpar2Rvec.real <- function(pType,natpar) {identity(pMat2pVec(pType,natpar))}
Rvec2natpar.real <- function(pType,Rvec) {pVec2pMat(pType,identity(Rvec))}
natpar2tvec.real <- function(pType,natpar) {pMat2pVec10(pType,natpar)}
tvec2natpar.real <- function(pType,Rvec) {pVec2pMat10(pType,Rvec)}
getZero.real <- function(pType) {0}

checkParam.pos <- function(pType,par) {
  par <- as.array(par)
  pVec <-pMat2pVec(pType,par)
  if (any(is.na(pVec)))
    return("Unexpected NAs in parameter.")
  if (any(pVec<=0))
    return("Unexpected negative or zero values.")
  NextMethod()
}
natpar2Rvec.pos <- function(pType,natpar) {log(pMat2pVec(pType,natpar))}
Rvec2natpar.pos <- function(pType,Rvec) {pVec2pMat(pType,exp(Rvec))}
natpar2tvec.pos <- function(pType,natpar) {pMat2pVec10(pType,natpar)$log_()}
tvec2natpar.pos <- function(pType,Rvec) {pVec2pMat10(pType,torch_exp(Rvec))}
getZero.pos <- function(pType) {1}

checkParam.unit <- function(pType,par) {
  pVec <-pMat2pVec(pType,as.array(par))
  if (any(is.na(pVec)))
    return("Unexpected NAs in parameter.")
  if (any(pVec<0)||any(pVec>1))
    return("Values outside the range [0,1]")
  NextMethod()
}
natpar2Rvec.unit <- function(pType,natpar) {logit(pMat2pVec(pType,natpar))}
Rvec2natpar.unit <- function(pType,Rvec) {pVec2pMat(pType,invlogit(Rvec))}
natpar2tvec.unit <- function(pType,natpar) {pMat2pVec10(pType,natpar)$logit_()}
tvec2natpar.unit <- function(pType,Rvec) {pVec2pMat10(pType,torch_sigmoid(Rvec))}
getZero.unit <- function(pType) {.5}

checkParam.pVec <- function(pType,par) {
  pVec <-pMat2pVec(pType,as.array(par))
  if (any(is.na(pVec)))
    return("Unexpected NAs in parameter.")
  if (any(pVec<0)) return("Negative probabilities in parameter.")
  if (abs(sum(pVec)-1)>.0001)
    return("Values do not sum to 1.")
  NextMethod()
}
natpar2Rvec.pVec <- function(pType,natpar) {log(pMat2pVec(pType,natpar))}
Rvec2natpar.pVec <- function(pType,Rvec) {pVec2pMat(pType,softmax(Rvec))}
natpar2tvec.pVec <- function(pType,natpar) {pMat2pVec10(pType,natpar)$log_()}
tvec2natpar.pVec <- function(pType,Rvec) {pVec2pMat10(pType,nnf_softmax(Rvec,1))}
getZero.pVec <- function(pType) {.5}

checkParam.cpMat <- function(pType,par) {
  pVeclist <-pMat2rowlist(pType,as.array(par))
  for (pVec in pVeclist) {
    if (any(is.na(pVec)))
      return("Unexpected NAs in parameter.")
    if (any(pVec<0)) return("Negative probabilities in parameter.")
    if (abs(sum(pVec)-1)>.0001)
      return("Row walues do not sum to 1.")
  }
  NextMethod()
}
natpar2Rvec.cpMat <- function(pType,natpar) {
  log(list2vec(pMat2rowlist(pType,natpar)))
}
Rvec2natpar.cpMat <- function(pType,Rvec) {
  rowlist2pMat(pType,lapply(vec2rowlist(pType,Rvec),softmax))
}
natpar2tvec.cpMat <- function(pType,natpar) {
  pMat2pVec10(pType,natpar)$log_()
}
tvec2natpar.cpMat <- function(pType,Rvec) {
  pVec2pMat10(pType,torch_cat(lapply(vec2rowlist(pType,Rvec),
                                     \(r) nnf_softmax(r,1))))
}
getZero.cpMat <- function(pType) {1/pTypeDim(pType)[2]}

natpar2Rvec.const <- function(pType,natpar) {numeric()}
Rvec2natpar.const <- function(pType,Rvec) {numeric()}
natpar2tvec.const <- function(pType,natpar) {NULL}
tvec2natpar.const <- function(pType,Rvec) {NULL}
getZero.const <- function(pType) {NA}

checkParam.incrK <- function(pType,par) {
  pVeclist <-pMat2collist(pType,as.array(par))
  for (pVec in pVeclist) {
    if (any(is.na(pVec)))
      return("Unexpected NAs in parameter.")
    if (pType$high2low) {
      if (any(diff(pVec) >=0))
        return("Columns are not decreasing.")
    } else {
      if (any(diff(pVec) <=0))
        return("Columns are not increasing.")
    }
  }
  NextMethod()
}
natpar2Rvec.incrK <- function(pType,natpar) {
  difffun <- ldiff
  if (pType$high2low) difffun <- \(v) ldiff(rev(v))
  list2vec(lapply(pMat2collist(pType,natpar),difffun))
}
Rvec2natpar.incrK <- function(pType,Rvec) {
  sumfun <- ecusum
  if (pType$high2low) sumfun <- \(v) rev(ecusum(v))
  pVec2pMat(pType,list2vec(lapply(vec2collist(pType,Rvec),sumfun)))
}

natpar2tvec.incrK <- function(pType,natpar) {
  if (pType$high2low) natpar <- torch_flipud(natpar)
  torch_cat(lapply(pMat2collist(pType,natpar),torch_ldiff))
}
tvec2natpar.incrK <- function(pType,Rvec) {
  natp <- collist2pMat10(pType,lapply(vec2collist(pType,Rvec),
                                             torch_ecusum))
  if (pType$high2low) natp <- torch_flipud(natp)
  natp
}
getZero.incrK <- function(pType) {0}


defaultParameter.incrK <- function(pType) {
  zero <- pType$zero
  if (is.null(zero)) zero <- getZero(pType)
  if (is.numeric(pTypeDim(pType))) {
    zero <- array(zero,pTypeDim(pType))
    zero <- sweep(zero,1,0:(nrow(zero)-1),"+")
  }
  zero
}









