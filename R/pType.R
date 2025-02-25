PType <- function(pType,dim=c(K,J), zero=NULL, used=TRUE) {
  res <- list(dimexpr=substitute(dim),dim=NULL,zero=zero, used=used)
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
  PType
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


checkParam <- function (pType,par) UseMethod("checkParam")
natpar2Rvec <- function(pType,natpar) UseMethod("natpar2Rvec")
Rvec2natpar <- function(pType,Rvec) UseMethod("Rvec2natpar")
natpar2tvec <- function(pType,natpar) UseMethod("natpar2tvec")
tvec2natpar <- function(pType,Rvec) UseMethod("tvec2natpar")


getZero <- function(pType) UseMethod("getZero")
getZero.character <- function(pType)
    do.call(getS3method("getZero",pType),list())
defaultParameter <- function(pType) UseMethod("DefaultParameter")
defaultParameter10 <- function(pType) UseMethod("DefaultParameter10")

defaultParameter.PType <- function(pType) {
  zero <- pType$zero
  if (is.null(zero)) zero <- getZero(pType)
  if (is.numeric(pTypeDim(pType))) array(zero,pTypeDim(pType))
  else zero
}



checkParam.default <- function(pType,par) {TRUE}
checkParam.PType <- function(pType,par) {
  if (!all.equal(pTypeDim(pType),dim(par)))
    return("Dimension mis-match, or dimensions not set.")
  NextMethod()
}

as.array.numeric <- function (x) x


checkParam.real <- function(pType,par) {
  par <- as.array(par)
  if (any(is.na(pMat2pVec(pType,par))))
    return("Unexpected NAs in parameter.")
  NextMethod()
}
natpar2Rvec.real <- function(pType,natpar) identity(pMat2pVec(pType,natpar))
Rvec2natpar.real <- function(pType,Rvec) pVec2pMat(pType,identity(Rvec))
natpar2tvec.real <- function(pType,natpar) pMat2pVec10(pType,natpar)
tvec2natpar.real <- function(pType,Rvec) pVec2pMat10(pType,Rvec)
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
natpar2Rvec.pos <- function(pType,natpar) log(pMat2pVec(pType,natpar))
Rvec2natpar.pos <- function(pType,Rvec) pVec2pMat(pType,exp(Rvec))
natpar2tvec.pos <- function(pType,natpar) pMat2pVec10(pType,natpar)$log_()
tvec2natpar.pos <- function(pType,Rvec) pVec2pMat(pType,torch_exp(Rvec))
getZero.pos <- function(pType) {1}

checkParam.unit <- function(pType,par) {
  pVec <-pMat2pVec(pType,as.array(par))
  if (any(is.na(pVec)))
    return("Unexpected NAs in parameter.")
  if (any(pVec<0)||any(pVec>1))
    return("Values outside the range [0,1]")
  NextMethod()
}
natpar2Rvec.unit <- function(pType,natpar) logit(pMat2pVec(pType,natpar))
Rvec2natpar.unit <- function(pType,Rvec) pVec2pMat(pType,invlogit(Rvec))
natpar2tvec.unit <- function(pType,natpar) pMat2pVec10(pType,natpar)$logit_()
tvec2natpar.unit <- function(pType,Rvec) pVec2pMat10(pType,torch_sigmoid(Rvec))
getZero.unit <- function(pType) {.5}

checkParam.pVec <- function(pType,par) {
  pVec <-pMat2pVec(pType,as.array(par))
  if (any(is.na(pVec)))
    return("Unexpected NAs in parameter.")
  if (any(pVec)<0) return("Negative probabilities in parameter.")
  if (abs(sum(pVec)-1)>.0001)
    return("Values do not sum to 1.")
  NextMethod()
}
natpar2Rvec.pVec <- function(pType,natpar) log(pMat2pVec(pType,natpar))
Rvec2natpar.pVec <- function(pType,Rvec) {pVec2pMat(pType,softmax(Rvec))}
natpar2tvec.pVec <- function(pType,natpar) pMat2pVec10(pType,natpar)$log_()
tvec2natpar.pVec <- function(pType,Rvec) pVec2pMat10(pType,nnf_softmax(Rvec))}
getZero.pVec <- function(pType) {.5}

checkParam.cpMat <- function(pType,par) {
  pVeclist <-pMat2rowlist(pType,as.array(par))
  for (pVec in pVeclist) {
    if (any(is.na(pVec)))
      return("Unexpected NAs in parameter.")
    if (any(pVec)<0) return("Negative probabilities in parameter.")
    if (abs(sum(pVec)-1)>.0001)
      return("Row walues do not sum to 1.")
  }
  NextMethod()
}
natpar2Rvec.cpMat <- function(pType,natpar) {
  log(pMat2pVec(pType,natpar))
}
Rvec2natpar.cpMat <- function(pType,Rvec) {
  pVec2pMat(pType,vec2list(lapply(vec2rowlist(pType,Rvec)),softmax))
}
natpar2tvec.cpMat <- function(pType,natpar) {
  pMat2pVec10(pType,natpar)$log_()
}
tvec2natpar.cpMat <- function(pType,Rvec) {
  pVec2pMat10(pType,torch_cat(lapply(vec2rowlist(pType,Rvec),nnf_softmax))
}
getZero.cpMat <- function(pType) {1/pTypeDim(pType)[2]}

natpar2Rvec.const <- function(pType,natpar) numeric()
Rvec2natpar.const <- function(pType,Rvec) numeric()
natpar2tvec.const <- function(pType,natpar) NULL
tvec2natpar.const <- function(pType,Rvec) NULL
getZero.const <- function(pType) {NA}

checkParam.incrK <- function(pType,par) {
  pVeclist <-pMat2collist(pType,as.array(par))
  for (pVec in pVeclist) {
    if (any(is.na(pVec)))
      return("Unexpected NAs in parameter.")
    if (any(diff(pVec) <=0))
      return("Columns are not increasing.")
  }
}
natpar2Rvec.incrK <- function(pType,natpar) {
  list2vec(lapply(pMat2colist(pType,natpar),ldiff))
}
Rvec2natpar.incrK <- function(pType,Rvec) {
  pVec2pMat(pType,list2vec(lapply(vec2collist(pType,Rvec),ecusum)))
}
getZero.incrK <- function(pType) {NA}

natpar2tvec.incrK <- function(pType,natpar) {
  torch_cat(lapply(pMat2colist(pType,natpar),torch_ldiff))
}
tvec2natpar.incrK <- function(pType,Rvec) {
  pVec2pMat10(pType,torch_cat(lapply(vec2collist(pType,Rvec),torch_ecusum)))
}
getZero.incrK <- function(pType) {NA}











