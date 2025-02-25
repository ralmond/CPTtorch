LinkFunction <- function(name,fun,etWidth=-1L,
                         scaleType=character()) {
  if (length(scaleType)==0L) {
    sPType<-NULL
  } 
  if (length(scaleType) > 1L) {
    stop("Unrecognized link scale type ", scaleType)
  }
  if (is.character(scaleType)) {
    sPType <- getPType(scaleType) 
    if (is.null(sPType))
      stop("Unrecognized link scale type ", scaleType)
  }
  if (!is.null(sPType) && !is(sPType,"PType")) {
    stop("Invalid scale type",scaleType)
  } 
  if (length(etWidth)!=1L)
    stop("etWidth must be a integer vector of length 1")
  res <- list(name=name,fun=fun,etWidth=as.integer(etWidth),
              sPType=sPType)
  class(res) <- "LinkFunction"
  res
}
setOldClass("LinkFunction")



LinkedList <- list()
getLink <- function (linkname) {
  whichType <- pmatch(linkname,names(LinkedList))
  if (is.na(whichType)) return (NULL)
  LinkedList[[whichType]]
}
setLink <- function (linkname,ptype) {
  if (!is.null(ptype) && !is(ptype,"PType"))
    stop("Second argument must be a PType or NULL.")
  LinkedList[[linkname]] <- ptype
  assignInMyNamespace("LinkedList",LinkedList)
}
availableLinkss <- function() {
  names(LinkedList)
}




setGeneric("linkFast",function(link,et,linkScale=NULL,nobs=2)
  standardGeneric("linkFast"))
setMethod("linkFast",c("LinkFunction"),
          function(link,et,linkScale=NULL,nobs=2) {
            link$fun(et,linkScale,nobs)
          })
setMethod("linkFast",c("character"),
          function(link,et,linkScale=NULL,nobs=2) {
            whichLink <- pmatch(link,names(LinkedList))
            if (is.null(whichLink))
              stop ("No link function registered for ",link)
            LinkedList[[whichLink]]$fun(et,linkScale,nobs)
          })


setGeneric("linkSafe",function(link,et,linkScale=NULL,obsLevels=NULL)
  standardGeneric("linkSafe"))
setMethod("linkSafe",c("LinkFunction"),
          function(link,et,linkScale=NULL,obsLevels=NULL) {
            mm <- ifelse(link$etWidth>0L,link$etWidth,nobs+link$etWidth)
            if (mm > 0L && mm != ncol(et)) {
              stop("The et matrix should have ", mm, " columns.")
            }
            whichLink <- pmatch(link,names(LinkedList))
            if (is.null(whichLink))
              stop ("No link function registered for ",link)
            sptype <- LinkedList[[whichLink]]$sPType
            if (is.null(sptype)) {
              if(length(linkScale)>0L)
                stop("No scale parameter expected, but one was supplied.")
            } else {
              typeCheck <- sptype$checker(linkScale)
              if (!isTRUE(typeCheck))
                stop("Bad LinkScale: ",typeCheck)
            }
            probs <- LinkedList[[whichLink]]$fun(et,linkScale,nobs)
            if (!is.null(obsLevels)) {
              dimnames(probs) <- list(NULL,obsLevels)
            }
            probs
          })
setMethod("linkSafe",c("character"),
          function(link,et,linkScale=NULL,obsLevels=NULL) {
            whichLink <- pmatch(link,names(LinkedList))
            if (is.null(whichLink))
              stop ("No link function registered for ",link)
            linkSafe(LinkedList[[whichLink]],et,linkScale,obsLevels)
          })




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

identityLink0 <- function(et,linkScale=NULL,nobs=2) {
  et <- ifelse(et<0,0,et)
  sweep(et,1,apply(et,1,sum),"/")
}

LinkedList$identity <- LinkFunction("identity",identityLink0,0,NULL)

stepProbLink0 <- function (et,LinkScale=NULL,nobs=2) {
  zt <- apply(cbind(et,1),1,function(x) rev(cumprod(rev(x))))
  identityLink0(zt,LinkScale)
}

LinkedList$stepProb <- LinkFunction("stepProb",stepProbLink0,-1,NULL)


diffLink0 <- function(et,LinkScale=NULL,nobs=2) {
  m <- ncol(et)+1
  pt <- apply(cbind(0,et,1),1,cummax)
  pt[,2:(m+1),drop=FALSE]-pt[,1:m,drop=FALSE]
  identityLink0(pt,LinkScale)
}

LinkedList$diff <- LinkFunction("diff",diffLink0,-1,NULL)


softMax0 <- function(et,linkScale=NULL,nobs=2) {
  identityLink0(exp(1.7*et),linkScale)
}

LinkedList$softMax <- LinkFunction("softMax",softMax0,0,NULL)

partialCredit0 <- function (et,linkScale=NULL,nobs=2) {
  zt <- apply(cbind(et,0),1,function(x) rev(cumsum(rev(x))))
  identityLink0(exp(1.7*zt),linkScale)
}



LinkedList$partialCredit <- LinkFunction("partialCredit",partialCredit0,
                                         -1,NULL)

gradedResponse0 <- function (et,linkScale=NULL,nobs=2) {
  zt <- 1/(1+exp(-1.7*cbind(-Inf,et,Inf)))
  pt <- apply(zt,1,cummax)
  pt[,2:(nobs+1),drop=FALSE]-pt[,1:nobs,drop=FALSE]
}


LinkedList$gradedResponse <- LinkFunction("gradedResponse",gradedResponse0,
                                         -1,NULL)


normalLink0 <- function(et,linkScale=NULL,nobs=2) {
  m <- nobs
  cuts <- qnorm( ((m-1):1)/m)
  ## Only play attention to the first column.
  pt <- pnorm(outer(-et[,1],cuts,"+")/linkScale)
  pt <- cbind(1,pt,0)
  pt[,1:m]-pt[,1+(1:m)]
}

LinkedList$normal <- LinkFunction("normal",normalLink0,1,"pos")


noisyLink0 <- function(et,linkScale=NULL,nobs=2) {
  et <- cbind(et,1-rowSums(et))
  identityLink0(et %*% linkScale)
}

LinkedList$noisy <- LinkFunction("noisy",noisyLink0,-1,"cpmat")


guessmat <- function(n,g) {
  mat <- matrix(0,n,n)
  for(nn in 1:(n-1)) 
    mgcv::sdiag(mat,nn) <- g^nn
  diag(mat) <- 1-rowSums(mat)
  mat
}
slipmat <- function(n,s) {
  mat <- matrix(0,n,n)
  for(nn in 1:(n-1)) 
    mgcv::sdiag(mat,-nn) <- s^nn
  diag(mat) <- 1-rowSums(mat)
  mat
}

slipLink0 <- function(et,linkScale=NULL,nobs=2) {
  mat <- slipmat(nobs,linkScale)
  noisyLink0(et,mat)
}

LinkedList$slip <- LinkFunction("slip",slipLink0,-1,"unit")


guessLink0 <- function(et,linkScale=NULL,nobs=2) {
  mat <- guessmat(nobs,linkScale)
  noisyLink0(et,mat)
}

LinkedList$guess <- LinkFunction("guess",guessLink0,-1,"unit")

slipGuessLink0 <- function(et,linkScale=NULL,nobs=2) {
  mat <- guessmat(nobs,linkScale[2])%*%
    slipmat(nobs,linkScale[1])
  noisyLink0(et,mat)
}

LinkedList$slipGuess <- LinkFunction("slipGuess",slipGuessLink0,-1,
                                     "unit")

