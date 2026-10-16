
buildProficiencies <- function (factors, tred=TRED) {
  cpt <- factors[[1]]
  prof <- cpt$getCPT()$reshape(cpt$fullDims)
  for (cpt in factors[-1]) {
    prof <- prof$mul(cpt$getCPT()$reshape(cpt$fullDims))
  }
  tred$SETt("CPT.prof",prof)
  prof
}



scoreOne <- function(isubj,data=DATA,tred=TRED){
  sm <- tred$GETt("CPT.prof")
  for (task in names(data)) {
    if (is.na(data[isubj,task])) next
    em <- tred$GETt(paste("CPT",task,data[isubj,task],
                          sep="."))
    sm <- sm$mul_(em)
  }
  sm
}

calcStats <- function(sm,isubj,statlist=STATLIST) {
  results <- lapply(statlist,\(s) {s$calc(sm)})
  names(results) <- names(statlist)
  as.data.frame(
    lapply(do.call("c",c(list(isubj=isubj),results)),
           \(x) ifelse(is.na(as.numeric(x)),x,as.numeric(x))))
}

scoreStats <- function(isubj,data=DATA,statlist=STATLIST,tred=TRED) {
  res <- scoreOne(isubj,data,tred) |>
    calcStats(isubj,statlist)
  res
}

expMatProf <- function (sm, omega=OMEGA,tred=TRED) {
  for (fact in rownames(omega)) {
    margin <- which(!omega[fact,])
    tab <- sm$sum(margin)
    tred$PUSHt(paste("Dat",fact,sep="."),tab)
  }
}

expMatEvid <- function (sm, isubj, data=DATA, qq=QQ, tred=TRED) {
  for (task in colnames(data)) {
    if (is.na(data[isubj,task])) next
    margin <- which(!qq[task,])
    tab <- sm$sum(margin)
    tred$PUSHt(paste("Dat",task,data[isubj,task],sep="."),tab)
  }
}
EStepScore <- function(isubj,data=DATA,omega=OMEGA,qq=QQ,tred=TRED) {
  tryCatch({
    sm <- scoreOne(isubj,data,tred)
    expMatProf(sm,omega,tred)
    expMatEvid(sm,isubj,data,qq,tred)
  },
  error=function(e) list("Estep Error",conditionMessage(e),
                         conditionCall(e))
  )
}


MstepEvid <- function (imod,ems=EMS,options=OPTIONS,tred=TRED) {
  cpt <- ems[[imod]]
  shp1 <- cpt$shp
  shp1[length(shp1)]<-1
  chunks <- lapply(1:length(cpt$stateNames),\(ichunk) {
    tag <- paste("Dat",cpt$child,ichunk,sep=".")
    tab <- tred$POPt(tag)
    tab1 <- tred$POPt(tag)
    while (!is.null(tab1)) {
      with_no_grad(tab <- tab$add_(tab1))
      tab1 <- tred$POPt(tag)
    }
    tab <- torch_reshape(tab,shp1)
  })
  dattab <- torch_cat(chunks,length(shp1))
  fit2table(cpt,dattab,options)
  tred$SETCHUNKS(cpt)
  lp <- cpt$deviance(dattab)
  tred$PUSHs("lp",as.numeric(lp))
}


MstepProf <- function(factors=FACTORS,options=OPTIONS,tred=TRED) {
  factors <- lapply(factors, \(fact) {
    tag <- paste("Dat",fact$child,sep=".")
    tab <- tred$POPt(tag)
    tab1 <- tred$POPt(tag)
    while (!is.null(tab1)) {
      with_no_grad(tab <- tab$add_(tab1))
      tab1 <- tred$POPt(tag)
    }
    fit2table(fact,tab,options)
    lp <- fact$deviance(tab)
    tred$PUSHs("lp",as.numeric(lp))
    fact
  })
  prof <- buildProficiencies(factors,tred)
}

### when first arg is zero, does MstepProf, otherwise MstepEvid
Mstep <- function(imod,ems=EMS,factors=FACTORS,options=OPTIONS,tred=TRED) {
  tryCatch({
    if (imod == 0)
      MstepProf(factors,options,tred)
    else
      MstepEvid(imod,ems,options,tred)
  },
  error=function(e) list("Mstep Error",conditionMessage(e),
                         conditionCall(e))
  )
}

calcLP <- function(tred=TRED) {
  lp <- 0
  lp1 <- tred$POPs("lp")
  while(!is.null(lp1)) {
    lp <- lp + lp1
    lp1 <- tred$POPs("lp")
  }
  lp
}

EMloop <- function(cl,data=DATA,ems=EMS,options=OPTIONS,tred=TRED) {
  ## Decode Options
  maxit <- options$maxit
  if (!is.numeric(maxit)) maxit
  tolerance <- options$tolerance
  if (!is.numeric(tolerance)) tolerance <- .0001
  progress <- options$progress
  if (!is.numeric(progress)) progress <- 10L

  lp0 <- -Inf
  nit <- 0L
  while (nit < maxit) {
    nit <- nit + 1L
    if ( (nit %% progress) ==0L) cat("Iteration ",nit,"\n")
    parallel::parLapply(cl,1:nrow(data),EStepScore)
    parallel::parLapply(cl,0:length(ems),Mstep)
    lp <- calcLP(tred)
    if (abs(lp-lp0)<tolerance) break
    lp0 <- lp
  }
  return (nit < maxit)
}
