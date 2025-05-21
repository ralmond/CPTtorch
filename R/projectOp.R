#' Turns a function argument into a name (without extra quotes if
#' supplied as a string.
fname <- function (fn) gsub('"(.*)"',"\\1",deparse(substitute(fn)))



setGeneric("projectOp",function(m1,m2,op="+") standardGeneric("projectOp"))
setOldClass("torch_tensor")

setMethod("projectOp",c("numeric","numeric"), function(m1,m2,op="+")
  do.call(op,list(m1,m2)))
setMethod("projectOp",c("array","numeric"), function(m1,m2,op="+")
  do.call(op,list(m1,m2)))
setMethod("projectOp",c("numeric","array"), function(m1,m2,op="+")
  do.call(op,list(m1,m2)))

setMethod("projectOp",c("array","array"), function (m1,m2,op="+") {
  dim1 <- dim(m1)
  dim2 <- dim(m2)
  if (length(dim1) != length(dim2))
    stop("Arrays have different dimensions.")
  if (all(dim1==1L) || all(dim2==1L) || all(dim1==dim2))
    return (do.call(op,list(m1,m2)))
  if (!all(dim1=dim2 | dim1==1L | dim2==1L))
    abort("Non-confomable arrays.")
  resdim <- pmax(dim1,dim2)
  stride <- cumprod(c(1,resdim))[1L:length(resdim)]
  res <- array(NA_real_,resdim)
  for (i in 1:length(res)) {
    ii <- ((i-1) %/% stride %% resdim) +1
    ii1 <- ifelse(dim1==1L,1L,ii)
    ii2 <- ifelse(dim2==1L,1L,ii)
    res[i] <- exec(op,exec("[",m1,as.list(ii1)),
                      exec("[",m2,as.list(ii2)))
  }
  res
})


"%^+%" <- function (m1,m2) projectOp(m1,m2,"+")
"%^-%" <- function (m1,m2) projectOp(m1,-m2,"+")
"%^*%" <- function (m1,m2) projectOp(m1,m2,"*")
"%^/%" <- function (m1,m2) projectOp(m1,1/m2,"*")

setMethod("projectOp",c("torch_tensor","torch_tensor"),
  function (m1,m2,op="+") {
    if (!is.null(getTorchOp(op))) op <- getTorchOp(op)
    do.call(op,list(m1,m2))
  }
)




setGeneric("marginalize",function(pot,dim=1,op="sum")
  StandardGeneric("marginalize"))

setMethod("marginalize","array",function(pot,dim=1,op="sum") {
  odim <- setdiff(1L:length(dim(pot)),dim)
  apply(pot,odim,op)
})

setMethod("marginalize","torch_tensor",function(pot,dim=1,op="sum") {
  if (!is.null(getTorchSummaryOp(op))) op <- getTorchSummaryOp(op)
  do.call(op,list(pot,dim))
})





################################################
## Matrix Multiplication


genmmttab <- new.env()


fetchMMt <- function(combOp,summaryOp)
  genmmttab[[paste("MMt",fname(combOb),fname(summaryOp),sep="_")]]
fetchMMtQ <- function(combOp,summaryOp)
  genmmttab[[paste("MMtQ",fname(combOb),fname(summaryOp),sep="_")]]
setMMt <- function(combOp,summaryOp,fval)
  genmmttab[[paste("MMt",fname(combOb),fname(summaryOp),sep="_")]] <-
    fval
setMMtQ <- function(combOp,summaryOp,fval)
  genmmttab[[paste("MMtQ",fname(combOb),fname(summaryOp),sep="_")]] <-
    fval



genMMt.matrix <- function(m1,m2,combOp,summaryOp) {
  result <- matrix(NA_real_,nrow(m1),nrow(m2))
  for (cc in 1L:nrow(m2))
    result[,cc] <- apply(sweep(m1,2,m2[cc,],combOp),1,summaryOp)
  result
}


genMMt.tt <- function(m1,m2,combOp,summaryOp) {
  result <- torch_empty(nrow(m1),nrow(m2))
  for (cc in 1L:nrow(m2))
    result[,cc] <- exec(summaryOp,exec(combOp,m1,m2[cc,]),2)
  result
}

make_MMt <- function(combOp,summaryOp) {
  if (!is.null(getTorchOp(combOp))) combOp <- getTorchOp(combOp)
  if (!is.null(getTorchOp(summaryOp))) combOp <- getTorchOp(summaryOp)
  function(m1,m2) genMMt.tt(m1,m2,summaryOp,combOp)
}

supplyMMt <- function(combOp,summaryOp,m1,m2) {
  if (is.null(fetchMMt(combOp,summaryOp))) {
    setMMt(combOp,summaryOp,
           jit_trace(make_MMt(combOp,summaryOp),m1,m1))
  }
  fetchMMt(combOp,summaryOp)
}


setGeneric("genMMt",function(m1,m2,combOp,summaryOp)
  standardGeneric("genMMt"))
setMethod("genMMt",c("matrix","matrix"),genMMt.matrix)
setMethod("genMMt",c("torch_tensor","torch_tensor"),
          function(m1,m2,combOp,summaryOp)
            exec(supplyMMt(combOp,summaryOp,m1,m2),m1,m2))

## Shortcut for this operator.
setMMt("*","sum",torch_matmul)


genMMtQ.matrix <- function(m1,m2,QQ,combOp,summaryOp) {
  result <- matrix(NA_real_,nrow(m1),nrow(m2))
  for (cc in 1L:nrow(m2))
    result[,cc] <- apply(sweep(m1[,QQ[cc,]],2,m2[cc,QQ[cc,]],combOp),1,
                         summaryOp)
  result
}


genMMtQ.tt <- function(m1,m2,QQ,combOp,summaryOp) {
  result <- torch_empty(nrow(m1),nrow(m2))
  for (cc in 1L:nrow(m2))
    result[,cc] <- exec(summaryOp,exec(combOp,m1[1,QQ[cc,]],
                                              m2[cc,QQ[cc,]]),2)
  result
}

make_MMtQ <- function(combOp,summaryOp) {
  if (!is.null(getTorchOp(combOp))) combOp <- getTorchOp(combOp)
  if (!is.null(getTorchOp(summaryOp))) combOp <- getTorchOp(summaryOp)
  function(m1,m2) genMMtQ.tt(m1,m2,summaryOp,combOp)
}

supplyMMtQ <- function(combOp,summaryOp,m1,m2) {
  if (is.null(fetchMMtQ(combOp,summaryOp))) {
    setMMtQ(combOp,summaryOp,
           jit_trace(make_MMt(combOp,summaryOp),m1,m1))
  }
  fetchMMtQ(combOp,summaryOp)
}


setGeneric("genMMtQ",function(m1,m2,QQ,combOp,summaryOp)
  standardGeneric("genMMtQ"))
setMethod("genMMtQ",c("matrix","matrix","matrix"),genMMtQ.matrix)
setMethod("genMMtQ",c("torch_tensor","torch_tensor","torch_tensor"),
          function(m1,m2,QQ,combOp,summaryOp)
            exec(supplyMMtQ(combOp,summaryOp,m1,m2,QQ),m1,m2,QQ))

setMMtQ("*","sum",function(m1,m2,QQ)
  torch_matmul(m1,torch_where(QQ,m2,torch_zeros_like(m2)))
  )


