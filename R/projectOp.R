#' Turns a function argument into a name (without extra quotes if
#' supplied as a string.  
fname <- function (fn) gsub('"(.*)"',"\\1",deparse(substitute(fn)))



setGeneric("projectOp",function(m1,m2,op="+") standardGeneric("projectOp"))

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
    res[i] <- exec(op,exec("[",m1,as.list(ii1))
                      exec("[",m2,as.list(ii2)))
  }
  res
})


"%^+%" <- function (m1,m2) projectOp(m1,m2,"+")
"%^-%" <- function (m1,m2) projectOp(m1,-m2,"+")
"%^*%" <- function (m1,m2) projectOp(m1,m2,"*")
"%^/%" <- function (m1,m2) projectOp(m1,1/m2,"*")

torchOpMap <- list(
    "+"=torch_add,
    "atan2"=torch_atan2,
    "&"=torch_logical_and,
    "|"=torch_logical_or,
    "xor"=torch_logical_xor,
    "/"=          torch_div,
    "=="=           torch_eq,
    max=          torch_max,
    min=         torch_min,
    floor_divide=      torch_floor_divide,
    "%%"=              torch_fmod,
    ">="=                torch_ge,
    ">"=                torch_gt,
    "<="=                torch_le,
    "<"=                torch_lt,
    "*"=               torch_mul,
    "!="=                torch_ne,
    "-"=               torch_sub,
    "^"=               torch_pow
)

getTorchOp <- function (op) {
  opname <- fname(op)
  if (is.null(torchOpMap[[opname]]))
    stop("Could not find torch equivalent of ",opname)
  torchOpMap[[opname]]
}

setMethod("projectOp",c("torch_tensor","torch_tensor"),
  function (m1,m2,op="+") {
    if (!is.null(getTorchOp(op))) op <- getTorchOp(op)
    do.call(op,list(m1,m2))
  }
)

sumrootk <- function(x)
  sum(x)/sqrt(length(x))
torch_sumrootk <- jit_trace(\(x,dim)
                            torch_sum(x,dim)$div(sqrt(prod(x$shape[dim]))),
                            1:5)

prodq <- function(x)
  1 - prod(1-x)
torch_prodq <- jit_trace(\(x,dim) {
  qqq <- torch_prod(torch_ones_like(x)$sub(x),dim)
  torch_ones_like(qqq)$sub(qqq)
})


torchSummaryMap <- list(
    "max"=torch_amax,
    "min"=torch_amin,
    logsumexp=         torch_logsumexp,
    median=   torch_median,
    mean=               torch_mean,
    nansum=            torch_nansum,
    prod=              torch_prod,
    std=               torch_std,
    std_mean=          torch_std_mean,
    sum=               torch_sum,
    var=               torch_var,
    var_mean=          torch_var_mean,
    
)


getTorchSummaryOp <- function (op) {
  opname <- fname(op)
  if (is.null(torchSummarypMap[[fname(opname)]]))
    stop("Could not find torch equivalent of ",opname)
  torchSummaryMap[[opname]]
}

zeroMap <- list(
    "max"=-Inf,
    "min"=Inf,
    logsumexp=-Inf,
    median=NA,
    mean=NA,
    nansum=0,
    prod=1,
    std=NA,
    std_mean=NA,
    sum=0,
    var=NA,
    var_mean=NA,
    sumrootk=NA,
    prodq=0
)

getZeroOp <- function (op) {
  opname <- fname(op)
  if (is.null(zeropMap[[opname]]))
    stop("Could not find torch equivalent of ",opname)
  zeroMap[[opname]]
}


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


## Define qnorm and pnorm in terms of erf and erfinv
cpt_pnorm <- function (x)
  torch_div(x,sqrt(2))$erf_()$add_(1)$div_(2)
torch_pnorm <- jit_trace(cpt_pnorm,torch_tensor(c(-.67,0,.67)))

cpt_qnorm <- function (x)
  torch_mul(x,2)$sub_(1)$erfinv_()$mul_(sqrt(2))
torch_qnorm <- jit_trace(cpt_qnorm,torch_tensor(c(.25,.5,.75)))



torchUnaryMap <- list(
    abs=torch_abs,
    acos=torch_acos,
    acosh=torch_acosh
    asin=         torch_asin,
    asinh=        torch_asinh,
    atan=         torch_atan,
    atanh=             torch_atanh,
    ceil=              torch_ceil,
    cos=               torch_cos,
    cosh=              torch_cosh,
    erf=               torch_erf,
    erfc=              torch_erfc,
    erfinv=            torch_erfinv,
    exp=               torch_exp,
    floor=             torch_floor,
    frac=              torch_frac,
    log=               torch_log,
    log10=             torch_log10,
    "!"=       torch_logical_not,
    logit=             torch_logit,
    "-"=               torch_neg,
    reciprocal=        torch_reciprocal,
    round=             torch_round,
    rsqrt=             torch_rsqrt,
    sign=              torch_sign,
    sin=               torch_sin,
    sinh=              torch_sinh,
    sqrt=              torch_sqrt,
    square=            torch_square,
    invlogit= torch_sigmoid,
    linvlogit=nn_log_sigmoid,
    probit=torch_qnorm,
    invprobit=torchpnorm,
    
)

getTorchUnaryOp <- function (opname) {
  if (is.null(torchUnarypMap[[opname]]))
    stop("Could not find torch equivalent of ",opname)
  torchUnaryMap[[opname]]
}


torchVectorMap <- list(
    cummax=            torch_cummax,
    cummin=            torch_cummin,
    cumprod=           torch_cumprod,
    cumsum=            torch_cumsum,
    diff=              torch_diff,
    renorm=            torch_renorm,
    t=                 torch_t,
    "%*%"=            torch_matmul,
    softmax=		nnf_softmax,
    invlogit=              nnf_sigmoid,
    lsoftmax=		nnf_log_softmax,
    linvlogit=              nnf_logsigmoid,
)

getTorchVectorOp <- function (opname) {
  if (is.null(torchVectorpMap[[opname]]))
    stop("Could not find torch equivalent of ",opname)
  torchVectorMap[[opname]]
}


## broadcast_all
## torch_cartesian_prod    Cartesian_prod
## torch_cat               Cat
## torch_clamp             Clamp
## torch_clip              Clip
## torch_narrow            Narrow
## torch_range             Range
## torch_reduction         Creates the reduction objet
## torch_reshape           Reshape
## torch_serialize         Serialize a torch object returning a raw object
## torch_split             Split
## torch_squeeze           Squeeze
## torch_where             ifelse
## torch_unsqueeze         Unsqueeze



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

  
