#####################################################
## Primitive Link functions

as_torch_tensor <- function (x) {UseMethod("as_torch_tensor")}
as_torch_tensor.numeric <- function(x) {torch_tensor(x,dtype=torch_float())}
as_torch_tensor.torch_tensor <- function(x) {x}


logit <- function (p) {log(p/(1-p))}
#torch_logit
invlogit <- function (x) {1/(1+exp(-x))}
#torch_sigmoid
linvlogit <- function (x) {log(1/(1+exp(-x)))}
probit <- function (p) {qnorm(p)}
invprobit <- function (x) {pnorm(x)}

logsumexp <- function (x) {log(sum(exp(x)))}

torch_simplexify_ <- function (x,dim=-1L) {
  cpt <- x$abs_()
  cpt$div_(torch_sum(cpt,dim,TRUE))
}
torch_simplexify <- function (x,dim=-1L) {
  cpt <- x$abs()
  cpt$div_(torch_sum(cpt,dim,TRUE))
}



"%//%" <- function (e1,e2) {
  if (is.finite(e2)) e1/e2
  else (as.numeric(!is.finite(e1)))
}
softmax <- function (x) {
  m <- exp(x)
  if (is.matrix(x))
    sweep(m,1,rowSums(m),"%/%")
  else
    m %//% sum(m)
}
lsoftmax <- function(x) {log(softmax(x))}

cloglog <- function(p) {log(-log(1-p))}
torch_cloglog <- function (p) {torch_neg(p)$add_(1)$log_()$neg_()$log_()}
invcloglog <- function (x) {1-exp(-exp(x))}
torch_invcloglog <- function (p) {torch_exp(p)$neg_()$exp_()$neg_()$add_(1)}

tcat <- function(v1,v2) {
  if (length(dim(v1))==0L) v1 <- torch_reshape(v1,-1)
  if (length(dim(v2))==0L) v2 <- torch_reshape(v2,-1)
  torch_cat(list(v1,v2))
}

ldiff <- function (v) {c(v[1],log(diff(v)))}
torch_ldiff <- function(v) {tcat(v[1],torch_diff(v)$log_())}
ecusum <- function (vv) {cumsum(c(vv[1],exp(vv[-1])))}
torch_ecusum <- function (vv) {
  torch_cumsum(tcat(vv[1],torch_exp(vv[2:-1])),1)
}

lldiff <- function (v) {log(c(v[1],diff(v)))}
torch_lldiff <- function(v) {tcat(v[1],torch_diff(v))$log_()}
eecusum <- function (vv) {cumsum(exp(vv))}
torch_eecusum <- function (vv) {torch_cumsum(torch_exp(vv),1)}


stickbreak <- function (p) {
  K <- length(p)-1L
  q <- p[1L:K]
  for (i in 1L:K) {
    q[i] <- 1-q[i]
    if (i<K) q[(i+1L):K] <- q[(i+1L):K]/q[i]
  }
  q
}

invstickbreak <- function (q) {
  qq <- cumprod(c(1,q))
  p <- c(1-q,1)*qq
  p[length(p)] <- 1-p[length(p)]
  p
}

torch_invstickbreak <- function (q) {
  qq <- torch_cat(torch_tensor(1),q)$cumprod_()
  p <- qq$mul_(torch_cat(torch_neg(q)$add_(1),torch_tensor(1)))
  p[-p] <- p[-p]$neg_()$add_(1)
  p
}

torchOpMap <- list(
    "!="= torch_ne,
    "%%"= torch_fmod,
    "&"= torch_logical_and,
    "*"= torch_mul,
    "+"= torch_add,
    "-"= torch_sub,
    "/"= torch_div,
    "<"= torch_lt,
    "<="= torch_le,
    "=="= torch_eq,
    ">"= torch_gt,
    ">="= torch_ge,
    "^"= torch_pow,
    "atan2"= torch_atan2,
    "xor"= torch_logical_xor,
    "|"= torch_logical_or,
    floor_divide= torch_floor_divide,
    pmax= torch_max,
    pmin= torch_min
)

getTorchOp <- function (op) {
  opname <- gsub('"(.*)"',"\\1",deparse(substitute(op)))
  if (is.null(torchOpMap[[opname]]))
    stop("Could not find torch equivalent of ",opname)
  torchOpMap[[opname]]
}

sumrootk <- function(x) {
  sum(x)/sqrt(length(x))
}
torch_sumrootk <- function(x,dim=-1,keepdim=FALSE,out=NULL) {
  result <- torch_sum(x,dim,keepdim,out)
  result$div_(sqrt(x$length()/result$length()))
}


prodq <- function(x) {
  1 - prod(1-x)
}
torch_prodq <- function(x,dim=-1L,keepdim=FALSE,out=NULL) {
  qqq <- torch_prod(torch_ones_like(x)$sub_(x),dim,keepdim,out)
  torch_ones_like(qqq)$sub(qqq)
}

prod_1 <- function(x)
  1 - prod(x)
torch_prod_1 <- function(x,dim=-1L,keepdim=FALSE,out=NULL) {
  torch_prod(x,dim,keepdim,out)$neg_()$add_(torch_tensor(1))
}


torchSummaryMap <- list(
        "max"=torch_amax,
    "min"=torch_amin,
    logsumexp=         torch_logsumexp,
    median=            torch_median,
    mean=               torch_mean,
    nansum=            torch_nansum,
    prod=              torch_prod,
    std=               torch_std,
    std_mean=          torch_std_mean,
    sum=               torch_sum,
    var=               torch_var,
    var_mean=          torch_var_mean,
    sumrootk=          torch_sumrootk,
    prodq=             torch_prodq,
    prod_1=            torch_prod_1
)

getTorchSummaryOp <- function (op) {
  opname <-  gsub('"(.*)"',"\\1",deparse(substitute(op)))
  if (is.null(torchSummaryMap[[opname]]))
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
  opname <- gsub('"(.*)"',"\\1",deparse(substitute(op)))
  if (is.null(zeroMap[[opname]]))
    stop("Could not find torch equivalent of ",opname)
  zeroMap[[opname]]
}

## Define qnorm and pnorm in terms of erf and erfinv
torch_pnorm <- function (x) {
  torch_div(x,sqrt(2))$erf_()$add_(1)$div_(2)
}


torch_qnorm <- function (x) {
  torch_mul(x,2)$sub_(1)$erfinv_()$mul_(sqrt(2))
}




torchUnaryMap <- list(
    "!"= torch_logical_not,
    "-"= torch_neg,
    abs= torch_abs,
    acos= torch_acos,
    acosh= torch_acosh,
    asin= torch_asin,
    asinh= torch_asinh,
    atan= torch_atan,
    atanh= torch_atanh,
    ceil= torch_ceil,
    cos= torch_cos,
    cosh= torch_cosh,
    erf= torch_erf,
    erfc= torch_erfc,
    erfinv= torch_erfinv,
    exp= torch_exp,
    floor= torch_floor,
    frac= torch_frac,
    invlogit= torch_sigmoid,
    invprobit= torch_pnorm,
    linvlogit=nn_log_sigmoid,
    log10= torch_log10,
    log= torch_log,
    logit= torch_logit,
    probit= torch_qnorm,
    reciprocal= torch_reciprocal,
    round= torch_round,
    rsqrt= torch_rsqrt,
    sign= torch_sign,
    sin= torch_sin,
    sinh= torch_sinh,
    sqrt= torch_sqrt,
    square= torch_square
)

getTorchUnaryOp <- function (opname) {
  if (is.null(torchUnaryMap[[opname]]))
    stop("Could not find torch equivalent of ",opname)
  torchUnaryMap[[opname]]
}


torchVectorMap <- list(
    cummax= torch_cummax,
    cummin= torch_cummin,
    cumprod= torch_cumprod,
    cumsum= torch_cumsum,
    diff= torch_diff,
    renorm= torch_renorm,
    t= torch_t,
    "%*%"= torch_matmul,
    softmax= nnf_softmax,
    lsoftmax= nnf_log_softmax
)

getTorchVectorOp <- function (opname) {
  if (is.null(torchVectorMap[[opname]]))
    stop("Could not find torch equivalent of ",opname)
  torchVectorMap[[opname]]
}




## broadcast_all
## torch_flip              rev
## torch_cartesian_prod    Cartesian_prod
## torch_cat               Cat
## torch_clamp             Clamp
## torch_clip              Clip
## torch_narrow            Narrow
## torch_range             Range
## torch_reduction         Creates the reduction object
## torch_reshape           Reshape
## torch_serialize         Serialize a torch object returning a raw object
## torch_split             Split
## torch_squeeze           Squeeze
## torch_where             ifelse
## torch_unsqueeze         Unsqueeze
## torch_embed_diag        diag
## torch_full              rep



guessmat <- function(n,g) {
  mat <- matrix(0,n,n)
  for(nn in 1:(n-1))
    mgcv::sdiag(mat,nn) <- g^nn
  diag(mat) <- 1-rowSums(mat)
  mat
}

torch_guessmat <- function(n,g) {
  mat <- torch_zeros(n,n)
  for (offset in 1L:(n-1L))
    mat <- mat$add_(torch_diag_embed(torch_full(n-offset,g$pow(offset)),
                                     offset))
  mat$add_(torch_diag_embed(mat$sum(2)$neg_()$add_(1),0))
}

slipmat <- function(n,s) {
  mat <- matrix(0,n,n)
  for(nn in 1:(n-1))
    mgcv::sdiag(mat,-nn) <- s^nn
  diag(mat) <- 1-rowSums(mat)
  mat
}

torch_slipmat <- function(n,s) {
  mat <- torch_zeros(n,n)
  for (offset in 1L:(n-1L))
    mat <- mat$add_(torch_diag_embed(torch_full(n-offset,s$pow(offset)),
                                     -offset))
  mat$add_(torch_diag_embed(mat$sum(2)$neg_()$add_(1),0))
}



