#####################################################
## Primitive Link functions



logit <- function (p) log(p/(1-p))
#torch_logit
invlogit <- function (x) 1/(1+exp(-x))
#torch_sigmoid
linvlogit <- function (x) log(1/(1+exp(-x)))
probit <- function (p) qnorm(p)
invprobit <- function (x) dnorm(x)

logsumexp <- function (x) log(sum(exp(x)))

torch_2simplex(x,dim=2) {
  cpt <- x$abs_()
  cpt$div_(torch_sum(cpt,dim))
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
lsoftmax <- function(x) log(softmax(x))

cloglog <- function(p) log(-log(1-p))
torch_cloglog <- function (p) torch_neg(p)$add_(1)$log_()$neg_()$log_()
invcloglog <- function (x) 1-exp(-exp(x))
torch_invcloglog <- function (p) torch_exp(p)$neg_()$exp_()$neg_()$add_(1)

ldiff <- function (v) c(v[1],log(diff(v)))
torch_ldiff <- function(v) torch_cat(v[1],torch_diff(v)$log_())
ecusum <- function (vv) cumsum(c(vv[1],exp(vv[-1])))
torch_ecusum <- function (vv) torch_cat(vv[1],torch_exp(vv[2:-1]))$cumsum_()

lldiff <- function (v) log(c(v[1],diff(v)))
torch_ldiff <- function(v) torch_cat(v[1],torch_diff(v))$log_()
eecusum <- function (vv) cumsum(exp(vv))
torch_ecusum <- function (vv) torch_exp(vv)$cumsum_()


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



guessmat <- function(n,g) {
  mat <- matrix(0,n,n)
  for(nn in 1:(n-1)) 
    mgcv::sdiag(mat,nn) <- g^nn
  diag(mat) <- 1-rowSums(mat)
  mat
}

torch_guessmat <- function(n,g) {
  mat <- torch_diagonal(

slipmat <- function(n,s) {
  mat <- matrix(0,n,n)
  for(nn in 1:(n-1)) 
    mgcv::sdiag(mat,-nn) <- s^nn
  diag(mat) <- 1-rowSums(mat)
  mat
}



## broadcast_all
## torch_flip              rev
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
## torch_embed_diag        diag



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
    mat <- mat$add_(torch_embed_diag(torch_tensor(rep(g^offset,n-offset)),
                                     offset))
  mat$add_(torch_embed_diag(mat$sum_(2)$neg_()$add_(1)),0)
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
    mat <- mat$add_(torch_embed_diag(torch_tensor(rep(s^offset,n-offset)),
                                     offset))
  mat$add_(torch_embed_diag(mat$sum_(2)$neg_()$add_(1)),0)
}



