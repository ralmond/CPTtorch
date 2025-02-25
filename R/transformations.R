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
