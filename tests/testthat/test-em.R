


expect_eqten <- function (obsdist,truedist,tol=.0001,what="tensor") {
  maxdistdif <- as.numeric(torch_sub(obsdist,truedist)$abs_()$max())
  if (maxdistdif > tol) {
    fail(paste("Observed and expected", what, "differ, maximum difference ",maxdistdif))
  } else {
    succeed(paste("Observed",what,"matches expected."))
  }
}

test_that("Build Loss Function.",{

  mod0 <- CPT_Model$new("Center","Normal",list(),c("A","B","C"))
  mod0$linkScale <- 1
  mod0$penalties <- list("bVec"=1,"sVec"=1)

  cpt <- mod0$getCPT()

  lf <-  build_loss_fun(mod0$ccbias,
                        mod0$penalities)
  expect_equal(as.numeric(lf(cpt,cpt,mod0$params())),
               -11*log(1/3),
               tolerance=.0001)
  expect_true(TRUE)

})


test_that("Recovery Normal 0 parents", {

 mod0 <- CPT_Model$new("Center","Normal",list(),c("A","B","C"))
 mod0$linkScale <- 1
 mod0$penalties <- list("bVec"=1,"sVec"=1)
 mod1 <- CPT_Model$new("Center","Normal",list(),c("A","B","C"))
 mod1$linkScale <- .5
 mod1$bMat <- matrix(.5,1,1)
 cpt1 <- mod1$getCPT()
 dattab <- torch_mul(cpt1,1000)

 conv <- fit2table(mod0,dattab,log=c("bVec","sVec","cpt"),maxit=100L)
 if (!conv) warning("Model fitting did not converge")

 expect_eqten(mod0$getCPT(),cpt1)

})

test_that("Recovery Normal 1 parent", {

})

test_that("Recovery Compensatory GR", {

})

test_that("Recovery Compensatory PC", {

})

test_that("Recovery Conjunctive PC", {

})

test_that("Recovery Conjunctive PC Guess/Slip", {

})
