test_that("logit",{
  expect_equal(invlogit(logit(.7)),.7)
  expect_equal(logit(invlogit(1.5)),1.5)
})

test_that("softmax",{
  expect_equal(softmax(rep(1,4)),rep(1/4,4))
  expect_equal(softmax(log(1:3)),(1:3)/6)
})

test_that("cloglog",{
  expect_equal(invcloglog(cloglog(.7)),.7)
  expect_equal(cloglog(invcloglog(1.5)),1.5)
  expect_equal(as.numeric(torch_invcloglog(
    torch_cloglog(torch_tensor(.7)))),.7,tolerance=.0001)
  expect_equal(as.numeric(torch_cloglog(
    torch_invcloglog(torch_tensor(1.5)))),1.5,tolerance=.00001)
})

test_that("ldiff",{
  expect_equal(ecusum(ldiff(0:5)),0:5)
  expect_equal(ldiff(ecusum(rep(0,5))),rep(0,5))
})

test_that("ldiff torch",{
  tv <- torch_tensor(0:5,dtype=torch_float())
  expect_equal(as.numeric(torch_ecusum(torch_ldiff(tv))),
               0:5,tolerance=.00001)
  tv0 <- torch_tensor(rep(0,5),dtype=torch_float())
  expect_equal(as.numeric(torch_ldiff(torch_ecusum(tv0))),rep(0,5),
               tolerance=.00001)
})

test_that("lldiff",{
  expect_equal(eecusum(lldiff(1:5)),1:5)
  tv <- torch_tensor(1:5,dtype=torch_float())
  expect_equal(as.numeric(torch_eecusum(torch_lldiff(tv))),1:5,
               tolerance=.00001)

  expect_equal(lldiff(eecusum(rep(0,5))),rep(0,5))
  tv0 <- torch_tensor(rep(0,5),dtype=torch_float())
  expect_equal(as.numeric(torch_lldiff(torch_eecusum(tv0))),rep(0,5),
               tolerance=.00001)
})


test_that("stickbreak",{
  pv <- (1:3)/6
  expect_equal(invstickbreak(stickbreak(pv)),pv)
  lpv <- c(1.5,.5)
  expect_equal(stickbreak(invstickbreak(lpv)),lpv)
})

test_that("logsumexp",{
  expect_equal(logsumexp(c(0,0)),log(2))
})

test_that("simplexify",{
  t1 <- torch_tensor(1.0:3.0,torch_double())
  t1norm <- as.numeric(torch_simplexify(t1))
  expect_equal(t1norm,(1:3)/6)
  t1[1] <- -1
  t1norm <- as.numeric(torch_simplexify_(t1))
  expect_equal(t1norm,(1:3)/6)
  expect_equal(as.numeric(t1),(1:3)/6)

})

test_that("prodq",{
  expect_equal(prodq(c(.9,.75)),.975)
  expect_equal(as.numeric(torch_prodq(torch_tensor(c(.9,.75)))),.975,
               tolerance=.00001)

})

test_that("sumrootk",{
  root22 <- rep(sqrt(2),2)
  expect_equal(sumrootk(root22),2,tolerance=.00001)
  expect_equal(as.numeric(torch_sumrootk(torch_tensor(root22))),2,
               tolerance=.00001)
})

test_that("torch_pnorm,qnorm",{
  expect_equal(as.numeric(torch_pnorm(torch_tensor(.25))),
               pnorm(.25), tolerance=.00001)
  expect_equal(as.numeric(torch_qnorm(torch_tensor(.25))),
               qnorm(.25), tolerance=.00001)
})

test_that("guessmat",{
  gm3.5 <- matrix(c(.25,.5,.25, 0,.5,.5, 0,0,1), 3,3, byrow=TRUE)
  expect_equal(guessmat(3,.5),gm3.5)
  expect_equal(as.matrix(torch_guessmat(3,.5)),gm3.5)
})

test_that("slipmat",{
  sm3.5 <- matrix(c(1,0,0, .5,.5,0, .25,.5,.25), 3,3, byrow=TRUE)
  expect_equal(slipmat(3,.5),sm3.5)
  expect_equal(as.matrix(torch_slipmat(3,.5)),sm3.5)
})


