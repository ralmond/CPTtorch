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
})

test_that("ldiff",{
  expect_equal(ecusum(ldiff(0:5)),0:5)
  expect_equal(ldiff(ecusum(rep(0,5))),rep(0,5))
})

test_that("lldiff",{
  expect_equal(eecusum(lldiff(1:5)),1:5)
  expect_equal(lldiff(eecusum(rep(0,5))),rep(0,5))
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

# sumrootk
# torch_pnorm, qnorm
# guessmat
# slipmat

