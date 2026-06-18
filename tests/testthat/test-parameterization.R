# Tests for the aMat/bMat parameterization toggle ("view" vs "direct").
#
# "view" (default): aVec/bVec are unconstrained; aMat/bMat are views via
#   tvec2natpar (e.g. pos -> exp, incrK -> cumsum).
# "direct": aVec/bVec hold the natural-scale (used) entries directly; aMat/bMat
#   round-trip via pVec2pMat10 with no scale transform or constraint enforcement.
#
# The behavioral payoff (direct reproduces the pre-view-refactor, dispersed-
# discrimination EM fits) is an EM-level property validated downstream against
# the real pipeline; a single-task M-step optimum is identical in both modes, so
# these tests pin the *contract* rather than a fit difference.

test_that("parameterization: construction, default, accessor, validation", {
  v <- CPT_Model$new("CompensatoryGR","GradedResponse",
                     list(A=c("L1","L2")), c("s1","s2","s3"))
  d <- CPT_Model$new("CompensatoryGR","GradedResponse",
                     list(A=c("L1","L2")), c("s1","s2","s3"),
                     parameterization="direct")
  expect_equal(v$rule$parameterization, "view")     # default
  expect_equal(d$rule$parameterization, "direct")
  expect_error(d$rule$parameterization <- "view")   # fixed at construction
  expect_error(                                      # match.arg rejects unknown
    CPT_Model$new("CompensatoryGR","GradedResponse",
                  list(A=c("L1","L2")), c("s1","s2","s3"),
                  parameterization="bogus"))
})

test_that("parameterization: identical initial CPT and parameter count", {
  args <- list("CompensatoryGR","GradedResponse",
               list(A=c("L1","L2")), c("s1","s2","s3"))
  v <- do.call(CPT_Model$new, args)
  d <- do.call(CPT_Model$new, c(args, list(parameterization="direct")))
  # defaults round-trip: same natural starting CPT in both modes
  expect_equal(as.matrix(v$forward()), as.matrix(d$forward()), tolerance = 1e-6)
  # aVec/bVec stay 1-D (whichUsed-masked) in both modes -> numparams unchanged
  expect_equal(names(v$params()), names(d$params()))
  expect_equal(v$numparams(), d$numparams())
  expect_equal(length(v$rule$aVec), length(d$rule$aVec))
})

test_that("parameterization: view stores transformed vec, direct stores natural", {
  # Compensatory aType is "pos": view aVec = log(aMat), direct aVec = aMat.
  mk <- function(mode) getRule("Compensatory")$new(
    as_Tvallist(list(A=c(0,1),B=c(0,1))), 2, parameterization=mode)
  v <- mk("view"); d <- mk("direct")
  amat <- matrix(2, 2, 2)                       # pos aType dim is c(K=2, J=2)
  v$aMat <- as_torch_tensor(amat)
  d$aMat <- as_torch_tensor(amat)
  # both read back the same natural matrix
  expect_equal(as.matrix(v$aMat), amat, tolerance = 1e-5)
  expect_equal(as.matrix(d$aMat), amat, tolerance = 1e-5)
  # but the trained vector differs: direct = natural (2), view = log(2)
  expect_equal(as.numeric(d$aVec), rep(2, 4), tolerance = 1e-5)
  expect_equal(as.numeric(v$aVec), rep(log(2), 4), tolerance = 1e-4)
})

test_that("parameterization direct: unconstrained set/get round-trip + autograd", {
  d <- CPT_Model$new("CompensatoryGR","GradedResponse",
                     list(A=c("L1","L2"), B=c("L1","L2")), c("s1","s2","s3"),
                     parameterization="direct")
  aD <- dim(d$rule$aMat)                          # pos c(1, J=2)
  # discrimination values the pos transform would handle differently;
  # direct stores them verbatim
  d$rule$aMat <- as_torch_tensor(matrix(c(2.5, 0.4), aD[1], aD[2]))
  expect_equal(as.numeric(d$rule$aMat), c(2.5, 0.4), tolerance = 1e-5)
  # gradient flows back to the registered aVec
  d$ccbias <- 0
  datatab <- torch_ones_like(d$forward())
  d$buildOptimizer()
  d$lossfn(dattab = datatab, d$forward(), d$params())$backward()
  expect_false(is.null(d$rule$aVec$grad))
})
