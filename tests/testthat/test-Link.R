test_that("Link initialize", {

  al <- CPT_Link$new(2,guess=.2,slip=.3,high2low=TRUE)
  expect_s3_class(al,"CPT_Link")
  expect_equal(al$K,2)
  expect_equal(as.numeric(al$slip),.3,tolerance=.00001)
  expect_equal(as.numeric(al$guess),.2,tolerance=.00001)
  expect_equal(al$high2low,TRUE)

})

test_that("Link leakMat", {
  al <- CPT_Link$new(3)
  expect_null(al$leakmat())
  al <- CPT_Link$new(3,guess=.3)
  expect_equal(as.matrix(al$leakmat()),guessmat(3,.3)[,3:1],tolerance=.00001)
  al <- CPT_Link$new(3,slip=.2)
  expect_equal(as.matrix(al$leakmat()),slipmat(3,.2)[,3:1],tolerance=.00001)
  al <- CPT_Link$new(3,slip=.2,guess=.3)
  expect_equal(as.matrix(al$leakmat()),guessmat(3,.3)[,3:1]%*%slipmat(3,.2)[,3:1],
               tolerance=.00001)
})


test_that("Link link forward", {
  gl <- PotentialLink$new(3)
  et <- torch_eye(3)
  expect_equal(as.matrix(gl$link(et)),diag(3),tolerance=.00001)
  expect_equal(as.matrix(gl$forward(et)),diag(3)[,3:1],tolerance=.00001)
  gl$guess <- .3
  expect_equal(as.matrix(gl$forward(et)),guessmat(3,.3),tolerance=.00001)

})

test_that("Link K", {
  al <- CPT_Link$new(3)
  expect_equal(al$K,3)
  expect_equal(al$etWidth(),2)
  al$K <- 4
  expect_equal(al$K,4)
  expect_equal(al$etWidth(),3)

})

test_that("Link sType", {
  gl <- GaussianLink$new(3)
  expect_s3_class(gl$sType,"pos")
  expect_equal(pTypeDim(gl$sType),1)
  gl$sType <- PType("unit",2)
  expect_s3_class(gl$sType,"unit")
  expect_equal(pTypeDim(gl$sType),2)

})

test_that("Link linkScale", {
  gl <- GaussianLink$new(3)
  gl$linkScale <- 2.5
  expect_equal(as.numeric(gl$linkScale),2.5,tolerance=.00001)
  expect_error({gl$linkScale<- -1})

})

test_that("Link guess slip", {
  al <- CPT_Link$new(2)
  expect_true(is.na(al$slip))
  expect_true(is.na(al$guess))
  al$slip <- .3
  al$guess <- .2
  expect_equal(as.numeric(al$slip),.3,tolerance=.00001)
  expect_equal(as.numeric(al$guess),.2)

})


test_that("getLink setLink availableLinks",{
  expect_s3_class(getLink("Normal"),"GaussianLink")
  setLink("Normalizing",getLink("Softmax"))
  expect_s3_class(getLink("Normalizing"),"SoftmaxLink")
  expect_true("Normalizing" %in% availableLinks())

})

test_that("PotentialLink",{

  pl <- getLink("Potential")$new(3)
  ## etWidth
  expect_equal(pl$etWidth(),3)
  ## link
  et <- torch_tensor(matrix(c(1.5,3,4.5,.3,.2,.1),2,3,byrow=TRUE))
  expt <- matrix(c(1,2,3,3,2,1),2,3,byrow=TRUE)/6
  expect_equal(as.matrix(pl$forward(et)),expt[,3:1],tolerance=.00001)

})

test_that("StepProbsLink",{
  spl <- getLink("StepProbs")$new(3)
  ## etWidth
  expect_equal(spl$etWidth(),2)
  ## link
  et <- torch_tensor(matrix(c(.5,.5,.8,.25),2,2,byrow=TRUE))
  expt <- matrix(c(.5,.25,.25,.2,.6,.2),2,3,byrow=TRUE)
  expect_equal(as.matrix(spl$forward(et)),expt[,3:1],tolerance=.00001)

  spl2 <- getLink("StepProbs")$new(2)
  et2 <- torch_tensor(matrix(c(.5,.8),2,1,byrow=TRUE))
  expt2 <- matrix(c(.5,.5,.2,.8),2,2,byrow=TRUE)
  expect_equal(as.matrix(spl2$forward(et2)),expt2[,2:1],tolerance=.00001)

})

test_that("cuts2simplex",{
  tm <- torch_tensor(matrix(c(.25,.75,.3,.7),2,2,byrow=TRUE))
  expp <- matrix(c(.25,.5,.25,.3,.4,.3),2,3,byrow=TRUE)
  expect_equal(as.matrix(CPTtorch:::cuts2simplex(tm)),expp,tolerance=.00001)

})

test_that("DifferenceLink",{
  dl <- getLink("Difference")$new(3)
  expect_equal(dl$etWidth(),2)
  tm <- torch_tensor(matrix(c(.75,.25,.7,.3),2,2,byrow=TRUE))
  expp <- matrix(c(.25,.5,.25,.3,.4,.3),2,3,byrow=TRUE)
  expect_equal(as.matrix(dl$forward(tm)),expp,tolerance=.00001)

  dl2 <- getLink("Difference")$new(2)
  tm2 <- torch_tensor(matrix(c(.75,.7),2,1,byrow=TRUE))
  expp2 <- matrix(c(.75,.25,.7,.3),2,2,byrow=TRUE)
  expect_equal(as.matrix(dl2$forward(tm2)),expp2,tolerance=.00001)

})

test_that("SoftmaxLink",{
  sml <- getLink("Softmax")$new(2)
  expect_equal(sml$etWidth(),2)
  tm2 <- matrix(c(1:3,1,1.5,2),3,2)
  cpt <- sml$forward(torch_tensor(tm2))
  expect_equal(as.numeric(cpt[1,]),softmax(1.7*tm2[1,]),
               tolerance=.00001)
  expect_equal(as.numeric(cpt[2,]),softmax(1.7*tm2[2,]),
               tolerance=.00001)
  expect_equal(as.numeric(cpt[3,]),softmax(1.7*tm2[3,]),
               tolerance=.00001)

  sml$D <- torch_tensor(1.0,dtype=torch_float(),device=sml$device)
  cpt <- sml$forward(torch_tensor(tm2))
  expect_equal(as.numeric(cpt[1,]),softmax(tm2[1,]),
               tolerance=.00001)
  expect_equal(as.numeric(cpt[2,]),softmax(tm2[2,]),
               tolerance=.00001)
  expect_equal(as.numeric(cpt[3,]),softmax(tm2[3,]),
               tolerance=.00001)

})

test_that("GradedResponseLink",{
  # Test GRL against hand-computed IRT GRM under the high2low-coupled-sign
  # convention. With h2l=FALSE (default), link applies sigmoid(-D * et).
  # The natural input is et = a*theta - b with ascending b — i.e., et
  # columns decrease across cuts, which is what CompensatoryRule (or any
  # IRT-style rule) produces.
  #
  # NOTE: the previous version of this test fed contrived increasing et
  # and compared against CPTtools::gradedResponse, bypassing the cummax
  # flattening artifact that surfaces with realistic Rule output. The
  # h2l-coupled-sign change in GRL flipped this; the test now uses a
  # realistic et + hand-computed ground truth.
  grl <- getLink("GradedResponse")$new(3)
  D     <- as.numeric(grl$D)              # default 1.7
  thetas <- effectiveTheta(3)
  b     <- c(-0.5, 0.5)                   # ascending IRT thresholds
  et    <- outer(thetas, b, FUN = "-")    # et = theta - b, columns decreasing

  cpt <- as.matrix(grl$forward(torch_tensor(et)))

  # Standard GRM: P(X >= k | theta) = sigmoid(D * (theta - b_k))
  # P(X = k) = P(X >= k-1) - P(X >= k), with P(X >= 0)=1, P(X >= K)=0.
  expected <- t(sapply(thetas, function(t) {
    pge <- plogis(D * (t - b))
    c(1 - pge[1], pge[1] - pge[2], pge[2])
  }))

  expect_equal(cpt, expected, tolerance = .00001)
})

test_that("PartialCreditLink",{
  pcl <- getLink("PartialCredit")$new(3)
  et <- matrix(c(effectiveTheta(3),effectiveTheta(3)+1),3,2)
  cpt <- pcl$forward(torch_tensor(et))
  #cptt <- CPTtools::partialCredit(et[,2:1])[,3:1]
  cptt <- matrix(c(0.006535826,0.133798105,0.715733243,
                   0.0338492, 0.1337981, 0.1381985,
                   0.9596150, 0.7324038, 0.1460683),3,3)
  expect_equal(as.matrix(cpt),cptt,tolerance=.00001)
  ##TODO:  Add test for D=1.0

})

test_that("GaussianLink",{
  gl <- getLink("Gaussian")$new(3)
  et <- matrix(effectiveTheta(3),3,1)
  expect_error(gl$forward(et))

  gl$linkScale <- .5
  cpt <- gl$forward(et)
  #cptt <- CPTtools::normalLink(matrix(et,3,2),.5)[,3:1]
  cptt <- matrix(c(0.002584588,0.194493858,0.858451586,
                   0.1389638,0.6110123,0.1389638,
                   0.858451586,0.194493858,0.002584588),3,3)
  expect_equal(as.matrix(cpt),cptt[,3:1],
               tolerance=.00001)


})

test_that("SlipLink",{

})




