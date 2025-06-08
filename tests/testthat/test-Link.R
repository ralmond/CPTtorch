test_that("Link initialize", {

  al <- CPT_Link$new(2,guess=.2,slip=.3,high2low=TRUE)
  expect_s3_class(al,"CPT_Link")
  expect_equal(CPT_Link$K,2)
  expect_equal(CPT_Link$slip,.3)
  expect_equal(CPT_Link$guess,.2)
  expect_equal(CPT_Link$high2low,TRUE)

})

test_that("Link leakMat", {
  al <- CPT_Link$new(3)
  expect_null(sl$leakmat())
  al <- CPT_Link$new(3,guess=.3)
  
  

})

test_that("Link forward", {

})

test_that("Link K", {

})

test_that("Link sType", {

})

test_that("Link linkScale", {

})

test_that("Link guess", {

})

test_that("Link slip", {

})

test_that("getLink",{

})
test_that("setLink",{

})
test_that("availableLinks",{

})

test_that("PotentialLink",{

})

test_that("StepProbsLink",{

})

test_that("cuts2simplex",{

})

test_that("DifferenceLink",{

})

test_that("SoftmaxLink",{

})

test_that("GradedResponseLink",{

})

test_that("PartialCreditLink",{

})

test_that("GaussianLink",{

})

test_that("addcolk",{

})

test_that("SlipLink",{

})




