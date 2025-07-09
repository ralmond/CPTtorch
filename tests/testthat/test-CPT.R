test_that("CPT initialize", {

  cpt1 <- CPT_Model$new("Center","Normal",list(),c("No","Yes"))
  expect_s3_class(cpt1$rule,"CenterRule")
  expect_s3_class(cpt1$link,"GaussianLink")
  expect_length(cpt1$parentVals,0)
  expect_equal(cpt1$stateNames,c("No","Yes"))


  cpt2 <- CPT_Model$new("Center","Normal",list(),c("Yes","No"),
                        slip=.1,guess=.2,high2low=TRUE)
  expect_s3_class(cpt2$rule,"CenterRule")
  expect_s3_class(cpt2$link,"GaussianLink")
  expect_length(cpt2$parentVals,0)
  expect_equal(cpt2$stateNames,c("Yes","No"))
  expect_equal(as.numeric(cpt2$slip),.1,tolerance=.00001)
  expect_equal(as.numeric(cpt2$guess),.2,tolerance=.00001)

  cpt3 <- CPT_Model$new("Compensatory","PartialCredit",
                        list(A=c("A1","A2","A3"),B=c("B1","B2","B3")),
                        c("C1","C2","C3"))
  expect_s3_class(cpt3$rule,"CompensatoryRule")
  expect_s3_class(cpt3$link,"PartialCreditLink")
  expect_length(cpt3$parentVals,2)
  expect_equal(cpt3$parentNames,c("A","B"))
  expect_equal(cpt3$stateNames,c("C1","C2","C3"))


  cpt4 <- CPT_Model$new("Compensatory","PartialCredit",
                        list(A=c("A1","A2","A3"),B=c("B1","B2","B3")),
                        c("C1","C2","C3"),
                        QQ=matrix(c(F,T,T,T),2,2))
  expect_equal(as.matrix(cpt4$QQ),matrix(c(F,T,T,T),2,2))

})

test_that("CPT forward", {
  cpt3 <- CPT_Model$new("Compensatory","PartialCredit",
                        list(A=c("A1","A2","A3"),B=c("B1","B2","B3")),
                        c("C1","C2","C3"))
  cpt3$aMat <- matrix((1:4)/3,2,2)
  cpt3$bMat <- matrix(c(-.9,.9),2,1)
  res <- cpt3$forward()
  expect_equal(dim(res),c(9,3))

})


test_that("CPT aMat", {

  cpt3 <- CPT_Model$new("Compensatory","PartialCredit",
                        list(A=c("A1","A2","A3"),B=c("B1","B2","B3")),
                        c("C1","C2","C3"))
  expect_equal(dim(cpt3$aMat),c(2,2))
  aa <- matrix((1:4)/3,2,2)
  cpt3$aMat <- aa
  expect_equal(as.matrix(cpt3$aMat),aa,tolerance=.00001)


})

test_that("CPT bMat", {
  cpt3 <- CPT_Model$new("Compensatory","PartialCredit",
                        list(A=c("A1","A2","A3"),B=c("B1","B2","B3")),
                        c("C1","C2","C3"))
  expect_equal(dim(cpt3$bMat),c(2,1))
  bb <- matrix(c(-.9,.9),2,1)
  cpt3$bMat <- bb
  expect_equal(as.matrix(cpt3$bMat),bb,tolerance=.00001)

})

test_that("CPT linkScale", {
  cpt1 <- CPT_Model$new("Center","Normal",list(),c("No","Yes"))
  cpt1$linkScale <- .5
  expect_equal(as.numeric(cpt1$linkScale),.5,tolerance=.00001)
})


test_that("CPT slip guess", {
  cpt2 <- CPT_Model$new("Center","Normal",list(),c("Yes","No"),
                        slip=.1,guess=.2,high2low=TRUE)
  cpt2$slip <- .05
  expect_equal(as.numeric(cpt2$slip),.05,tolerance=.00001)
  cpt2$guess <- .025
  expect_equal(as.numeric(cpt2$guess),.025,tolerance=.00001)

})

test_that("CPT parentVals parentNames", {
  cpt1 <- CPT_Model$new("Center","Normal",list(),c("No","Yes"))
  expect_length(cpt1$parentNames,0)
  expect_equal(cpt1$parentVals,list())
  expect_equal(cpt1$parentStates,list())

  oldpars <- list(A=c("A1","A2","A3"),B=c("B1","B2","B3"))
  cpt3 <- CPT_Model$new("Compensatory","PartialCredit",
                        oldpars, c("C1","C2","C3"))
  expect_equal(sapply(cpt3$parentVals,sum),c(A=0,B=00),tolerance=.0001)
  expect_equal(cpt3$parentNames,c("A","B"))
  expect_equal(cpt3$parentStates,oldpars)
  newpars <- list(D=c("D1","D2"),E=c("E1","E2"),F=c("F1","F2"))
  cpt3$parentVals <- newpars
  expect_equal(cpt3$parentStates,newpars)
  expect_equal(cpt3$parentVals,as_Tvallist(newpars),tolerance=.00001)
  expect_equal(dim(cpt3$rule$pTheta),c(8,3))
  expect_equal(dim(cpt3$aMat),c(2,3))

})


test_that("CPT stateNames", {

  cpt3 <- CPT_Model$new("Compensatory","PartialCredit",
                        list(A=c("A1","A2","A3"),B=c("B1","B2","B3")),
                        c("C1","C2","C3"))
  expect_equal(dim(cpt3$aMat),c(2,2))
  expect_equal(dim(cpt3$bMat),c(2,1))
  cpt3$stateNames <- c("C1","C2")
  expect_equal(dim(cpt3$aMat),c(1,2))
  expect_equal(dim(cpt3$bMat),c(1,1))
  expect_equal(cpt3$stateNames,c("C1","C2"))

})


test_that("CPT getCPT", {
  cpt3 <- CPT_Model$new("Compensatory","PartialCredit",
                        list(A=c("A1","A2","A3"),B=c("B1","B2","B3")),
                        c("C1","C2","C3"))
  expect_false(cpt3$cptBuiltp())
  expect_equal(dim(cpt3$forward()),c(9,3))
  expect_true(cpt3$cptBuiltp())
  expect_equal(dim(cpt3$getCPT()),c(3,3,3))
})

test_that("CPT getCPTFrame", {
  cpt3 <- CPT_Model$new("Compensatory","PartialCredit",
                        list(A=c("A1","A2","A3"),B=c("B1","B2","B3")),
                        c("C1","C2","C3"))
  expect_equal(dim(cpt3$getCPTframe()),c(9,5))

})

test_that("CPT getETFrame", {
  cpt3 <- CPT_Model$new("Compensatory","PartialCredit",
                        list(A=c("A1","A2","A3"),B=c("B1","B2","B3")),
                        c("C1","C2","C3"))
  expect_equal(dim(cpt3$getETframe()),c(9,4))

})




test_that("CPT deviance", {

})

test_that("CPT AIC", {

})


test_that("CPT buildOptimizer", {

})


test_that("CPT set", {

})


test_that("CPT QQ", {

})

test_that("CPT high2low", {

})




