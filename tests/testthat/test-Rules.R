
test_that("effectiveTheta 10",{
  ef1 <- effectiveTheta(3)
  ef2 <- effectiveTheta(3,TRUE)
  expect_equal(ef1,-ef2,tolerance=.0001)
  tf1 <- effectiveTheta10(3)
  tf2 <- effectiveTheta10(3,TRUE)
  expect_equal(as.numeric(tf1),-as.numeric(ef2),tolerance=.0001)

})


test_that("as_Tvallist",{
  ## No parents

  tvl0 <- as_Tvallist(c())
  expect_length(tvl0,0)
  expect_true(is.list(tvl0))

  ## 2 parents no names

  tvl20 <- as_Tvallist(c(3,4))
  expect_length(tvl20,2)
  expect_equal(names(tvl20),paste0("P",1:2))
  expect_equal(names(tvl20[[1]]),paste0("S",1:3))
  expect_equal(names(tvl20[[2]]),paste0("S",1:4))


  ## 2 parents P names
  tvl20p <- as_Tvallist(c(A=3,B=4))
  expect_length(tvl20p,2)
  expect_equal(names(tvl20p),c("A","B"))
  expect_equal(names(tvl20p[[1]]),paste0("S",1:3))
  expect_equal(names(tvl20p[[2]]),paste0("S",1:4))

  ## 2 parents P names, S vals
  tvl20psv <- as_Tvallist(list(A=1:3,B=1:4))
  expect_length(tvl20psv,2)
  expect_equal(names(tvl20psv),c("A","B"))
  expect_equal(names(tvl20psv[[1]]),paste0("S",1:3))
  expect_equal(names(tvl20psv[[2]]),paste0("S",1:4))
  expect_equal(sum(tvl20psv[[1]]),6)
  expect_equal(sum(tvl20psv[[2]]),10)

  ## 2 parents S names
  tvl20sv <- as_Tvallist(list(c(A=1,B=2,C=3),c(X=1,Y=2,Z=3)))
  expect_length(tvl20sv,2)
  expect_equal(names(tvl20sv),c("P1","P2"))
  expect_equal(names(tvl20sv[[1]]),c("A","B","C"))
  expect_equal(names(tvl20sv[[2]]),c("X","Y","Z"))
  expect_equal(sum(tvl20sv[[1]]),6)
  expect_equal(sum(tvl20sv[[2]]),6)

  ## 2 parents P, S names
  tvl20ps <- as_Tvallist(list(S1=c(A=1,B=2,C=3),S2=c(X=1,Y=2,Z=3)))
  expect_length(tvl20ps,2)
  expect_equal(names(tvl20ps),c("S1","S2"))
  expect_equal(names(tvl20ps[[1]]),c("A","B","C"))
  expect_equal(names(tvl20ps[[2]]),c("X","Y","Z"))
  expect_equal(sum(tvl20ps[[1]]),6)
  expect_equal(sum(tvl20ps[[2]]),6)

  ## 2 parents low2high
  tvl2lh <- as_Tvallist(c(A=3,B=4))
  expect_lt(tvl2lh[[1]][1],tvl2lh[[1]][3])
  expect_lt(tvl2lh[[2]][1],tvl2lh[[2]][4])

    ## 2 parents high2low
  tvl2hl <- as_Tvallist(c(A=3,B=4),high2low=TRUE)
  expect_gt(tvl2hl[[1]][1],tvl2hl[[1]][3])
  expect_gt(tvl2hl[[2]][1],tvl2hl[[2]][4])

  ## 2 parents mixed
  tvl2hlm <- as_Tvallist(c(A=3,B=4),high2low=c(TRUE,FALSE))
  expect_gt(tvl2hlm[[1]][1],tvl2hlm[[1]][3])
  expect_lt(tvl2hlm[[2]][1],tvl2hlm[[2]][4])

})

test_that("buildpTheta10",{

  tt0 <- buildpTheta10(list())
  expect_equal(as.numeric(tt0),0)

  tt1 <- buildpTheta10(as_Tvallist(3))
  expect_equal(dim(tt1),c(3,1))

  tt2 <- buildpTheta10(as_Tvallist(c(3,4)))
  expect_equal(dim(tt2),c(12,2))

})


test_that("Rule setDim",{
  cr <- getRule("Compensatory")$new(as_Tvallist(c(3,4)),3)
  cr$setDim(S=12,J=2,K=3)
  expect_equal(pTypeDim(cr$aType),c(3,2))
  expect_equal(dim(cr$aMat),c(3,2))
  expect_equal(pTypeDim(cr$bType),c(3,1))
  expect_equal(dim(cr$bMat),c(3,1))
})


test_that("Rule setParents",{
  cr <- getRule("Compensatory")$new(as_Tvallist(c(2,2)),3)
  cr$setParents(as_Tvallist(c(3,4)))
  expect_equal(dim(cr$pTheta),c(12,2))
  expect_equal(pTypeDim(cr$aType),c(3,2))
  expect_equal(pTypeDim(cr$bType),c(3,1))
  expect_equal(names(cr$pNames),c("P1","P2"))
  expect_equal(cr$pNames[[1]],paste0("S",1:3))
  expect_equal(cr$pNames[[2]],paste0("S",1:4))

})


test_that("Rule initialize",{
  cr <- getRule("Compensatory")$new(as_Tvallist(c(2,2)),3)
  expect_equal(dim(cr$pTheta),c(4,2))
  expect_equal(pTypeDim(cr$aType),c(3,2))
  expect_equal(pTypeDim(cr$bType),c(3,1))
  expect_equal(names(cr$pNames),c("P1","P2"))
  expect_equal(cr$high2low,FALSE)
  expect_equal(cr$QQ,TRUE)

  cr1 <- getRule("Compensatory")$new(as_Tvallist(c(2,2)),3,
                                     QQ=matrix(c(TRUE,FALSE,TRUE,FALSE,TRUE,TRUE),3,2),
                                     high2low=TRUE)
  expect_equal(cr1$high2low,TRUE)
  expect_equal(dim(cr1$QQ),c(3,2))
})


test_that("Rule aType",{
  cr <- getRule("Compensatory")$new(as_Tvallist(c(3,4)),3)
  expect_s3_class(cr$aType,"pos")
  expect_equal(as.numeric(cr$aMat[1,1]),1)
  cr$aType <- PType("real",c(K,J))
  expect_equal(pTypeDim(cr$aType),c(3,2))
  expect_equal(dim(cr$aMat),c(3,2))
  expect_equal(as.numeric(cr$aMat[1,1]),0)


})

test_that("Rule bType",{
  cr <- getRule("Compensatory")$new(as_Tvallist(c(3,4)),3)
  expect_s3_class(cr$bType,"real")
  expect_equal(as.numeric(cr$bMat[1,1]),0)

  cr$bType <- PType("pos",c(K,1))
  expect_equal(pTypeDim(cr$bType),c(3,1))
  expect_equal(dim(cr$bMat),c(3,1))
  expect_equal(as.numeric(cr$bMat[1,1]),1)

})

test_that("Rule aMat",{
  cr <- getRule("Compensatory")$new(as_Tvallist(c(3,4)),3)

  expect_equal(as.numeric(cr$aVec),rep(0,6),tolerance=.00001)

  av <- (1:6)/10

  cr$aMat <- torch_tensor(matrix(av,3,2,byrow=TRUE))

  expect_equal(as.numeric(cr$aVec),log(av),tolerance=.001)

})

test_that("Rule bMat",{

  cr <- getRule("Compensatory")$new(as_Tvallist(c(3,4)),3)

  expect_equal(as.numeric(cr$bVec),rep(0,3),tolerance=.00001)

  bv <- 1:3

  cr$bMat <- torch_tensor(matrix(bv,3,1,byrow=TRUE),dtype=torch_float())

  expect_equal(as.numeric(cr$bVec),bv,tolerance=.00001)

})

test_that("Rule et",{
  cr <- getRule("Compensatory")$new(as_Tvallist(c(3,4)),2)

  et <- cr$et
  expect_equal(dim(et),c(3,4,2))


})

test_that("Rule et_p",{
  cr <- getRule("Compensatory")$new(as_Tvallist(c(3,4)),2)
  expect_false(cr$et_p)

  cr$et
  expect_true(cr$et_p)

  cr$et_p <- FALSE
  expect_false(cr$et_p)

  cr$et
  cr$aMat <- torch_tensor(matrix(.5,2,2))
  expect_false(cr$et_p)

  cr$et
  cr$bMat <- torch_tensor(matrix(.5,2,1))
  expect_false(cr$et_p)

  cr$et
  cr$high2low <- TRUE
  expect_false(cr$et_p)

})


test_that("Rule getETframe",{
  cr <- getRule("Compensatory")$new(as_Tvallist(c(3,4)),2)

  etf <- cr$getETframe()
  expect_equal(dim(etf),c(12,4))
  expect_equal(names(etf),c("P1","P2","et.1","et.2"))

})

test_that("Rule high2low",{

})



test_that("RuleASB",{

})

test_that("RuleASB QQ",{

})

test_that("RuleASB forward",{

})

test_that("RuleBSA",{

})

test_that("RuleBSA forward",{

})

test_that("RuleBSA QQ",{

})

test_that("RuleBAS forward",{

})

test_that("RuleBAS QQ",{

})


test_that("RuleConstB aTypeMat",{

})

test_that("RuleConstB QQ",{

})

test_that("RuleConstB forward",{

})

test_that("RuleConstA bTypeMat",{

})

test_that("RuleConstA QQ",{

})

test_that("RuleConstA forward",{

})


test_that("CompensatoryRule",{

})

test_that("CompensatoryGRRule",{

})

test_that("ConjunctiveRule",{

})

test_that("DisjunctiveRule",{

})

test_that("NoisyAndRule",{

})

test_that("NoisyOrRule",{

})

test_that("CenterRule",{

})

test_that("DirichletRule",{

})

test_that("getRule",{

})

test_that("setRule",{

})

test_that("availableRule",{

})




