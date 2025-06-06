test_that("pMat2pVec", {

  pt <- setpTypeDim(PType("pos",dim=c(2,3)))
  natpar <- matrix(1:6,2,3)
  qq <- matrix(c(FALSE,TRUE,TRUE,TRUE,TRUE,FALSE),2,3,byrow=TRUE)

  expect_equal(pMat2pVec(pt,natpar),1:6)

  whichUsed(pt) <- qq
  expect_equal(pMat2pVec(pt,natpar),2:5)
})

test_that("pMat2rowlist", {

  pt <- setpTypeDim(PType("pos",dim=c(2,3)))
  natpar <- matrix(1:6,2,3)
  qq <- matrix(c(FALSE,TRUE,TRUE,TRUE,TRUE,FALSE),2,3,byrow=TRUE)

  expect_equal(pMat2rowlist(pt,natpar),list(c(1,3,5),c(2,4,6)))

  whichUsed(pt) <- qq
  expect_equal(pMat2rowlist(pt,natpar),list(c(3,5),c(2,4)))
})


test_that("pMat2collist", {

  pt <- setpTypeDim(PType("pos",dim=c(2,3)))
  natpar <- matrix(1:6,2,3)
  qq <- matrix(c(FALSE,TRUE,TRUE,TRUE,TRUE,FALSE),2,3,byrow=TRUE)

  expect_equal(pMat2collist(pt,natpar),list(1:2,3:4,5:6))

  whichUsed(pt) <- qq
  expect_equal(pMat2collist(pt,natpar),list(2,3:4,5))
})

test_that("list2vec", {
  expect_equal(list2vec(list(2,3:4,5)),2:5)
})


test_that("pVec2pMat", {

  pt <- setpTypeDim(PType("pos",dim=c(2,3)))

  natpar <- matrix(1:6,2,3)
  qq <- matrix(c(FALSE,TRUE,TRUE,TRUE,TRUE,FALSE),2,3,byrow=TRUE)

  expect_equal(pVec2pMat(pt,1:6),natpar)

  whichUsed(pt) <- qq
  is.na(natpar) <- !qq

  expect_equal(pVec2pMat(pt,2:5),natpar)
})

test_that("vec2rowlist", {

  pt <- setpTypeDim(PType("pos",dim=c(2,3)))
  natpar <- matrix(1:6,2,3)
  qq <- matrix(c(FALSE,TRUE,TRUE,TRUE,TRUE,FALSE),2,3,byrow=TRUE)

  expect_equal(vec2rowlist(pt,c(1,3,5,2,4,6)),list(c(1,3,5),c(2,4,6)))

  whichUsed(pt) <- qq
  expect_equal(vec2rowlist(pt,c(3,5,2,4)),list(c(3,5),c(2,4)))
})


test_that("vec2collist", {

  pt <- setpTypeDim(PType("pos",dim=c(2,3)))
  natpar <- matrix(1:6,2,3)
  qq <- matrix(c(FALSE,TRUE,TRUE,TRUE,TRUE,FALSE),2,3,byrow=TRUE)

  expect_equal(vec2collist(pt,1:6),list(1:2,3:4,5:6))

  whichUsed(pt) <- qq
  expect_equal(vec2collist(pt,2:5),list(2,3:4,5))
})


test_that("PType isPType",{
  expect_true(isPType(PType("real")))
  expect_false(isPType(PType("undefined")))
})

test_that("PType availablePtypes",{
  expect_true(all(c("real","pos","unit") %in% availablePTypes()))
  expect_false("rubbish" %in% availablePTypes())
})

test_that("PType pTypeDim setpTypeDim",{
  pt <- PType("real")
  expect_null(pTypeDim(pt))
  pt <- setpTypeDim(pt,K=4,J=3)
  expect_equal(pTypeDim(pt),c(4,3))

})


test_that("PType whichUsed",{
  pt <- PType("real")
  pt <- setpTypeDim(pt,K=3,J=2)
  expect_equal(whichUsed(pt),TRUE)
  qq <- matrix(c(T,F,T,F,T,T),3,2)
  whichUsed(pt) <- qq
  expect_equal(pt$used,qq)
})


## real

test_that("checkParam real",{
  pt <- setpTypeDim(PType("real"),K=3,J=2)
  af <- matrix(rnorm(6),3,2)
  expect_true(isTRUE(checkParam(pt,af)))
  af[2,2] <- NA
  expect_false(isTRUE(checkParam(pt,af)))
})

test_that("natpar2Rvec Rvec2natpar real",{
  pt <- setpTypeDim(PType("real"),K=3,J=2)
  pv <- (1:6)+.1
  pm <- matrix(pv,3,2)
  expect_equal(natpar2Rvec(pt,pm),pv)
  expect_equal(Rvec2natpar(pt,pv),pm)
})

test_that("natpar2tvec tvec2natpar real",{
  pt <- setpTypeDim(PType("real"),K=3,J=2)
  pv <- (1:6)+.1
  pm <- matrix(pv,3,2,byrow=TRUE)
  expect_equal(as.numeric(natpar2tvec(pt,torch_tensor(pm))),pv,
               tolerance=.00001)
  expect_equal(as.matrix(tvec2natpar(pt,torch_tensor(pv))),pm,
               tolerance=.00001)
})

test_that("getZero defaultParameter real",{
  pt <- PType("real")
  expect_equal(getZero(pt),0)
  expect_equal(defaultParameter(pt),0)

  pt <- setpTypeDim(pt,K=3,J=2)
  expect_equal(defaultParameter(pt),matrix(0,3,2))
})


## pos

test_that("checkParam pos",{
  pt <- setpTypeDim(PType("pos"),K=3,J=2)
  af <- matrix(1:6+.1,3,2)
  expect_true(isTRUE(checkParam(pt,af)))
  af[2,2] <- -3
  expect_false(isTRUE(checkParam(pt,af)))
  af[2,2] <- NA
  expect_false(isTRUE(checkParam(pt,af)))
})

test_that("natpar2Rvec Rvec2natpar pos",{
  pt <- setpTypeDim(PType("pos"),K=3,J=2)
  pv <- (-2:3)+.1
  pm <- matrix(exp(pv),3,2)
  expect_equal(natpar2Rvec(pt,pm),pv)
  expect_equal(Rvec2natpar(pt,pv),pm)
})

test_that("natpar2tvec tvec2natpar pos",{
  pt <- setpTypeDim(PType("pos"),K=3,J=2)
  pv <- (-2:3)+.1
  pm <- matrix(exp(pv),3,2,byrow=TRUE)
  expect_equal(as.numeric(natpar2tvec(pt,torch_tensor(pm))),pv,
               tolerance=.000001)
  expect_equal(as.matrix(tvec2natpar(pt,torch_tensor(pv))),pm,
               tolerance=.000001)
})

test_that("getZero defaultParameter pos",{
  pt <- PType("pos")
  expect_equal(getZero(pt),1)
  expect_equal(defaultParameter(pt),1)

  pt <- setpTypeDim(pt,K=3,J=2)
  expect_equal(defaultParameter(pt),matrix(1,3,2))
})

## unit

test_that("checkParam unit",{
  pt <- setpTypeDim(PType("unit"),K=3,J=2)
  af <- matrix((1:6)/10,3,2)
  expect_true(isTRUE(checkParam(pt,af)))
  af[2,2] <- -3
  expect_false(isTRUE(checkParam(pt,af)))
  af[2,2] <- 3
  expect_false(isTRUE(checkParam(pt,af)))
  af[2,2] <- NA
  expect_false(isTRUE(checkParam(pt,af)))
})

test_that("natpar2Rvec Rvec2natpar unit",{
  pt <- setpTypeDim(PType("unit"),K=3,J=2)
  pv <- (1:6)/10
  pm <- matrix(invlogit(pv),3,2)
  expect_equal(natpar2Rvec(pt,pm),pv)
  expect_equal(Rvec2natpar(pt,pv),pm)
})

test_that("natpar2tvec tvec2natpar unit",{
  pt <- setpTypeDim(PType("unit"),K=3,J=2)
  pv <- (1:6)/10
  pm <- matrix(invlogit(pv),3,2,byrow=TRUE)
  expect_equal(as.numeric(natpar2tvec(pt,torch_tensor(pm))),pv,
               tolerance=.000001)
  expect_equal(as.matrix(tvec2natpar(pt,torch_tensor(pv))),pm,
               tolerance=.000001)
})

test_that("getZero defaultParameter unit",{
  pt <- PType("unit")
  expect_equal(getZero(pt),.5)
  expect_equal(defaultParameter(pt),.5)

  pt <- setpTypeDim(pt,K=3,J=2)
  expect_equal(defaultParameter(pt),matrix(.5,3,2))
})


## pVec

test_that("checkParam pVec",{
  pt <- setpTypeDim(PType("pVec",dim=c(K)),K=3,J=1)
  af <- c(4,5,6)/15
  expect_false(isTRUE(checkParam(pt,af)))
  dim(af) <- 3
  expect_true(isTRUE(checkParam(pt,af)))
  af[2] <- .9
  expect_false(isTRUE(checkParam(pt,af)))
  af[2] <- -3
  expect_false(isTRUE(checkParam(pt,af)))
  af[2] <- 3
  expect_false(isTRUE(checkParam(pt,af)))
  af[2] <- NA
  expect_false(isTRUE(checkParam(pt,af)))
})

test_that("natpar2Rvec Rvec2natpar pVec",{
  pt <- setpTypeDim(PType("pVec",dim=c(K)),K=3,J=2)
  pv1 <- log(4:6)
  pm <- (4:6)/15
  pv2 <- log(pm)
  dim(pm) <- length(pm)
  expect_equal(natpar2Rvec(pt,pm),pv2)
  expect_equal(Rvec2natpar(pt,pv1),pm)
})

test_that("natpar2tvec tvec2natpar pVec",{
  pt <- setpTypeDim(PType("pVec",dim=c(K)),K=3,J=2)
  pv1 <- log(4:6)
  pm <- (4:6)/15
  pv2 <- log(pm)
  expect_equal(as.numeric(natpar2tvec(pt,torch_tensor(pm))),pv2,
               tolerance=.00001)
  expect_equal(as.numeric(tvec2natpar(pt,torch_tensor(pv1))),pm,
               tolerance=.00001)
})

test_that("getZero defaultParameter pVec",{
  pt <- PType("pVec")
  expect_equal(getZero(pt),.5)
  expect_equal(defaultParameter(pt),.5)

  pt <- setpTypeDim(pt,K=3,J=2)
  expect_equal(defaultParameter(pt),matrix(.5,3,2))
})

## cpMat

test_that("checkParam cpMat",{
  pt <- setpTypeDim(PType("cpMat",dim=c(S,K)),K=3,S=2)
  af <- rbind(4:6,6:4)/15
  expect_true(isTRUE(checkParam(pt,af)))
  af[2,2] <- .9
  expect_false(isTRUE(checkParam(pt,af)))
  af[2,2] <- -3
  expect_false(isTRUE(checkParam(pt,af)))
  af[2,2] <- 3
  expect_false(isTRUE(checkParam(pt,af)))
  af[2,2] <- NA
  expect_false(isTRUE(checkParam(pt,af)))
})

test_that("natpar2Rvec Rvec2natpar cpMat",{
  pt <- setpTypeDim(PType("cpMat",dim=c(S,K)),K=3,S=2)
  pv1 <- log(c(4:6,6:4))
  pm <- rbind((4:6)/15,(6:4)/15)
  pv2 <- pv1 - log(15)
  expect_equal(natpar2Rvec(pt,pm),pv2)
  expect_equal(Rvec2natpar(pt,pv1),pm)

  whichUsed(pt) <- matrix(TRUE,2,3)
  whichUsed(pt)[2,2]<-FALSE
  pv1a <- log(c(4:6,7.5,NA,7.5))
  pma <- rbind((4:6)/15,c(7.5,NA,7.5)/15)
  pv2a <- pv1a - log(15)
  expect_equal(natpar2Rvec(pt,pma),pv2a[-5])
  expect_equal(Rvec2natpar(pt,pv1a[-5]),pma)
})

test_that("natpar2tvec tvec2natpar cpMat",{
  pt <- setpTypeDim(PType("cpMat",dim=c(S,K)),K=3,S=2)
  pv1 <- log(c(4:6,6:4))
  pm <- rbind((4:6)/15,(6:4)/15)
  pv2 <- pv1 - log(15)
  expect_equal(as.numeric(natpar2tvec(pt,torch_tensor(pm))),pv2,
               tolerance=.00001)
  expect_equal(as.matrix(tvec2natpar(pt,torch_tensor(pv1))),pm,
               tolerance=.00001)

  whichUsed(pt) <- matrix(TRUE,2,3)
  whichUsed(pt)[2,2]<-FALSE
  pv1a <- log(c(4:6,7.5,NA,7.5))
  pma <- rbind((4:6)/15,c(7.5,NA,7.5)/15)
  pv2a <- pv1a - log(15)
  expect_equal(as.numeric(natpar2tvec(pt,torch_tensor(pma))),pv2a[-5],
               tolerance=.00001)
  pmaa <- as.matrix(tvec2natpar(pt,torch_tensor(pv1a[-5])))
  pma[2,2]<- -999
  pmaa[2,2]<- -999
  expect_equal(pmaa,pma,tolerance=.00001)

})

test_that("getZero defaultParameter cpMat",{
  pt <- PType("cpMat",dim=c(S,K))
  pt <- setpTypeDim(pt,K=4,S=2)
  expect_equal(getZero(pt),.25)
  expect_equal(defaultParameter(pt),matrix(.25,2,4))
})


## const

test_that("checkParam const",{
  pt <- PType("const")
  af <- matrix((1:6)/10,3,2)
  expect_warning(result <- isTRUE(checkParam(pt,af)))
  expect_true(result)
  pt <- setpTypeDim(pt,K=3,J=2)
  expect_true(isTRUE(checkParam(pt,af)))
  expect_false(isTRUE(checkParam(pt,t(af))))
})

test_that("natpar2Rvec Rvec2natpar const",{
  pt <- setpTypeDim(PType("const"),K=3,J=2)
  pv <- (1:6)/10
  pm <- matrix(invlogit(pv),3,2)
  expect_length(natpar2Rvec(pt,pm),0)
  expect_length(Rvec2natpar(pt,pv),0)
})

test_that("natpar2tvec tvec2natpar const",{
  pt <- setpTypeDim(PType("const"),K=3,J=2)
  pv <- (1:6)/10
  pm <- matrix(invlogit(pv),3,2)
  expect_length(as.numeric(natpar2tvec(pt,torch_tensor(pm))),0)
  expect_null(tvec2natpar(pt,torch_tensor(pv)))
})

test_that("getZero defaultParameter const",{
  pt <- PType("const")
  expect_true(is.na(getZero(pt)))
  expect_true(is.na(defaultParameter(pt)))

  pt <- setpTypeDim(pt,K=3,J=2)
  expect_equal(defaultParameter(pt),matrix(NA,3,2))
})


## incrK

test_that("checkParam incrK",{
  pt <- setpTypeDim(PType("incrK",dim=c(K,J)),K=3,J=2)
  pmi <- matrix(1:6,3,2)
  pmd <- matrix(6:1,3,2)
  expect_true(isTRUE(checkParam(pt,pmi)))
  expect_false(isTRUE(checkParam(pt,pmd)))

  pt$high2low <- TRUE
  expect_true(isTRUE(checkParam(pt,pmd)))
  expect_false(isTRUE(checkParam(pt,pmi)))
})

test_that("natpar2Rvec Rvec2natpar incrK",{
  pt <- setpTypeDim(PType("incrK",dim=c(K,J)),K=3,J=2)
  pmi <- matrix(1:6,3,2)
  pvi <- c(1,0,0,4,0,0)

  expect_equal(natpar2Rvec(pt,pmi),pvi)
  expect_equal(Rvec2natpar(pt,pvi),pmi)

  pt$high2low <- TRUE
  pmd <- matrix(6:1,3,2)
  pvd <- c(4,0,0,1,0,0)

  expect_equal(natpar2Rvec(pt,pmd),pvd)
  expect_equal(Rvec2natpar(pt,pvd),pmd)

})

test_that("natpar2tvec tvec2natpar incrK",{
  pt <- setpTypeDim(PType("incrK",dim=c(K,J)),K=3,J=2)
  pmi <- matrix((1:6)+.1,3,2)
  pvi <- c(1.1,0,0,4.1,0,0)

  expect_equal(as.numeric(natpar2tvec(pt,torch_tensor(pmi))),pvi,
               tolerance=.000001)
  expect_equal(as.matrix(tvec2natpar(pt,torch_tensor(pvi))),pmi,
               tolerance=.000001)

  pt$high2low <- TRUE
  pmd <- matrix(6:1+.1,3,2)
  pvd <- c(4.1,0,0,1.1,0,0)

  expect_equal(as.numeric(natpar2tvec(pt,torch_tensor(pmd))),pvd,
               tolerance=.00001)
  expect_equal(as.matrix(tvec2natpar(pt,torch_tensor(pvd))),pmd,
               tolerance=.00001)

})

test_that("getZero defaultParameter incrK",{
  pt <- setpTypeDim(PType("incrK",dim=c(K,J)),K=3,J=2)
  expect_true(is.na(getZero(pt)))
})


## Note:  The interaction of the inner Q-matrix and these functions is
## not fully tested.


