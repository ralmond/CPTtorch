test_that("pMat2pVec", {

  pt <- setpTypeDim(PType("pos",dim=c(2,3)))

  natpar <- matrix(1:6,2,3)
  qq <- matrix(c(FALSE,TRUE,TRUE,TRUE,TRUE,FALSE),2,3,byrow=TRUE)

  expect_equal(pMat2pVec(pt,natpar),1:6)

  whichUsed(pt) <- qq

  expect_equal(pMat2pVec(pt,natpar),2:5)

}

test_that("pMat2rowlist", {

  pt <- setpTypeDim(PType("pos",dim=c(2,3)))

  natpar <- matrix(1:6,2,3)
  qq <- matrix(c(FALSE,TRUE,TRUE,TRUE,TRUE,FALSE),2,3,byrow=TRUE)

  expect_equal(pMat2rowlist(pt,natpar),list(c(1,3,5),c(2,4,6)))

  whichUsed(pt) <- qq

  expect_equal(pMat2rowlist(pt,natpar),list(c(3,5),c(2,4)))
               
}


test_that("pMat2collist", {

  pt <- setpTypeDim(PType("pos",dim=c(2,3)))

  natpar <- matrix(1:6,2,3)
  qq <- matrix(c(FALSE,TRUE,TRUE,TRUE,TRUE,FALSE),2,3,byrow=TRUE)

  expect_equal(pMat2collist(pt,natpar),list(1:2,3:4,5:6))

  whichUsed(pt) <- qq

  expect_equal(pMat2rowlist(pt,natpar),list(2,3:4,5))

}

test_that("list2vec", {
  expect_equal(list2vec(list(2,3:4,5)),2:5)
}


test_that("pVec2pMat", {

  pt <- setpTypeDim(PType("pos",dim=c(2,3)))

  natpar <- matrix(1:6,2,3)
  qq <- matrix(c(FALSE,TRUE,TRUE,TRUE,TRUE,FALSE),2,3,byrow=TRUE)

  expect_equal(pVec2pMat(pt,1:6),natpar)

  whichUsed(pt) <- qq
  is.na(natpar) <- !qq

  expect_equal(pMat2pVec(pt,2:5),natpar)

}

test_that("vec2rowlist", {

  pt <- setpTypeDim(PType("pos",dim=c(2,3)))

  natpar <- matrix(1:6,2,3)
  qq <- matrix(c(FALSE,TRUE,TRUE,TRUE,TRUE,FALSE),2,3,byrow=TRUE)

  expect_equal(vec2rowlist(pt,c(1,3,5,2,4,6)),list(c(1,3,5),c(2,4,6)))

  whichUsed(pt) <- qq

  expect_equal(vec2rowlist(pt,c(3,5,2,4)),list(c(3,5),c(2,4)))
               
}


test_that("vec2collist", {

  pt <- setpTypeDim(PType("pos",dim=c(2,3)))

  natpar <- matrix(1:6,2,3)
  qq <- matrix(c(FALSE,TRUE,TRUE,TRUE,TRUE,FALSE),2,3,byrow=TRUE)

  expect_equal(vec2collist(pt,1:6),list(1:2,3:4,5:6))

  whichUsed(pt) <- qq

  expect_equal(vec2rowlist(pt,2:5),list(2,3:4,5))

}


test_that("PType isPType",{

})

test_that("PType availablePtypes",{

})

test_that("PType pTypeDim",{

})

test_that("PType setpTypeDim",{

})

test_that("PType whichUsed",{

})


## real

test_that("checkParam real",{

})

test_that("natpar2Rvec real",{

})

test_that("Rvec2natpar real",{

})

test_that("natpar2tvec real",{

})

test_that("tvec2natpar real",{

})

test_that("getZero real",{

})

test_that("getDefaultParameter real",{

})

## pos

test_that("checkParam pos",{

})

test_that("natpar2Rvec pos",{

})

test_that("Rvec2natpar pos",{

})

test_that("natpar2tvec pos",{

})

test_that("tvec2natpar pos",{

})

test_that("getZero pos",{

})

test_that("getDefaultParameter pos",{

})

## unit

test_that("checkParam unit",{

})

test_that("natpar2Rvec unit",{

})

test_that("Rvec2natpar unit",{

})

test_that("natpar2tvec unit",{

})

test_that("tvec2natpar unit",{

})

test_that("getZero unit",{

})

test_that("getDefaultParameter unit",{

})


## pVec

test_that("checkParam pVec",{

})

test_that("natpar2Rvec pVec",{

})

test_that("Rvec2natpar pVec",{

})

test_that("natpar2tvec pVec",{

})

test_that("tvec2natpar pVec",{

})

test_that("getZero pVec",{

})

test_that("getDefaultParameter pVec",{

})

## cpMat

test_that("checkParam cpMat",{

})

test_that("natpar2Rvec cpMat",{

})

test_that("Rvec2natpar cpMat",{

})

test_that("natpar2tvec cpMat",{

})

test_that("tvec2natpar cpMat",{

})

test_that("getZero cpMat",{

})

test_that("getDefaultParameter cpMat",{

})


## const

test_that("checkParam const",{

})

test_that("natpar2Rvec const",{

})

test_that("Rvec2natpar const",{

})

test_that("natpar2tvec const",{

})

test_that("tvec2natpar const",{

})

test_that("getZero const",{

})

test_that("getDefaultParameter const",{

})

## incrK

test_that("checkParam incrK",{

})

test_that("natpar2Rvec incrK",{

})

test_that("Rvec2natpar incrK",{

})

test_that("natpar2tvec incrK",{

})

test_that("tvec2natpar incrK",{

})

test_that("getZero incrK",{

})

test_that("getDefaultParameter incrK",{

})




