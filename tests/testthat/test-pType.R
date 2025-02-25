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
