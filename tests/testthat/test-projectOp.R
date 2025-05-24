### projectOp

test_that("projectOp num num",{
  expect_equal(projectOp(1.5,1.5),3.0)
  expect_equal(projectOp(1.5,3.0,"/"),.5)

})

test_that("projectOp arr num",{
  mat <- matrix((1:6)+.1,3,2)
  arr <- array((1:12)+.1,c(2,3,4))
  expect_equal(projectOp(mat,.1),matrix((1:6)+.2,3,2))
  expect_equal(projectOp(arr,.2,"-"),array((1:12)-.1,c(2,3,4)))
  expect_equal(projectOp(.1,mat),matrix((1:6)+.2,3,2))
  expect_equal(projectOp(.2,arr,"-"),-array((1:12)-.1,c(2,3,4)))

})


test_that("projectOp arr arr",{
  mat <- matrix((1:6)+.1,3,2)
  col <- matrix(1:2,2,1)
  row <- matrix(1:3,1,3)
  summ <- matrix(c(2,3,3,4,4,5),2,3)
  prodd <- matrix(c(1,2,2,4,3,6),2,3)

  expect_equal(projectOp(mat,mat,"-"),matrix(0,3,2))
  expect_equal(projectOp(row,col),summ)
  expect_equal(projectOp(col,row,"*"),prodd)

})

test_that("projectOp ten ten",{
  mat <- torch_tensor(matrix((1:6)+.1,3,2))
  col <- torch_tensor(matrix(1:2,2,1))
  row <- torch_tensor(matrix(1:3,1,3))
  summ <- matrix(c(2,3,3,4,4,5),2,3)
  prodd <- matrix(c(1,2,2,4,3,6),2,3)

  expect_equal(as.matrix(projectOp(mat,mat,"-")),matrix(0,3,2))
  expect_equal(as.matrix(projectOp(row,col)),summ)
  expect_equal(as.matrix(projectOp(col,row,"*")),prodd)

})

### Marginalize  marginalize(pot,dim=1,op="sum")

test_that("marginalize arr",{
  arr <- array((1:24)+.1,c(4,3,2))
  expect_equal(marginalize(arr),
               matrix(c(sum((1:4)+.1), sum((5:8)+.1), sum((9:12)+.1),
                        sum((13:16)+.1), sum((17:20)+.1), sum((21:24)+.1)),
                      3,2))
  expect_equal(marginalize(arr,2,"max"),
               matrix(c(9:12,21:24)+.1,4,2))
  expect_equal(marginalize(arr,c(1,2),"max"),c(12.1,24.1))


})

test_that("marginalize ten",{
  arr <- torch_tensor(array((1:24)+.1,c(4,3,2)))
  expect_equal(as.matrix(marginalize(arr)),
               matrix(c(sum((1:4)+.1), sum((5:8)+.1), sum((9:12)+.1),
                        sum((13:16)+.1), sum((17:20)+.1), sum((21:24)+.1)),
                      3,2),
               tolerance=.00001)
  expect_equal(as.matrix(marginalize(arr,2,"max")),
               matrix(c(9:12,21:24)+.1,4,2),
               tolerance=.00001)
  expect_equal(as.numeric(marginalize(arr,c(1,2),"max")),c(12.1,24.1),
               tolerance=.00001)

})

### Generalized Matrix Multiplication genMMt(m1,m2,combOp,summaryOp)

test_that("genMMt matrix", {
  et <- cbind(rep(c(-1,0,1),3),rep(c(-1,0,1),each=3))
  a1 <- matrix(c(1,2),1,2)
  a2 <- matrix(c(1,0,0,1),2,2)
  expect_equal(genMMt(et,a1,"*","sum"), et%*%t(a1))
  etexp <- cbind(pmax(et[,1]+1,et[,2]),pmax(et[,1],et[,2]+1))
  expect_equal(genMMt(et,a2,"+","max"),etexp)

})

test_that("genMMt tensor", {
  et <- cbind(rep(c(-1,0,1),3),rep(c(-1,0,1),each=3))
  ett <- torch_tensor(et)
  a1 <- matrix(c(1,2),1,2)
  a1t <- torch_tensor(a1)
  a2 <- torch_tensor(matrix(c(1,0,0,1),2,2))
  expect_equal(as.matrix(genMMt(ett,a1t,"*","sum")), et%*%t(a1))
  etexp <- cbind(pmax(et[,1]+1,et[,2]),pmax(et[,1],et[,2]+1))
  expect_equal(as.matrix(genMMt(ett,a2,"+","max")),etexp)

})

### Generalized Matrix Multiplication with Q-matrix
### genMMtQ(m1,m2,QQ,combOp,summaryOp)


test_that("genMMtQ matrix", {
  et <- cbind(rep(c(-1,0,1),3),rep(c(-1,0,1),each=3))
  a1 <- matrix(1,3,2)
  qq <- matrix(c(TRUE,FALSE,TRUE, FALSE,TRUE,TRUE),3,2)
  expt <- cbind(et[,1],et[,2],et[,1]+et[,2])
  expect_equal(genMMtQ(et,a1,qq,"*","sum"),expt)

})

test_that("genMMtQ tensor", {
  et <- cbind(rep(c(-1,0,1),3),rep(c(-1,0,1),each=3))
  a1 <- torch_tensor(matrix(1,3,2))
  qq <- torch_tensor(matrix(c(TRUE,FALSE,TRUE, FALSE,TRUE,TRUE),3,2))
  expt <- cbind(et[,1],et[,2],et[,1]+et[,2])
  expect_equal(as.matrix(genMMtQ(torch_tensor(et),a1,qq,"*","sum")),expt)

})





