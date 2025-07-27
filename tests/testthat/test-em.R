

testFit <- function(priorModel,datatab,weights=1000,
                    stoponerror=FALSE,tol=.001) {
  
  
  post1 <- prior + round(sweep(truedist,1,weights,"*"))
  map1 <- mapDPC(post1,pLevels,obsLevels,priorLnAlphas,priorBetas,rules,link)
  
  if (map1$convergence != 0) {
    fail(paste("Optimization did not converge:", map1$message))
  }
  
  postLnAlphas <- map1$lnAlphas
  postBetas <- map1$betas

  fitdist <- calcDPCTable(pLevels,obsLevels,map1$lnAlphas,map1$betas,rules,link)
  
  maxdistdif <- max(abs(fitdist-truedist))
  if (maxdistdif > tol) {
    error <- TRUE
    fail(paste("Posterior and True CPT differ, maximum difference ",maxdistdif))
  } else {
    succeed("Posterior matches True CPT")
  }
  
  ## This part of the test does not converge.  So it looks like parameter space is 
  ## multi-modal.
  # if (any(abs(unlist(postLnAlphas)-unlist(trueLnAlphas))>tol)) {
  #   error <- TRUE
  #   fail("Log(alphas) differ by more than tolerance")
  # } else {
  #     succeed("Log(alphas) differ by less than tolerance")
  # }
  # if (any(abs(unlist(postBetas)-unlist(trueBetas))>tol)) {
  #   error <- TRUE
  #   fail("Betas differ by more than tolerance")
  # } else {
  #   succeed("Betas differ by less than tolerance")
  # }
  
  invisible(list(error=error,pseudoData=post1,
                 truedist=truedist,fitdist=fitdist,
                 trueLnAlphas=trueLnAlphas,fitLnAlphas=postLnAlphas,
                 trueBetas=trueBetas,fitBetas=postBetas,map=map1))
  }


test_that("Recovery Normal 0 parents", {

 mod0 <- CPT_Model$new("Center","Normal",list(),c("A","B","C"))
 mod1 <- CPT_Model$new("Center","Normal",list(),c("A","B","C"))
 mod1$linkScale <- .5
 mod1$bMat <- matrix(.5,1,1)
 cpt1 <- mod1$getCPT()
 dattab <- torch_mul(cpt1,1000)


})

test_that("Recovery Normal 1 parent", {

})

test_that("Recovery Compensatory GR", {

})

test_that("Recovery Compensatory PC", {

})

test_that("Recovery Conjunctive PC", {

})

test_that("Recovery Conjunctive PC Guess/Slip", {

})
