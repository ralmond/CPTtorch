expect_eqten <- function (obsdist,truedist,tol=.0001,what="tensor") {
  # copy both tensors to the same device before comparing
  obsdist <- torch_tensor(obsdist, device = torch_device("CPU"))
  truedist <- torch_tensor(truedist, device = torch_device("CPU"))
  maxdistdif <- as.numeric(torch_sub(obsdist,truedist)$abs_()$max())

  if (maxdistdif > tol) {
    fail(paste("Observed and expected", what, "differ, maximum difference ",maxdistdif))
  } else {
    succeed(paste("Observed",what,"matches expected."))
  }
}

#' Sample the results of N students taking a test form with two tasks.
#' probs is the probabilities of students getting both wrong, only the first
#'  task wrong, only the second task wrong, and both correct.
#' seed in should be an integer or NULL.
setup_sample_test_responses <- function(probs = c(0.25, 0.25, 0.25, 0.25),
                                        N = 200, seed = 42) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  outcomes <- matrix(c(1L, 1L,  1L, 2L,  2L, 1L,  2L, 2L), ncol = 2, byrow = TRUE)
  row_indices <- sample(1:nrow(outcomes), size = N, replace = TRUE, prob = probs)
  mat <- outcomes[row_indices,]
  return(mat)
}

test_that("Recover CPT Compensatory/PC", {
  mod0 <- CPT_Model$new("Compensatory","PartialCredit",
                        list('Par1'=c('P1_low', 'P1_hi'), 'Par2'=c('P2_low', 'P2_hi')),
                        c("A","B","C"))
  mod0$penalties <- list("bVec"=1,"sVec"=1)

  mod0Ser <- writeCPT(mod0)
  modRW <- readCPT(mod0Ser, device = torch_device("CPU"))

  expect_eqten(mod0$getCPT(), modRW$getCPT())
})


test_that("Recover CPT Compensatory/PC Different Devices", {
  mod0 <- CPT_Model$new("Compensatory","PartialCredit",
                        list('Par1'=c('P1_low', 'P1_hi'), 'Par2'=c('P2_low', 'P2_hi')),
                        c("A","B","C"))
  mod0$penalties <- list("bVec"=1,"sVec"=1)

  mod0Ser <- writeCPT(mod0)
  modRW <- readCPT(mod0Ser, device = torch_device("CPU"))

  cptMod0 <- mod0$getCPT()
  cptModRW <- modRW$getCPT()

  expect_false(identical(cptMod0$device, cptModRW$device))
  expect_eqten(cptMod0, cptModRW)
})


test_that("Recover Center/Normal parents", {
  mod0 <- CPT_Model$new("Center","Normal",list(),c("A","B","C"))
  mod0$linkScale <- 1
  mod0$penalties <- list("bVec"=1,"sVec"=1)

  mod0Ser <- writeCPT(mod0)
  modRW <- readCPT(mod0Ser)

  expect_eqten(mod0$linkScale, modRW$linkScale)
  expect_eqten(mod0$getCPT(), modRW$getCPT())
})


test_that("Recover CPT Compensatory/PC after Training", {
  mod0 <- CPT_Model$new("Compensatory","PartialCredit",
                        list('Par1'=c('P1_low', 'P1_hi'), 'Par2'=c('P2_low', 'P2_hi')),
                        c("A","B","C"))
  mod0$penalties <- list("bVec"=1,"sVec"=1)
  mod1 <- CPT_Model$new("Compensatory","PartialCredit",
                        list('Par1'=c('P1_low', 'P1_hi'), 'Par2'=c('P2_low', 'P2_hi')),
                        c("A","B","C"))
  mod1$bMat <- matrix(c(0,.5),2,1)

  cpt1 <- mod1$getCPT()
  dattab <- torch_mul(cpt1,1000)

  #  conv <- fit2table(mod0,dattab,log=c("bVec","sVec","cpt"), maxit=100L)
  conv <- fit2table(mod0,dattab, maxit=100L)
  if (!conv) warning("Model fitting did not converge")

  expect_eqten(mod0$getCPT(),cpt1,tol=3e-3)

  mod0Ser <- writeCPT(mod0)
  modRW <- readCPT(mod0Ser)

  expect_eqten(modRW$getCPT(),cpt1,tol=3e-3)
})


test_that("Recover CPT Center/Normal after training", {
  mod0 <- CPT_Model$new("Center","Normal",list(),c("A","B","C"))
  mod0$linkScale <- 1
  mod0$penalties <- list("bVec"=1,"sVec"=1)
  mod1 <- CPT_Model$new("Center","Normal",list(),c("A","B","C"))
  mod1$linkScale <- .5
  mod1$bMat <- matrix(.5,1,1)
  cpt1 <- mod1$getCPT()
  dattab <- torch_mul(cpt1,1000)


  #conv <- fit2table(mod0,dattab,log=c("bVec","sVec","cpt"),maxit=200L)
  conv <- fit2table(mod0,dattab,maxit=200L)
  if (!conv) warning("Model fitting did not converge")

  mod0Ser <- writeCPT(mod0)
  modRW <- readCPT(mod0Ser)

  expect_eqten(modRW$getCPT(),cpt1,tol=3e-3)
})


test_that("Recover CPT after Training, saving to .json File", {
  mod0 <- CPT_Model$new("Center","Normal",list(),c("A","B","C"))
  mod0$linkScale <- 1
  mod0$penalties <- list("bVec"=1,"sVec"=1)
  mod1 <- CPT_Model$new("Center","Normal",list(),c("A","B","C"))
  mod1$linkScale <- .5
  mod1$bMat <- matrix(.5,1,1)
  cpt1 <- mod1$getCPT()
  dattab <- torch_mul(cpt1,1000)


  #conv <- fit2table(mod0,dattab,log=c("bVec","sVec","cpt"),maxit=200L)
  conv <- fit2table(mod0,dattab,maxit=200L)
  if (!conv) warning("Model fitting did not converge")

  mod0Ser <- writeCPT(mod0)
  cat(mod0Ser, file = "trained_CPT_model.json")
  mod0SerFileW <- paste(readLines("trained_CPT_model.json", warn=F), collapse = "\n")
  modRW <- readCPT(mod0SerFileW)

  expect_eqten(modRW$getCPT(),cpt1,tol=3e-3)
})


test_that("Recover CDM", {
  # Setup a simple quiz with two tasks designed to test two latent skills
  qmat <- matrix(c(T,T,F,T), nrow=2)
  colnames(qmat) <- c("Skill 1", "General Skill")
  row.names(qmat) <- c("Sk1 spfc Task", "Gen Task")

  scoring_states <- lapply(rownames(qmat), function(name) paste0(name, "_", c("incorr", "corr")))
  latent_skill_levels <- lapply(colnames(qmat), function(s) paste0(s, "_", 1:2))

  mod0 <- Cognitively_Diagnostic_Model(
    "Compensatory", "PartialCredit", qmat, latent_skill_levels, scoring_states
  )

  # Suppose that students are better at the general skill than skill 1 and
  #   students who complete the Skill 1-specific Task are 90% likely to get the
  #   general task correct too
  task_scores <- setup_sample_test_responses(c(0.37, 0.33, 0.03, 0.27))

  mod0Ser <- writeCDM(mod0)
  modRW <- readCDM(mod0Ser)

  exp_CPTs <- mod0$get_expected_contingency_tables(task_scores)
  exp_CPTs_RW <- modRW$get_expected_contingency_tables(task_scores)

  expect_eqten(exp_CPTs$Phi_I0, exp_CPTs_RW$Phi_I0)
  expect_equal(length(exp_CPTs$Phi_Ijs), 2)
  expect_equal(length(exp_CPTs$Phi_Ijs), length(exp_CPTs_RW$Phi_Ijs))
  for (i in length(exp_CPTs$Phi_Ijs)) {
    expect_eqten(exp_CPTs$Phi_Ijs[[i]], exp_CPTs_RW$Phi_Ijs[[i]])
  }
})


test_that("Recover CDM .json File", {
  qmat <- matrix(c(T,T,F,T), nrow=2)
  colnames(qmat) <- c("Skill 1", "General Skill")
  row.names(qmat) <- c("Sk1 spfc Task", "Gen Task")
  scoring_states <- lapply(rownames(qmat), function(name) paste0(name, "_", c("incorr", "corr")))
  latent_skill_levels <- lapply(colnames(qmat), function(s) paste0(s, "_", 1:2))
  task_scores <- setup_sample_test_responses(c(0.37, 0.33, 0.03, 0.27))

  mod0 <- Cognitively_Diagnostic_Model(
    "Compensatory", "PartialCredit", qmat, latent_skill_levels, scoring_states
  )

  mod0Ser <- writeCDM(mod0)
  cat(mod0Ser, file = "CDM_model.json")
  mod0SerFileW <- paste(readLines("CDM_model.json", warn=F), collapse = "\n")

  modRW <- readCDM(mod0SerFileW)

  exp_CPTs <- mod0$get_expected_contingency_tables(task_scores)
  exp_CPTs_RW <- modRW$get_expected_contingency_tables(task_scores)

  expect_eqten(exp_CPTs$Phi_I0, exp_CPTs_RW$Phi_I0)
  expect_equal(length(exp_CPTs$Phi_Ijs), 2)
  expect_equal(length(exp_CPTs$Phi_Ijs), length(exp_CPTs_RW$Phi_Ijs))
  for (i in length(exp_CPTs$Phi_Ijs)) {
    expect_eqten(exp_CPTs$Phi_Ijs[[i]], exp_CPTs_RW$Phi_Ijs[[i]])
  }
})


test_that("Recover CDM Train", {
  # Setup a simple quiz with two tasks designed to test two latent skills
  qmat <- matrix(c(T,T,F,T), nrow=2)
  colnames(qmat) <- c("Skill 1", "General Skill")
  row.names(qmat) <- c("Sk1 spfc Task", "Gen Task")

  scoring_states <- lapply(rownames(qmat), function(name) paste0(name, "_", c("incorr", "corr")))
  latent_skill_levels <- lapply(colnames(qmat), function(s) paste0(s, "_", 1:2))


  mod0 <- Cognitively_Diagnostic_Model(
    "Compensatory", "PartialCredit", qmat, latent_skill_levels, scoring_states
  )

  # Suppose that students are better at the general skill than skill 1 and
  #   students who complete the Skill 1-specific Task are 90% likely to get the
  #   general task correct too
  task_scores <- setup_sample_test_responses(c(0.37, 0.33, 0.03, 0.27))

  fit_with_EM(mod0, task_scores, maxit=50L, tolerance=0.01)

  mod0Ser <- writeCDM(mod0)
  modRW <- readCDM(mod0Ser)

  exp_CPTs <- mod0$get_expected_contingency_tables(task_scores)
  exp_CPTs_RW <- modRW$get_expected_contingency_tables(task_scores)

  expect_eqten(exp_CPTs$Phi_I0, exp_CPTs_RW$Phi_I0)
  expect_equal(length(exp_CPTs$Phi_Ijs), 2)
  expect_equal(length(exp_CPTs$Phi_Ijs), length(exp_CPTs_RW$Phi_Ijs))
  for (i in length(exp_CPTs$Phi_Ijs)) {
    expect_eqten(exp_CPTs$Phi_Ijs[[i]], exp_CPTs_RW$Phi_Ijs[[i]])
  }
})


test_that("Recover CDM Train: Different Devices", {
  qmat <- matrix(c(T,T,F,T), nrow=2)
  colnames(qmat) <- c("Skill 1", "General Skill")
  row.names(qmat) <- c("Sk1 spfc Task", "Gen Task")

  scoring_states <- lapply(rownames(qmat), function(name) paste0(name, "_", c("incorr", "corr")))
  latent_skill_levels <- lapply(colnames(qmat), function(s) paste0(s, "_", 1:2))


  mod0 <- Cognitively_Diagnostic_Model(
    "Compensatory", "PartialCredit", qmat, latent_skill_levels, scoring_states
  )

  task_scores <- setup_sample_test_responses(c(0.37, 0.33, 0.03, 0.27))

  fit_with_EM(mod0, task_scores, maxit=50L, tolerance=0.01)

  mod0Ser <- writeCDM(mod0)
  modRW <- readCDM(mod0Ser, device=torch_device("cpu"))

  exp_CPTs <- mod0$get_expected_contingency_tables(task_scores)
  exp_CPTs_RW <- modRW$get_expected_contingency_tables(task_scores)

  expect_false(identical(exp_CPTs$Phi_I0$device, exp_CPTs_RW$Phi_I0$device))

  expect_eqten(exp_CPTs$Phi_I0, exp_CPTs_RW$Phi_I0)
  expect_equal(length(exp_CPTs$Phi_Ijs), 2)
  expect_equal(length(exp_CPTs$Phi_Ijs), length(exp_CPTs_RW$Phi_Ijs))
  for (i in length(exp_CPTs$Phi_Ijs)) {
    expect_eqten(exp_CPTs$Phi_Ijs[[i]], exp_CPTs_RW$Phi_Ijs[[i]])
  }
})
