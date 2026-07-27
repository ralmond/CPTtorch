writeCPT <- function(cpt) {
  jlist <- list(
    classname="CPT_Model",
    parents=lapply(cpt$parentVals,as.list),
    stateNames=cpt$stateNames,
    QQ=cpt$QQ,
    guess=cpt$link$guess,
    slip=cpt$link$slip,
    high2low=cpt$high2low,
    link=list(
      classname=class(cpt$link)[[1]],
      sVec=NULL
    ),
    rule=list(
      classname=class(cpt$rule)[[1]],
      aVec=NULL,
      bVec=NULL
    )
  )
  if (!is.null(cpt$link$linkScale))
    jlist$link$linkScale <- jsonlite::base64_enc(torch_serialize(cpt$link$linkScale))
  if (!is.null(cpt$rule$aMat))
    jlist$rule$aMat <- jsonlite::base64_enc(torch_serialize(cpt$rule$aMat))
  if (!is.null(cpt$rule$bVec))
    jlist$rule$bMat <- jsonlite::base64_enc(torch_serialize(cpt$rule$bMat))
  jsonlite::toJSON(jlist)
}

readCPT <- function (serial,device=CPTtorch_device()) {
  jlist <- jsonlite::fromJSON(serial,FALSE)
  if (jlist$classname != "CPT_Model") {
    stop("Expected CPT_Model JSON")
  }
  jlist$parentNames <- lapply(jlist$parents, names)
  names(jlist$parentNames) <- NULL
  QMatrix <- jlist$QQ[[1]]
  qd1 <- length(QMatrix)
  qd2 <- length(QMatrix[[1]])
  if (qd1 > 1 || qd2 > 1) {
    jlist$QQ <- matrix(as.logical(purrr::flatten(QMatrix)),qd1,qd2,byrow=TRUE)
  } else {
    jlist$QQ <- TRUE
  }
  if (is.null(jlist$slip[[1]])) jlist$slip[[1]] <- NA
  if (is.null(jlist$guess[[1]])) jlist$guess[[1]] <- NA

  cpt <- CPT_Model$new(jlist$rule$classname[[1]],jlist$link$classname[[1]],
                       jlist$parentNames,unlist(jlist$stateNames),jlist$QQ,
                       jlist$guess[[1]],jlist$slip[[1]],jlist$high2low[[1]],
                       device=device)
  jlist$rule$aMat <- unlist(jlist$rule$aMat)
  if (!is.null(jlist$rule$aMat)) {
    cpt$rule$aMat <- torch_load(jsonlite::base64_dec(jlist$rule$aMat),
                                device=device)
  }
  jlist$rule$bMat <- unlist(jlist$rule$bMat)
  if (!is.null(jlist$rule$bMat)) {
    cpt$rule$bMat <- torch_load(jsonlite::base64_dec(jlist$rule$bMat),
                                device=device)
  }
  jlist$link$linkScale <- unlist(jlist$link$linkScale)
  if (!is.null(jlist$link$linkScale)) {
    cpt$link$linkScale <- torch_load(jsonlite::base64_dec(jlist$link$linkScale),
                                device=device)
  }
  cpt
}

writeCDM <- function(cdm) {
  jlist <- list(
    classname="Cognitively_Diagnostic_Model",
    q_matrix=cdm$q_matrix,
    latent_skill_levels=cdm$latent_skill_levels,
    scoring_states=cdm$scoring_states,
    proficiency_potential=jsonlite::base64_enc(torch_serialize(cdm$proficiency_potential)),
    CPTs=lapply(cdm$evidence_models, writeCPT)
  )
  jsonlite::toJSON(jlist)
}

readCDM <- function (serial,device=CPTtorch_device()) {
  # Three stages:
  # 1. build all CPTs
  # 2. call CDM constructor
  # 3. update CDM state

  jlist <- jsonlite::fromJSON(serial,FALSE)
  if (jlist$classname != "Cognitively_Diagnostic_Model") {
    stop("Expected Cognitively_Diagnostic_Model JSON")
  }

  # 1. build all CPTs
  CPTs <- lapply(unlist(jlist$CPTs), function(x) readCPT(x, device = device))

  # 2. call CDM constructor
  #   make guess, slip, high2low vectors
  guesses <- sapply(CPTs, function(x) x$link$guess)
  slips <- sapply(CPTs, function(x) x$link$slip)
  high2lows <- sapply(CPTs, function(x) x$rule$high2low)

  QMatrix <- jlist$q_matrix
  qd1 <- length(QMatrix)
  qd2 <- length(QMatrix[[1]])
  if (qd1 > 1 || qd2 > 1) {
    jlist$q_matrix <- matrix(as.logical(purrr::flatten(QMatrix)),qd1,qd2,byrow=TRUE)
  } else {
    jlist$q_matrix <- TRUE
  }

  ruleType <- class(CPTs[[1]]$rule)[[1]]
  linkType <- class(CPTs[[1]]$link)[[1]]

  latent_skill_levels <- lapply(jlist$latent_skill_levels, unlist)
  scoring_states <- lapply(jlist$scoring_states, unlist)

  cdm <- Cognitively_Diagnostic_Model(
    ruleType, linkType, jlist$q_matrix, latent_skill_levels, scoring_states,
    guesses, slips, high2lows, device = device
  )

  # 3. update CDM state
  cdm$proficiency_potential <- torch_load(jsonlite::base64_dec(jlist$proficiency_potential[[1]]),
                                          device=device)
  cdm$proficiency_potential <- torch_tensor(cdm$proficiency_potential, requires_grad = F)
  cdm$evidence_models <- CPTs

  cdm
}
