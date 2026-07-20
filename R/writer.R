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
  if (!is.null(cpt$link$sVec))
    jlist$link$sVec <- jsonlite::base64_enc(torch_serialize(cpt$link$sVec))
  if (!is.null(cpt$rule$aVec))
    jlist$rule$aVec <- jsonlite::base64_enc(torch_serialize(cpt$rule$aVec))
  if (!is.null(cpt$rule$bVec))
    jlist$rule$bVec <- jsonlite::base64_enc(torch_serialize(cpt$rule$bVec))
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
  jlist$rule$aVec <- unlist(jlist$rule$aVec)
  if (!is.null(jlist$rule$aVec)) {
    cpt$rule$aVec <- torch_load(jsonlite::base64_dec(jlist$rule$aVec),
                                device=device)
  }
  jlist$rule$bVec <- unlist(jlist$rule$bVec)
  if (!is.null(jlist$rule$bVec)) {
    cpt$rule$bVec <- torch_load(jsonlite::base64_dec(jlist$rule$bVec),
                                device=device)
  }
  jlist$link$sVec <- unlist(jlist$link$sVec)
  if (!is.null(jlist$link$sVec)) {
    cpt$link$sVec <- torch_load(jsonlite::base64_dec(jlist$link$sVec),
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
  # Goals:
  # 0. build all CPTs
  # 1. call CDM constructor
  # 2. update state

  # CDM(ruletype,linktype,q_matrix,latent_skill_levels=list(),scoring_states=list(),
  #          guess=NA,slip=NA,high2low=FALSE,device=CPTtorch::CPTtorch_device())
  # of these params, q_matrix,latent_skill_levels,scoring_states are used for non-CPT-only ops
  jlist <- jsonlite::fromJSON(serial,FALSE)
  if (jlist$classname != "Cognitively_Diagnostic_Model") {
    stop("Expected Cognitively_Diagnostic_Model JSON")
  }

  # 0. build all CPTs
  CPTs <- lapply(unlist(jlist$CPTs), function(x) readCPT(x, device = device))

  # 1. call CDM constructor
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

  cdm
}
