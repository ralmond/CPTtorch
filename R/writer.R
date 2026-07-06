writeCPT <- function(cpt) {
  jlist <- list(
    classname="CPT_Model",
    parents=lapply(cpt$parentVals,as.list),
    states=cpt$states,
    QQ=cpt$QQ,
    guess=cpt$link$guess,
    slip=cpt$link$slip,
    high2low=cpt$high2low,
    link=list(
      classname=cpt$link$classname,
      sVec=NULL
    ),
    rule=list(
      classname=cpt$rule$classname,
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
  jlist$parentVals <- lapply(jlist$parents, \(v) {
    res <- as.numeric(v)
    names(res) <- names(v)
    res})
  qd1 <- length(jlist$QQ)
  qd2 <- length(jlist$QQ[[1]])
  if (qd1 > 1 || qd2 > 1) {
    jlist$QQ <- matrix(as.logical(purrr::flatten(jlist$QQ)),qd1,qd2,byrow=TRUE)
  }
  if (is.null(jlist$slip)) jlist$slip <- NA
  if (is.null(jlist$guess)) jlist$guess <- NA

  cpt <- CPT_Model$new(jlist$rule$classname,jlist$link$classname,
                       jlist$parents,jlist$states,jlist$QQ,
                       jlist$guess,jlist$slip,jlist$high2low,
                       device=device)
  if (!is.null(jlist$rule$aVec)) {
    cpt$rule$aVec <- torch_load(jsonlite::base64_dec(jlist$rule$aVec),
                                device=device)
  }
  if (!is.null(jlist$rule$bVec)) {
    cpt$rule$bVec <- torch_load(jsonlite::base64_dec(jlist$rule$bVec),
                                device=device)
  }
  if (!is.null(jlist$link$sVec)) {
    cpt$link$sVec <- torch_load(jsonlite::base64_dec(jlist$link$sVec),
                                device=device)
  }
  cpt
}
