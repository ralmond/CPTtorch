massageParents <- function (parents) {
  if (length(parents) > 0L && length(names(parents)) > 0L) {
    names(parents) <- paste0("P",1L:length(parents))
  if (is.numeric(parents)) {
    parents <- lapply(parents,\(p) paste0("S",1L:p))
  }
  parents <- lapply(parents, \(p) {
    if (is.character(p)) {
      pp <- effectiveTheta(length(p))
      names(pp) <- p
      pp
    } else {
      if (length(names(p)) == 0L) {
        names(p) <- paste0("S",1L:length(p))
      }
      p
    }
    parents
}


CPT <- nn_module(
    classname="CPT",
    inherit=nnModule,
    rule=NULL,
    link=NULL,
    initialize = function(ruletype,linktype,parents=list(),states=character(),
                          QQ=true) {
      self$parentVals <- parents
      self$stateNames <- states
      link <- getLink(linktype)
      if (is.null(link)) abort("Unknown link type",lintype)
      self$link <- link$new(length(states))

      rule <- getLink(ruletype)
      if (is.null(rule)) abort("Unknown rule type",lintype)
      self$rule <- rule$new(self$parentVals,
                            self$link$etWidth(),
                            QQ)
    },
    forward = function (input) {
      private$cpt <- self$link$forward(self$rule$forward())
      private$cpt
    },
    lscore = function (datatab) {
      torch_log(self$forward(NULL))$mul_(datatab)$sum(1:2)
    },
    getCPT = function () {
      if (is.null(private$cpt)) self$forward(NULL)
      private$cpt$reshape_(private$shape)
    },
    getCPF = function () {
      if (is.null(private$cpt)) self$forward(NULL)
      frame <- data.frame(expand.grid(self$parentNames),as_array(private$cpt))
      names(frame) <- c(names(self$parentNames),self$stateNames)
      frame
    },
    getETframe = function () {
      if (is.null(private$rule)) return(NULL)
      et <- self$rule$et
      frame <- data.frame(expand.grid(self$parentNames),as_array(private$cpt))
      names(frame) <- c(names(self$parentNames),self$stateNames[1L:ncol(et)])
      frame
    },
    private=list(
        parents=list(),
        states=character(),
        shape=c(1L,1L)
        cpt=NULL,
        ),
    active=list(
        aMat = function (value) {
          if (missing(value)) return(self$rule$aMat)
          self$rule$aMat <- value
          private$cpt <- NULL
          invisible(self)
        },
        bMat = function (value) {
          if (missing(value)) return(self$rule$bMat)
          self$rule$bMat <- value
          private$cpt <- NULL
          invisible(self)
        },
        linkScale = function(value) {
          if (missing(value)) return (self$link$linkScale)
          self$rule$linkScale <- value
          private$cpt <- NULL
          invisible(self)
        },
        parentVals = function (value) {
          if (missing(value)) private$parents
          private$parents <- massageParents(value)
          private$shape <- c(sapply(private$parents,length),
                             private$shape[length(private$shape)])
          if (!is.null(self$rule)) self$rule$setParents(private$parents)
          private$cpt <- NULL
          invisible(self)
        },
        parentNames = function () {
          lapply(private$parents,names)
        }
        stateNames = function (value) {
          if (missing(value)) private$states
          private$states <- value
          private$shape[length(private$shape)] <- length(value)
          if (!is.null(self$link)) {
            self$link$K <- length(states)
            if (!is.null(self$rule))
              self$rule$setDim(K=self$rule$etWidth())
          }
          private$cpt <- NULL
          invisible(self)
        }
        QQ = function (value) {
          if (is.null(self$rule)) return(NULL)
          if (missing(value)) return(self$rule$QQ)
          self$rule$QQ <- value
          private$cpt <- NULL
          invisible(self)
        }
    )
)
