# !diagnostics suppress=self,private,super

### Node ----

Node <- R6Class(
  "Node",
  public=list(
    name=character(),
    states=character(),
    values=numeric(),
    pos=NULL,
    cpt=NULL,
    initialize=function(name,states,values=effectiveTheta(length(states))) {
        self$name <- name
        self$states <- states
        self$values <- values
    }
  )
)

### Statistic ----

Statistic <- R6Class(
  "Statistic",
  public=list(
    name=character(),
    node=character(),
    fun=NULL,
    marginDims=NULL,
    initialize = function(name,fun,node) {
      self$name <- name
      self$fun <- fun
      self$node <- node
    },
    setProfOrder = function(proficiencies) {
      self$marginDims <- (1:length(proficiencies))[
        -match(self$node$name,proficiencies)]
    },
    calc = function(sm) {
      dist <- sm$sum(self$marginDims)
      do.call(self$fun,list(dist,self$node))
    })
)

Margin <- function (dist,node) {
  stat <- as.numeric(dist)
  names(stat) <- node$states
  stat
}
EAP <- function (dist,node){
  sum(as.numeric(dist)*node$values)
}
SD <- function (dist,node){
  s1 <- sum(as.numeric(dist)*node$values)
  s2 <- sum(as.numeric(dist)*node$values*node$values)
  sqrt(s2-s1^2)
}
Median <- function(dist,node){
  cdist <- cumsum(as.numeric(dist))
  node$states[min(which(cdist>.5))]
}
Mode <- function(dist,node){
  node$states[which.max(as.numeric(dist))]
}


