### This is an abstraction of redis combining it with
### tensor serialization/deserialization.
### It only uses the redis operations SET, GET, RPUSH and RPOP.

tred <- R6Class(
    "tred",
    public=list(
        red = NULL,
        device="cpu",
        initialize = function(url="redis://localhost:6379",
                              device=CPTtorch_device()) {
          self$red <- redux::hiredis(url=url)
          self$device <- device
        },
        SETt = function(name,tensor) {
          self$red$SET(name,torch_serialize(tensor))
        },
        SETs = function(name,val) {
          self$red$SET(name,val)
        },
        SETCHUNKS = function(cpt) {
          chunks <- cpt$getChunks()
          cmds <- lapply(seq_len(length(chunks)), \(i) {
            redux::redis$SET(paste("CPT",cpt$child,i,sep="."),
                             torch_serialize(chunks[[i]]))
          })
          self$red$pipeline(.commands=cmds)
        },
        GETt = function(name) {
          torch_load(self$red$GET(name),device=self$device)
        },
        GETs = function(name) {
          as.numeric(self$red$GET(name))
        },
        PUSHt = function(name,tab) {
          self$red$RPUSH(name,torch_serialize(tab))
        },
        POPt = function(name) {
          val <- self$red$RPOP(name)
          if (is.null(val)) return(NULL)
          torch_load(val,device=self$device)
        },
        PUSHs = function(name,val) {
          self$red$RPUSH(name,val)
        },
        POPs = function(name) {
          val <- self$red$RPOP(name)
          if (is.null(val)) return(NULL)
          as.numeric(val)
        }
    )
)

