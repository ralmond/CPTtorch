lr_phased <- lr_scheduler(
    classname="lr_phased",
    initialize=function(optimizer,phases,lrms,msteps,last_epoch=-1) {
      self$phases <- phases
      self$lrms <- lrms
      self$msteps <- msteps
      self$ilrs <- sapply(optimizer$param_groups, \(x) x$lr)
      super$initialize(optimizer, last_epoch)
    },
    get_phase = function () {
      sum(self$last_epoch < self$phases)
    },
    get_lr = function() {
      self$ilrs*self$lrms[self$get_phase()]
    },
    get_msteps = function() {
      self$msteps[self$get_phase()]
    }
)
    
