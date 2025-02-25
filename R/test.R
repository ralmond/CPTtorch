Parent <- R6::R6Class("Parent",
                     public=list(value=function() {private$val}),
                     private=list(val=0))
Child <- R6::R6Class("Child",inherit=Parent,
                    private=list(val=1))

cc <- Child$new()
cc$value()
pp <- Parent$new()
