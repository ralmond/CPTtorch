# !diagnostics suppress=self,private,super

normalize_tensor <- function(t) {
  t / torch_sum(t)
}

deviance_loss_T <- function(Phi_I, cpt, ccbias=0) {
  Phi_tilde <- Phi_I$add(cpt, alpha=ccbias)
  cpt$log()$mul_(Phi_tilde)$sum()$mul_(-2)
}

cdm_loss_fun <- function(phi_0, Phi_I0, phi_js, Phi_Ijs,
                         ccbias=0, penalties=list()) {
  loss <- deviance_loss_T(Phi_I0, phi_0, ccbias)

  for (j in names(Phi_Ijs)) {
    loss$add_(deviance_loss_T(Phi_Ijs[[j]], phi_js[[j]], ccbias))
  }

  for (ipar in names(penalties))
    loss$add_(penalty_fun(params,ipar,penalties[[ipar]]))

  loss
}

Cognitively_Diagnostic_Model <-
  torch::nn_module(
    classname="Cognitively_Diagnostic_Model",
    ccbias=10,
    optimizer=NULL,
    oconstructor="optim_adam",
    oparams=list(lr=.1),
    lossfn=NULL,
    initialize = function(ruletype,linktype,q_matrix,latent_skill_levels=list(),scoring_states=list(),
                          guess=NA,slip=NA,high2low=FALSE) {
      stopifnot(length(latent_skill_levels)==dim(q_matrix)[[2]])
      stopifnot(self$n_tasks==dim(q_matrix)[[1]])
      self$n_tasks <- length(scoring_states)

      self$latent_skill_levels <- latent_skill_levels
      self$scoring_states <- scoring_states
      self$q_matrix <- q_matrix

      n_levels_per_latent_skill <- sapply(latent_skill_levels, length)
      self$latent_skills_per_task <- rowSums(q_matrix, na.rm=TRUE)
      self$proficiency_potential <- torch_ones(n_levels_per_latent_skill, requires_grad=TRUE)
      # given the i'th row of the Q matrix, create a function to marginalize out all latent vars where the row is FALSE
      self$marginalize_ops <- lapply(1:self$n_tasks,
        function(i) {function(pot) {if (all(q_matrix[i,])) {pot} else {marginalize(pot, dim=which(!q_matrix[i,]))}}}
      )
      # given the i'th row of the Q matrix, create a function to recover all dims after marginalization
      self$extend_shapes <- lapply(1:self$n_tasks,
        function(j) as.array(ifelse(q_matrix[j,], n_levels_per_latent_skill, 1))
      )
      if (!is.vector(guess))
        guess <- rep(guess, self$n_tasks)
      if (!is.vector(slip))
        slip <- rep(slip, self$n_tasks)
      if (!is.vector(high2low) || length(high2low)==1)
        high2low <- rep(high2low, self$n_tasks)
      self$evidence_models <- lapply(1:self$n_tasks, function(j) CPT_Model$new(ruletype, linktype,
          latent_skill_levels[which(q_matrix[j,])], scoring_states[[j]],
          guess=guess[j] ,slip=slip[j], high2low=high2low[j]
        )
      )
    },
    forward = function (task_scores) {
      #' task_scores: matrix of dim(n_students, self$n_tasks) where each entry is NA or an int corresponding to the index of the score (i.e., `task_scores[j] = k` means the j'th task has the k'th possible value among possibilities, 2 for 'correct' from ('incorrect', 'correct'), 1 for 'high' from ('high', 'medium', 'low'), etc.)
      #' get the global and task-specific CPTs and expected contingency tables
      if (!is.matrix(task_scores) || !is.integer(task_scores))
        stop("Error `task_scores` must be an integer matrix (NAs allowed, see docstring)")
      stopifnot("Error: 2nd dim of task_scores must equal # of tasks with scoring_states set during initialization"=dim(task_scores)[2]==self$n_tasks)

      # convert task_scores to a tensor type
      task_scores_T <- torch_tensor(task_scores, torch_int64())
      n_stus <- dim(task_scores)[1]

      # First, get an estimate of each student's posterior proficiency (phi_J_it)
      normed_stu_proficiencies = rep(list(NULL), n_stus)
      model_CPTs <- lapply(self$evidence_models, function(ev_model) ev_model$getCPT())
      for (i in 1:n_stus) {
        stu_i_proficiencies <- self$proficiency_potential
        for (j in 1:self$n_tasks) {
          if (!is.na(task_scores[i,j])) {
            # get the CPT corresponding to the j'th task's task_scores[i,j]'th score
            task_j_ev_pot <- model_CPTs[[j]][.., task_scores_T[i,j]]
            stu_i_proficiencies <- torch_mul(stu_i_proficiencies, task_j_ev_pot$reshape(self$extend_shapes[[j]]))
          }
        }
        normed_stu_proficiencies[[i]] <- normalize_tensor(stu_i_proficiencies)
      }

      # Next, get the global statistics
      phi_0 <- self$proficiency_potential
      Phi_I0 <- torch_zeros_like(phi_0, requires_grad=FALSE)
      for (normed_stu_i_proficiencies in normed_stu_proficiencies) {
        Phi_I0$add_(normed_stu_i_proficiencies)
      }

      # Finally, get the task-specific statistics
      phi_js <- model_CPTs
      Phi_Ijs <- lapply(j:self$n_tasks, function(j) {
        Phi_j <- torch_stack(
          lapply(1:length(self$scoring_states[[j]]), function(k) {
            Phi_jk <- 0
            for (i in 1:n_stus)
              if (isTRUE(task_scores[i,j]==k))
                Phi_jk <- self$marginalize_ops[[j]](normed_stu_proficiencies[[i]])$add(Phi_jk)
            return(Phi_jk)
          }
          ), dim=length(self$scoring_states[j])+1
        )
      })

      list(phi_0 = phi_0, Phi_I0 = Phi_I0, phi_js = phi_js, Phi_Ijs = Phi_Ijs)
    },
    get_expected_contingency = function(task_scores) {
      #' task_scores: matrix of dim(n_students, self$n_tasks) where each entry is NA or an int corresponding to the index of the score (i.e., `task_scores[j] = k` means the j'th task has the k'th possible value among possibilities, 2 for 'correct' from ('incorrect', 'correct'), 1 for 'high' from ('high', 'medium', 'low'), etc.)
      #' returns the expected prior contingency table (Phi_I0 bar) given this object's current state and `task_scores`
      if (!is.matrix(task_scores) || !is.integer(task_scores))
        stop("Error `task_scores` must be an integer matrix (NAs allowed, see docstring)")
      stopifnot("Error: 2nd dim of task_scores must equal # of tasks with scoring_states set during initialization"=dim(task_scores)[2]==self$n_tasks)

      local_no_grad()

      task_score_tensor <- torch_tensor(task_scores, torch_int64())
      n_stus <- dim(task_scores)[1]
      stu_proficiencies = rep(list(NULL), n_stus)
      for (i in 1:n_stus) {
        stu_i_proficiencies <- torch_tensor(self$proficiency_potential)
        for (j in 1:self$n_tasks) {
          if (!is.na(task_scores[i,j])) {
            # get the CPT corresponding to the j'th task's task_scores[i,j]'th score
            task_j_CPT <-  self$evidence_models[[j]]$getCPT()
            task_j_ev_pot <- task_j_CPT[.., task_score_tensor[i,j]]
            stu_i_proficiencies <- torch_mul(stu_i_proficiencies, task_j_ev_pot$reshape(self$extend_shapes[[j]]))
          }
        }
        stu_proficiencies[[i]] <- stu_i_proficiencies
      }

      Phi_I0 <- torch_zeros_like(self$proficiency_potential)
      for (stu_i_proficiencies in stu_proficiencies) {
        Phi_I0$add_(stu_i_proficiencies/torch_sum(stu_i_proficiencies))
      }
      return(Phi_I0)
    },
    numparams = function () {
      prod(dim(self$proficiency_potential)) + sum(lapply(self$evidence_models, function(ev_model) ev_model$numparams()))
    },
    params = function() {
      plist <- list(self$proficiency_potential)
      for (ev_model in self$evidence_models) {
        plist <- c(plist, ev_model$params())
      }
      plist[!sapply(plist,is.null)]
    },
    AIC = function(datatab) {
      as_array(self$deviance(datatab)) + 2*self$numparams()
    },
    deviance=function(dattab) {
      deviance_loss(dattab,self$forward(),self$ccbias)
    },
    buildOptimizer = function() {
      self$cache <- NULL
      self$lossfn <-
        jit_trace(build_loss_fun(self$ccbias,
                                 self$penalities),
                  self$forward(),
                  self$params())
      self$optimizer <-
        do.call(self$oconstructor,
                c(list(self$params()),self$oparams))
      self$optimizer
    },
    step = function (datatab,r=1L) {
      if (is.null(self$optimizer)) self$buildOptimizer()
      if (is.null(self$lossfn)) {
        self$cache <- NULL
        self$lossfn <-
          jit_trace(build_loss_fun(self$ccbias,self$penalties),
                    datatab,self$forward(),self$params())
      }
      for (rr in 1:r) {
        self$optimizer$zero_grad()
        self$lossfn(dattab=datatab,self$forward(),
                    self$params())$backward(retain_graph=TRUE)
        self$optimizer$step()
      }
      self$cache <- NULL
      self$deviance(datatab)
    }
  )

fit_with_EM <- function(model, task_scores, penalties=list(),
                        maxit=100L, tolerance=.0001) {
  if (!is.matrix(task_scores) || !is.integer(task_scores))
    stop("Error `task_scores` must be an integer matrix (NAs allowed, see docstring for forward())")
  stopifnot("Error: 2nd dim of task_scores must equal # of tasks set during model initialization"=dim(task_scores)[2]==model$n_tasks)
  task_score_tensor <- torch_tensor(task_scores, torch_int64())

  n_stus <- dim(task_scores)[1]
  rit <- 0L
  old_loss <- Inf
  loss_hist <- rep(NA, maxit)
  optimizer <- do.call(model$oconstructor,
            list(params=model$params(), lr=model$oparams$lr))

  while (rit < maxit) {
    # E step
    stu_and_evidence_potentials <- model$forward(task_scores)

    # M step
    optimizer$zero_grad()
    loss <- cdm_loss_fun(stu_and_evidence_potentials$phi_0,
                         stu_and_evidence_potentials$Phi_I0,
                         stu_and_evidence_potentials$phi_js,
                         stu_and_evidence_potentials$Phi_Ijs,
                         model$ccbias)#, model$penalties)
    loss_val <- loss$item()
    loss$backward(retain_graph=TRUE)
    optimizer$step()

    rit <- rit+1
    loss_hist[rit] <- loss_val
    if (abs(old_loss-loss_hist[rit]) < tolerance)
      break
    old_loss <- loss_val
    print(paste0('TRAINING ITER ', rit, ' loss: ', old_loss))
  }
  print(paste0(rit,'/',maxit,' iters taken'))
  print(paste0('min loss diff b/w iters ', min(rit[2:rit]-rit[1:(rit-1)])))
  plot(1:rit, loss_hist[1:rit], 'l', main='loss curve', xlab='iter num')
  return (rit < maxit)
}
