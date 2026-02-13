# !diagnostics suppress=self,private,super

normalize_tensor <- function(t) {
  t / torch_sum(t)
}

deviance_loss_T <- function(Phi_I, cpt, ccbias=0) {
  Phi_tilde <- Phi_I$add(cpt, alpha=ccbias)
  cpt$log()$mul(Phi_tilde)$sum()$mul(-2)
}

cdm_loss_fun <- function(phi_0, Phi_I0, phi_js, Phi_Ijs, ccbias) {
  loss <- deviance_loss_T(Phi_I0, phi_0, ccbias)

  for (j in 1:length(Phi_Ijs)) {
    loss <- loss$add(deviance_loss_T(Phi_Ijs[[j]], phi_js[[j]], ccbias))
  }
  loss
}

build_loss_fun_cdm <- function (phi_0, Phi_I0, Phi_Ijs, ccbias, penalties) {
  function(phi_js, params) {
    loss <- cdm_loss_fun(phi_0, Phi_I0, phi_js, Phi_Ijs, ccbias)

    for (ipar in names(penalties))
      loss <- loss$add(penalty_fun(params,ipar,penalties[[ipar]]))

    loss
  }
}


Cognitively_Diagnostic_Model <- nn_module(
    classname="Cognitively_Diagnostic_Model",
    ccbias=10,
    bin_eps=1e-6, # 1/(N_STATES = 3^10) 1.69e-5
    optimizer=NULL,
    oconstructor="optim_adam",
    oparams=list(lr=0.001),
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
      # phi_0
      self$proficiency_potential <- torch_ones(n_levels_per_latent_skill, requires_grad=FALSE)
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

      self$evidence_models <- lapply(1:self$n_tasks, function(j) CPT_Model$new(
        ruletype, linktype, latent_skill_levels[which(q_matrix[j,])], scoring_states[[j]],
        guess=guess[j] ,slip=slip[j], high2low=high2low[j])
      )
    },
    forward = function (task_scores) {
      #' task_scores: matrix of dim(n_students, self$n_tasks) where each entry is NA or an int corresponding to the index of the score (i.e., `task_scores[j] = k` means the j'th task has the k'th possible value among possibilities, 2 for 'correct' from ('incorrect', 'correct'), 1 for 'high' from ('high', 'medium', 'low'), etc.)
      #' get the global and task-specific CPTs and expected contingency tables
      if (!is.matrix(task_scores) || !is.integer(task_scores))
        stop("Error `task_scores` must be an integer matrix (NAs allowed, see docstring)")

      c(list(phi_0=self$proficiency_potential, phi_js=self$get_evidence_CPTs()),
        self$get_expected_contingency_tables(task_scores))
    },
    get_expected_contingency_tables = function(task_scores, calc_for_tasks=T) {
      #' task_scores: matrix of dim(n_students, self$n_tasks) where each entry is NA or an int corresponding to the index of the score (i.e., `task_scores[j] = k` means the j'th task has the k'th possible value among possibilities, 2 for 'correct' from ('incorrect', 'correct'), 1 for 'high' from ('high', 'medium', 'low'), etc.)
      #' get the global and task-specific expected contingency tables
      if (!is.matrix(task_scores) || !is.integer(task_scores))
        stop("Error `task_scores` must be an integer matrix (NAs allowed, see docstring)")
      stopifnot("Error: 2nd dim of task_scores must equal # of tasks with scoring_states set during initialization"=dim(task_scores)[2]==self$n_tasks)

      # convert task_scores to a tensor type
      task_scores_T <- torch_tensor(task_scores, torch_int64(), requires_grad=F)
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
      Phi_I0 <- torch_zeros_like(self$proficiency_potential, requires_grad=FALSE)
      for (normed_stu_i_proficiencies in normed_stu_proficiencies) {
        Phi_I0 <- Phi_I0$add(normed_stu_i_proficiencies)
      }
      Phi_I0$add_(self$bin_eps)

      # Finally, get the task-specific statistics
      Phi_Ijs <- NULL
      if (calc_for_tasks)
        Phi_Ijs <- lapply(1:self$n_tasks, function(j) {
          Phi_j <- torch_stack(
            lapply(1:length(self$scoring_states[[j]]), function(k) {
              Phi_jk <- 0
              for (i in 1:n_stus)
                if (isTRUE(task_scores[i,j]==k)) {
                  Phi_jk <- self$marginalize_ops[[j]](normed_stu_proficiencies[[i]])$add(Phi_jk)
                }
              return(Phi_jk)
            }
            ), dim=sum(aced_model$q_matrix[j,])+1
          )
        })

      list(Phi_I0 = Phi_I0, Phi_Ijs = Phi_Ijs)
    },
    get_evidence_CPTs = function() {
      lapply(self$evidence_models, function(ev_model) normalize_tensor(ev_model$getCPT()))
    },
    numparams = function () {
      sum(lapply(self$evidence_models, function(ev_model) ev_model$numparams()))
    },
    params = function() {
      plist <- list()
      for (ev_model in self$evidence_models) {
        plist <- c(plist, ev_model$params())
      }
      plist[!sapply(plist,is.null)]
    },
    deviance = function(task_scores) {
      with_no_grad({
        fwd_out = self$forward(task_scores)

        dev_loss <- cdm_loss_fun(
          fwd_out$phi_0, fwd_out$Phi_I0,
          fwd_out$phi_js, fwd_out$Phi_Ijs, self$ccbias)$item()
      })
      dev_loss
    },
    AIC = function(task_scores) {
      as_array(self$deviance(task_scores)) + 2*self$numparams()
    },
    train = function() {
      super$train()
      for (ev_model in self$evidence_models)
        ev_model$train()
    },
    eval = function() {
      super$eval()
      for (ev_model in self$evidence_models)
        ev_model$eval()
    },
    buildOptimizer = function() {
      self$optimizer <-
        do.call(self$oconstructor, c(list(self$params()), self$oparams))
      self$optimizer
    },
    step = function (phi_0, Phi_I0, Phi_Ijs, r=1L) {
      if (is.null(self$optimizer)) self$buildOptimizer()

      lossfn <- jit_trace(
        build_loss_fun_cdm(phi_0, Phi_I0, Phi_Ijs, self$ccbias, self$penalties),
        self$get_evidence_CPTs(), self$params()
      )

      for (rr in 1:r) {
        self$optimizer$zero_grad()
        ev_cpts <- self$get_evidence_CPTs()
        step_loss <- lossfn(ev_cpts, self$params())
        step_loss_val <- step_loss$item()
        step_loss$backward(retain_graph=TRUE)
        self$optimizer$step()
        # print(c('    Step loss ', round(step_loss_val, 4)))
        # print(c('    cpt_1 ', ev_cpts[[1]]))
        # print(c('    A_1 ', self$evidence_models[[1]]$aMat))
      }
      final_loss <- NA
      with_no_grad({
        final_loss <- cdm_loss_fun(phi_0, Phi_I0, self$get_evidence_CPTs(),
                                   Phi_Ijs, self$ccbias)
      })
      final_loss
    }
  )

fit_with_EM <- function(model, task_scores, penalties=list(), tolerance=1e-3,
                        maxit=100L, maxit_m_step=5L, half_life=20L) {
  if (!is.matrix(task_scores) || !is.integer(task_scores))
    stop("Error `task_scores` must be an integer matrix (NAs allowed, see docstring for forward())")
  stopifnot("Error: 2nd dim of task_scores must equal # of tasks set during model initialization"=dim(task_scores)[2]==model$n_tasks)
  n_stus <- dim(task_scores)[1]

  # setup training loop params
  rit <- 0L
  old_loss <- Inf
  loss_hist <- rep(NA, maxit)
  optimizer <- model$buildOptimizer()
  scheduler <- lr_step(optimizer, step_size = half_life, gamma = 0.5)
  m_step_iters <- maxit_m_step

  while (rit < maxit) {
    # E step
    with_no_grad({
      exp_contin_tables <- model$get_expected_contingency_tables(task_scores)
      phi_0 <- exp_contin_tables$Phi_I0 / n_stus
    })

    # M step
    model$train()
    final_loss <- model$step(phi_0, exp_contin_tables$Phi_I0,
                             exp_contin_tables$Phi_Ijs, r=m_step_iters)

    # display training info
    loss_val <- final_loss$item()
    loss_hist[rit] <- loss_val

    print(paste('TRAINING ITER', rit, '| LOSS', round(loss_val, 4), '| DELTA LOSS', round(old_loss-loss_val, 4)))
    if (rit > 2) {
      while (!is.null(dev.list())) dev.off()
      plot(1:rit, loss_hist[1:rit], 'l', main=paste0('Loss Curve at r=', rit), xlab='epoch', ylab='Penalized NLL Loss')
    }

    # update model and optimization state
    with_no_grad({
      model$proficiency_potential <- model$get_expected_contingency_tables(task_scores, F)$Phi_I0 / n_stus
    })

    rit <- rit+1
    scheduler$step()

    if (rit %% half_life == 0)
      # m_step_iters <- m_step_iters + maxit_m_step
      m_step_iters <- m_step_iters*2L

    if (abs(old_loss-loss_val) < tolerance)
      break
    old_loss <- loss_val
  }

  model$eval()

  print(paste0(rit,'/', maxit, ' iters taken'))
  print(paste0('min loss diff b/w iters ', min(rit[2:rit]-rit[1:(rit-1)])))


  # plot the loss curve
  while (!is.null(dev.list())) dev.off()
  plot(1:rit, loss_hist[1:rit], 'l', main='Loss Curve', xlab='epoch', ylab='Penalized NLL Loss')

  return (rit < maxit)
}
