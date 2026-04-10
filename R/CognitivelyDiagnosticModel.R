# !diagnostics suppress=self,private,super
# library(future.apply)
# plan(multisession, workers = parallel::detectCores() - 2)

normalize_tensor <- function(t) {
  t / torch_sum(t)
}

deviance_loss_T <- function(Phi_I, cpt, ccbias=0) {
  Phi_tilde <- Phi_I$add(cpt, alpha=ccbias)
  # return perplexity (mean log-likelihood) by dividing by Phi_I$sum() by the
  #   number of effective students (more "students" are added to Phi_0 due to bin_eps)
  cpt$log()$mul(Phi_tilde)$sum()$mul(-2)$div(Phi_I$sum())
}

cdm_loss_fun <- function(phi_0, Phi_I0, phi_js, Phi_Ijs, ccbias) {
  loss <- deviance_loss_T(Phi_I0, phi_0, ccbias)

  for (j in seq_along(Phi_Ijs))
    loss <- loss$add(deviance_loss_T(Phi_Ijs[[j]], phi_js[[j]], ccbias))

  loss
}

build_obs_index <- function(task_scores) {
  #' Refactors the (potentially very sparse) task_scores matrix into a list of list:
  #' Returns a list obs_index of length n_tasks.
  #' obs_index[[j]] is a list with one integer vector per score level k:
  #'   obs_index[[j]][[k]] = integer vector of student indices who got score k on task j.
  n_tasks <- ncol(task_scores)
  lapply(seq_len(n_tasks), function(j) {
    col <- task_scores[, j]
    score_levels <- sort(unique(col[!is.na(col)]))
    idx_by_score <- vector("list", max(score_levels, 0L))
    for (k in score_levels)
      idx_by_score[[k]] <- which(col == k)
    idx_by_score
  })
}

Cognitively_Diagnostic_Model <- nn_module(
    classname="Cognitively_Diagnostic_Model",
    ccbias=10,
    bin_eps=2^-35,
    optimizer=NULL,
    oconstructor="optim_adam",
    oparams=list(lr=0.1),
    lossfn=NULL,
    initialize = function(ruletype,linktype,q_matrix,latent_skill_levels=list(),scoring_states=list(),
                          guess=NA,slip=NA,high2low=FALSE,device=TORCH_DEVICE) {
      stopifnot(length(latent_skill_levels)==dim(q_matrix)[[2]])
      stopifnot(self$n_tasks==dim(q_matrix)[[1]])
      self$n_tasks <- length(scoring_states)
      self$device <- device

      self$latent_skill_levels <- latent_skill_levels
      self$scoring_states <- scoring_states
      self$q_matrix <- q_matrix

      self$n_levels_per_latent_skill <- sapply(latent_skill_levels, length)
      self$latent_skills_per_task <- rowSums(q_matrix, na.rm=TRUE)

      # phi_0
      self$proficiency_potential <- torch_ones(self$n_levels_per_latent_skill, requires_grad=F, device=device)

      self$task_skill_dims <- lapply(seq_len(self$n_tasks), function(j) { which(!q_matrix[j, ]) })
      # given the i'th row of the Q matrix, create a function to marginalize out all latent vars where the row is FALSE
      self$marginalize_ops <- lapply(seq_len(self$n_tasks),
        function(j) {function(pot) {if (all(q_matrix[j,])) {pot} else {marginalize(pot, dim=self$task_skill_dims[[j]])}}}
      )
      # given the j'th row of the Q matrix, create a function to recover all dims after marginalization
      self$extend_shapes <- lapply(seq_len(self$n_tasks),
        function(j) as.array(ifelse(q_matrix[j,], self$n_levels_per_latent_skill, 1L))
      )

      if (!is.vector(guess))
        guess <- rep(guess, self$n_tasks)
      if (!is.vector(slip))
        slip <- rep(slip, self$n_tasks)
      if (!is.vector(high2low) || length(high2low)==1)
        high2low <- rep(high2low, self$n_tasks)

      self$evidence_models <- lapply(1:self$n_tasks, function(j) CPT_Model$new(
        ruletype, linktype, latent_skill_levels[which(q_matrix[j,])], scoring_states[[j]],
        guess=guess[j] ,slip=slip[j], high2low=high2low[j])$to(device=device)
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
    get_expected_contingency_tables = function(task_scores, calc_for_tasks=T, obs_index=NULL) {
      #' task_scores: matrix of dim(n_students, self$n_tasks) where each entry is NA or an int corresponding to the index of the score (i.e., `task_scores[j] = k` means the j'th task has the k'th possible value among possibilities, 2 for 'correct' from ('incorrect', 'correct'), 1 for 'high' from ('high', 'medium', 'low'), etc.)
      #' get the global and task-specific expected contingency tables
      if (!is.matrix(task_scores) || !is.integer(task_scores))
        stop("Error `task_scores` must be an integer matrix (NAs allowed, see docstring)")
      stopifnot("Error: 2nd dim of task_scores must equal # of tasks with scoring_states set during initialization"=dim(task_scores)[2]==self$n_tasks)

      # build sparse index once if caller didn't pre-build it
      if (is.null(obs_index))
        obs_index <- build_obs_index(task_scores)

      n_stus <- dim(task_scores)[1]
      latent_skill_dims <- self$n_levels_per_latent_skill
      n_latent_skills <- length(latent_skill_dims)

      # precompute the log of all task CPTs, the prior phi_0, and the evidence CPT Phi_I0
      #   each has shape = a subset of (latent_skill_dims)
      log_CPTs <- lapply(self$evidence_models, function(m) m$getCPT()$log())
      # shape = (latent_skill_dims)
      log_prior <- torch_log(self$proficiency_potential)
      # shape = (n_stus, latent_skill_dims)
      log_post <- log_prior$unsqueeze(1)$expand(c(n_stus, latent_skill_dims))$clone()

      # First, using individual tasks, get an estimate of the number of students at each latent ability level
      for (j in seq_len(self$n_tasks)) {
        obs_j <- obs_index[[j]]
        log_cpt_j <- log_CPTs[[j]]
        n_scores_j <- length(self$scoring_states[[j]])

        for (k in seq_len(n_scores_j)) {
          # get the indices (a bij to IDs) of students who scored level k on task j
          stus_k <- obs_j[[k]]
          if (is.null(stus_k) || length(stus_k) == 0L) next

          # Slice the k-th score column from the CPT (which has shape=(skill_dims_j, n_scores_j)), ending with shape (skill_dims_j)
          # then reshape to [1, *extend_shape_j] for broadcasting across students.
          log_cpt_k <- log_cpt_j[.., k]$reshape(c(1L, self$extend_shapes[[j]]))
          log_post[stus_k, ..] <- log_post[stus_k, ..]$add(log_cpt_k)
        }
      }
      #   normalize in log-space
      log_sum_i <- torch_logsumexp(log_post, dim = 2L:(1L+n_latent_skills), keepdim=T)
      normed_posteriors <- torch_exp(log_post$sub(log_sum_i))

      # Next, get the global statistics
      #   get the expected number of students at each skill level by summing over each student's Pr(latent level)
      Phi_I0 <- normed_posteriors$sum(dim = 1L)$add(self$bin_eps)

      # Finally, get the task-specific statistics
      Phi_Ijs <- NULL
      if (calc_for_tasks) {
        Phi_Ijs <- lapply(seq_len(self$n_tasks), function(j) {
          obs_j <- obs_index[[j]]
          n_scores <- length(self$scoring_states[[j]])
          marg_dims <- self$task_skill_dims[[j]]

          slices <- lapply(seq_len(n_scores), function(k) {
            stus_k <- obs_j[[k]]
            if (is.null(stus_k) || length(stus_k) == 0L) {
              # No student got this score: return a zero tensor of correct shape.
              keep_dims <- setdiff(seq_along(skill_dims), marg_dims)
              out_shape  <- if (length(keep_dims) > 0) skill_dims[keep_dims] else 1L
              return(torch_zeros(out_shape, device=self$device))
            }
            # marginalize out irrelevant skill dims
            stu_sum <- normed_posteriors[stus_k, .., drop = FALSE]$sum(dim = 1L)
            if (length(marg_dims) > 0)
              marginalize(stu_sum, dim = marg_dims)
            else
              stu_sum
          })

          # Stack along a new last dimension (score level dimension)
          torch_stack(slices, dim = length(dim(slices[[1]])) + 1L)
        })
      }

      list(Phi_I0 = Phi_I0, Phi_Ijs = Phi_Ijs)
    },
    get_evidence_CPTs = function() {
      lapply(self$evidence_models, function(ev_model) normalize_tensor(ev_model$getCPT()))
    },
    numparams = function () {
      sum(sapply(self$evidence_models, function(m) m$numparams()))
    },
    params = function() {
      plist <- do.call(c, lapply(self$evidence_models, function(m) m$params()))
      plist[!sapply(plist, is.null)]
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
    train = function(mode=TRUE) {
      super$train(mode=mode)
      for (ev_model in self$evidence_models)
        ev_model$train(mode)
    },
    buildOptimizer = function() {
      self$optimizer <-
        do.call(self$oconstructor, c(list(self$params()), self$oparams))
      self$optimizer
    },
    buildLossFn = function(phi_0, Phi_I0, Phi_Ijs) {
      # make sure that we are not accessing `self$` during the JIT trace
      penalties <- self$penalties
      ccbias <- self$ccbias

      # jit_trace needs to know the shapes of all phi_j
      example_phi_js <- self$get_evidence_CPTs()

      # The function signature matches what step() will call each iteration.
      raw_fn <- function(phi_js) {
        loss <- cdm_loss_fun(phi_0, Phi_I0, phi_js, Phi_Ijs, ccbias)
        for (ipar in names(penalties))
          loss <- loss$add(penalty_fun(params, ipar, penalties[[ipar]]))
        loss
      }

      self$lossfn <- jit_trace(raw_fn, example_phi_js)

      invisible(self)
    },
    step = function (phi_0, Phi_I0, Phi_Ijs, r=1L) {
      # initialize any unbuilt utilities
      if (is.null(self$optimizer))
        self$buildOptimizer()

      self$buildLossFn(phi_0, Phi_I0, Phi_Ijs)

      for (rr in 1:r) {
        # Invalidate CPT cache so each iteration builds a fresh graph
        # from the current parameter values. Without this, getCPT() returns
        # a view of the previous iteration's graph, which is freed after
        # its .backward() call.
        for (ev_model in self$evidence_models)
          ev_model$train(mode = TRUE)

        # actual training logic starts now
        self$optimizer$zero_grad()
        ev_cpts <- self$get_evidence_CPTs()
        step_loss <- self$lossfn(ev_cpts)
        step_loss$backward()
        self$optimizer$step()
      }

      final_loss <- NA
      with_no_grad({
        final_loss <- cdm_loss_fun(phi_0, Phi_I0, self$get_evidence_CPTs(),
                                   Phi_Ijs, self$ccbias)$item()
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

  # convert task_scores from a (sparse) matrix to list (of task) of small lists (students and their scores on that task)
  obs_index <- build_obs_index(task_scores)

  # setup training loop params
  rit <- 0L
  old_loss <- Inf
  loss_hist <- rep(NA_real_, maxit)
  optimizer <- model$buildOptimizer()
  scheduler <- lr_step(optimizer, step_size = half_life, gamma = 0.5)
  m_step_iters <- maxit_m_step

  while (rit < maxit) {
    # E step
    with_no_grad({
      exp_contin_tables <- model$get_expected_contingency_tables(task_scores)
      phi_0 <- exp_contin_tables$Phi_I0/n_stus
    })

    # M step
    model$train()
    final_loss <- model$step(phi_0, exp_contin_tables$Phi_I0,
                             exp_contin_tables$Phi_Ijs, r=m_step_iters)

    # display training info
    loss_hist[rit+1L] <- final_loss

    print(paste('TRAINING ITER', rit, '| LOSS', round(final_loss, 4), '| DELTA LOSS', round(old_loss-final_loss, 4)))
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

    if (abs(old_loss-final_loss) < tolerance)
      break
    old_loss <- final_loss
  }

  model$eval()

  print(paste0(rit,'/', maxit, ' iters taken'))
  print(paste0('min loss diff b/w iters ', min(loss_hist[1:(rit-1)]-loss_hist[2:rit], na.rm=T)))


  # plot the loss curve
  while (!is.null(dev.list())) dev.off()
  plot(1:rit, loss_hist[1:rit], 'l', main='Loss Curve', xlab='epoch', ylab='Loss (Perplexity + Penalties)')

  return (rit < maxit)
}
