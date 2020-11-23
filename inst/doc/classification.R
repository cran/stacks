## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup, eval = FALSE------------------------------------------------------
#  library(tidymodels)
#  library(tidyverse)
#  library(stacks)

## ----packages, include = FALSE------------------------------------------------
library(tune)
library(rsample)
library(parsnip)
library(workflows)
library(recipes)
library(yardstick)
library(stacks)
library(purrr)
library(dplyr)
library(tidyr)

## ---- message = FALSE, warning = FALSE----------------------------------------
data("tree_frogs")

# subset the data
tree_frogs <- tree_frogs %>%
  select(-c(clutch, latency))

## ---- message = FALSE, warning = FALSE----------------------------------------
library(ggplot2)

ggplot(tree_frogs) +
  aes(x = treatment, y = age, color = reflex) +
  geom_jitter() +
  labs(y = "Embryo Age (s)", 
       x = "treatment",
       color = "Response")

## -----------------------------------------------------------------------------
# some setup: resampling and a basic recipe
set.seed(1)

tree_frogs_split <- initial_split(tree_frogs)
tree_frogs_train <- training(tree_frogs_split)
tree_frogs_test  <- testing(tree_frogs_split)

folds <- rsample::vfold_cv(tree_frogs_train, v = 5)

tree_frogs_rec <- 
  recipe(reflex ~ ., data = tree_frogs_train) %>%
  step_dummy(all_nominal(), -reflex) %>%
  step_zv(all_predictors())

tree_frogs_wflow <- 
  workflow() %>% 
  add_recipe(tree_frogs_rec)

## -----------------------------------------------------------------------------
ctrl_grid <- control_stack_grid()

## ---- message = FALSE, warning = FALSE----------------------------------------
rand_forest_spec <- 
  rand_forest(
    mtry = tune(),
    min_n = tune(),
    trees = 500
  ) %>%
  set_mode("classification") %>%
  set_engine("ranger")

rand_forest_wflow <-
  tree_frogs_wflow %>%
  add_model(rand_forest_spec)

rand_forest_res <- 
  tune_grid(
    object = rand_forest_wflow, 
    resamples = folds, 
    grid = 10,
    control = ctrl_grid
  )

## ---- message = FALSE, warning = FALSE----------------------------------------
nnet_spec <-
  mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>%
  set_mode("classification") %>%
  set_engine("nnet")

nnet_rec <- 
  tree_frogs_rec %>% 
  step_normalize(all_predictors())

nnet_wflow <- 
  tree_frogs_wflow %>%
  add_model(nnet_spec)

nnet_res <-
  tune_grid(
    object = nnet_wflow, 
    resamples = folds, 
    grid = 10,
    control = ctrl_grid
  )

## ---- message = FALSE, warning = FALSE----------------------------------------
tree_frogs_model_st <- 
  # initialize the stack
  stacks() %>%
  # add candidate members
  add_candidates(rand_forest_res) %>%
  add_candidates(nnet_res) %>%
  # determine how to combine their predictions
  blend_predictions() %>%
  # fit the candidates with nonzero stacking coefficients
  fit_members()

tree_frogs_model_st

## ----penalty-plot-------------------------------------------------------------
theme_set(theme_bw())
autoplot(tree_frogs_model_st)

## ----members-plot-------------------------------------------------------------
autoplot(tree_frogs_model_st, type = "members")

## ----weight-plot--------------------------------------------------------------
autoplot(tree_frogs_model_st, type = "weights")

## -----------------------------------------------------------------------------
collect_parameters(tree_frogs_model_st, "rand_forest_res")

## ---- eval = FALSE------------------------------------------------------------
#  tree_frogs_pred <-
#    tree_frogs_test %>%
#    bind_cols(predict(tree_frogs_model_st, ., type = "prob"))

## ---- eval = FALSE------------------------------------------------------------
#  yardstick::roc_auc(
#    tree_frogs_pred,
#    truth = reflex,
#    contains(".pred_")
#    )

## -----------------------------------------------------------------------------
tree_frogs_pred <-
  tree_frogs_test %>%
  select(reflex) %>%
  bind_cols(
    predict(
      tree_frogs_model_st,
      tree_frogs_test,
      type = "class",
      members = TRUE
      )
    )

tree_frogs_pred

map_dfr(
  setNames(colnames(tree_frogs_pred), colnames(tree_frogs_pred)),
  ~mean(tree_frogs_pred$reflex == pull(tree_frogs_pred, .x))
) %>%
  pivot_longer(c(everything(), -reflex))

