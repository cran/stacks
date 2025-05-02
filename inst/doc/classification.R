## -----------------------------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## -----------------------------------------------------------------------------
# library(tidymodels)
# library(stacks)

## -----------------------------------------------------------------------------
library(tune)
library(rsample)
library(parsnip)
library(workflows)
library(recipes)
library(stacks)
library(purrr)
library(dplyr)
library(tidyr)
library(ggplot2)

## -----------------------------------------------------------------------------
if (rlang::is_installed("ranger") && 
    rlang::is_installed("nnet") &&
    rlang::is_installed("kernlab") &&
    rlang::is_installed("yardstick")) {
  run <- TRUE
  library(yardstick)
} else {
  run <- FALSE
}

knitr::opts_chunk$set(
  eval = run
)

## -----------------------------------------------------------------------------
# data("tree_frogs")
# 
# # subset the data
# tree_frogs <- tree_frogs |>
#   select(-c(clutch, latency))

## -----------------------------------------------------------------------------
# theme_set(theme_bw())
# 
# ggplot(tree_frogs) +
#   aes(x = treatment, y = age, color = reflex) +
#   geom_jitter() +
#   labs(
#     y = "Embryo Age (s)",
#     x = "treatment",
#     color = "Response"
#   )

## -----------------------------------------------------------------------------
# # some setup: resampling and a basic recipe
# set.seed(1)
# 
# tree_frogs_split <- initial_split(tree_frogs)
# tree_frogs_train <- training(tree_frogs_split)
# tree_frogs_test  <- testing(tree_frogs_split)
# 
# folds <- rsample::vfold_cv(tree_frogs_train, v = 5)
# 
# tree_frogs_rec <-
#   recipe(reflex ~ ., data = tree_frogs_train) |>
#   step_dummy(all_nominal_predictors(), -reflex) |>
#   step_zv(all_predictors())
# 
# tree_frogs_wflow <-
#   workflow() |>
#   add_recipe(tree_frogs_rec)

## -----------------------------------------------------------------------------
# ctrl_grid <- control_stack_grid()

## -----------------------------------------------------------------------------
# rand_forest_spec <-
#   rand_forest(
#     mtry = tune(),
#     min_n = tune(),
#     trees = 500
#   ) |>
#   set_mode("classification") |>
#   set_engine("ranger")
# 
# rand_forest_wflow <-
#   tree_frogs_wflow |>
#   add_model(rand_forest_spec)
# 
# rand_forest_res <-
#   tune_grid(
#     object = rand_forest_wflow,
#     resamples = folds,
#     grid = 10,
#     control = ctrl_grid
#   )

## -----------------------------------------------------------------------------
# nnet_spec <-
#   mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) |>
#   set_mode("classification") |>
#   set_engine("nnet")
# 
# nnet_rec <-
#   tree_frogs_rec |>
#   step_normalize(all_predictors())
# 
# nnet_wflow <-
#   tree_frogs_wflow |>
#   add_model(nnet_spec) |>
#   update_recipe(nnet_rec)
# 
# nnet_res <-
#   tune_grid(
#     object = nnet_wflow,
#     resamples = folds,
#     grid = 10,
#     control = ctrl_grid
#   )

## -----------------------------------------------------------------------------
# tree_frogs_model_st <-
#   # initialize the stack
#   stacks() |>
#   # add candidate members
#   add_candidates(rand_forest_res) |>
#   add_candidates(nnet_res) |>
#   # determine how to combine their predictions
#   blend_predictions() |>
#   # fit the candidates with nonzero stacking coefficients
#   fit_members()
# 
# tree_frogs_model_st

## -----------------------------------------------------------------------------
# autoplot(tree_frogs_model_st)

## -----------------------------------------------------------------------------
# autoplot(tree_frogs_model_st, type = "members")

## -----------------------------------------------------------------------------
# autoplot(tree_frogs_model_st, type = "weights")

## -----------------------------------------------------------------------------
# collect_parameters(tree_frogs_model_st, "rand_forest_res")

## -----------------------------------------------------------------------------
# tree_frogs_pred <-
#   tree_frogs_test |>
#   bind_cols(predict(tree_frogs_model_st, ., type = "prob"))

## -----------------------------------------------------------------------------
# yardstick::roc_auc(
#   tree_frogs_pred,
#   truth = reflex,
#   contains(".pred_")
#   )

## -----------------------------------------------------------------------------
# tree_frogs_pred <-
#   tree_frogs_test |>
#   select(reflex) |>
#   bind_cols(
#     predict(
#       tree_frogs_model_st,
#       tree_frogs_test,
#       type = "class",
#       members = TRUE
#       )
#     )
# 
# tree_frogs_pred
# 
# map(
#   colnames(tree_frogs_pred),
#   ~mean(tree_frogs_pred$reflex == pull(tree_frogs_pred, .x))
# ) |>
#   set_names(colnames(tree_frogs_pred)) |>
#   as_tibble() |>
#   pivot_longer(c(everything(), -reflex))

