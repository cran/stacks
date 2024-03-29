## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup, eval = FALSE------------------------------------------------------
#  library(tidymodels)
#  library(stacks)

## ----packages, include = FALSE------------------------------------------------
library(tune)
library(rsample)
library(parsnip)
library(workflows)
library(recipes)
library(yardstick)
library(stacks)
library(dplyr)
library(purrr)
library(ggplot2)

## ----include = FALSE----------------------------------------------------------
if (rlang::is_installed("ranger") && 
    rlang::is_installed("nnet") &&
    rlang::is_installed("kernlab")) {
  run <- TRUE
} else {
  run <- FALSE
}

knitr::opts_chunk$set(
  eval = run
)

## ----message = FALSE, warning = FALSE-----------------------------------------
data("tree_frogs")

# subset the data
tree_frogs <- tree_frogs %>%
  filter(!is.na(latency)) %>%
  select(-c(clutch, hatched))

## ----message = FALSE, warning = FALSE, fig.alt = "A ggplot scatterplot with embryo age in seconds on the x axis, time to hatch on the y axis, and points colored by treatment. The ages range from 350,000 to 500,000 seconds, while the times to hatch range from 0 to 450 seconds. There are two treatments—control and gentamicin—and the time to hatch is generally larger for the gentamicin group. The embryo ages are generally distributed in three clouds, where the older embryos tend to hatch more quickly after stimulus than the younger ones."----
theme_set(theme_bw())

ggplot(tree_frogs) +
  aes(x = age, y = latency, color = treatment) +
  geom_point() +
  labs(x = "Embryo Age (s)", y = "Time to Hatch (s)", col = "Treatment")

## ----echo = FALSE, fig.alt = "A diagram representing 'model definitions,' which specify the form of candidate ensemble members. Three colored boxes represent three different model types; a K-nearest neighbors model (in salmon), a linear regression model (in yellow), and a support vector machine model (in green)."----
knitr::include_graphics("https://raw.githubusercontent.com/tidymodels/stacks/main/man/figures/model_defs.png")

## -----------------------------------------------------------------------------
# some setup: resampling and a basic recipe
set.seed(1)
tree_frogs_split <- initial_split(tree_frogs)
tree_frogs_train <- training(tree_frogs_split)
tree_frogs_test  <- testing(tree_frogs_split)

set.seed(1)
folds <- rsample::vfold_cv(tree_frogs_train, v = 5)

tree_frogs_rec <- 
  recipe(latency ~ ., data = tree_frogs_train)

metric <- metric_set(rmse)

## -----------------------------------------------------------------------------
ctrl_grid <- control_stack_grid()
ctrl_res <- control_stack_resamples()

## -----------------------------------------------------------------------------
# create a model definition
knn_spec <-
  nearest_neighbor(
    mode = "regression", 
    neighbors = tune("k")
  ) %>%
  set_engine("kknn")

knn_spec

## -----------------------------------------------------------------------------
# extend the recipe
knn_rec <-
  tree_frogs_rec %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())

knn_rec

## -----------------------------------------------------------------------------
# add both to a workflow
knn_wflow <- 
  workflow() %>% 
  add_model(knn_spec) %>%
  add_recipe(knn_rec)

knn_wflow

## -----------------------------------------------------------------------------
# tune k and fit to the 5-fold cv
set.seed(2020)
knn_res <- 
  tune_grid(
    knn_wflow,
    resamples = folds,
    metrics = metric,
    grid = 4,
    control = ctrl_grid
  )

knn_res

## -----------------------------------------------------------------------------
# create a model definition
lin_reg_spec <-
  linear_reg() %>%
  set_engine("lm")

# extend the recipe
lin_reg_rec <-
  tree_frogs_rec %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

# add both to a workflow
lin_reg_wflow <- 
  workflow() %>%
  add_model(lin_reg_spec) %>%
  add_recipe(lin_reg_rec)

# fit to the 5-fold cv
set.seed(2020)
lin_reg_res <- 
  fit_resamples(
    lin_reg_wflow,
    resamples = folds,
    metrics = metric,
    control = ctrl_res
  )

lin_reg_res

## -----------------------------------------------------------------------------
# create a model definition
svm_spec <- 
  svm_rbf(
    cost = tune("cost"), 
    rbf_sigma = tune("sigma")
  ) %>%
  set_engine("kernlab") %>%
  set_mode("regression")

# extend the recipe
svm_rec <-
  tree_frogs_rec %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_corr(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# add both to a workflow
svm_wflow <- 
  workflow() %>% 
  add_model(svm_spec) %>%
  add_recipe(svm_rec)

# tune cost and sigma and fit to the 5-fold cv
set.seed(2020)
svm_res <- 
  tune_grid(
    svm_wflow, 
    resamples = folds, 
    grid = 6,
    metrics = metric,
    control = ctrl_grid
  )

svm_res

## ----echo = FALSE, fig.alt = "A diagram representing 'candidate members' generated from each model definition. Four salmon-colored boxes labeled 'KNN' represent K-nearest neighbors models trained on the resamples with differing hyperparameters. Similarly, the linear regression (LM) model generates one candidate member, and the support vector machine (SVM) model generates six."----
knitr::include_graphics("https://raw.githubusercontent.com/tidymodels/stacks/main/man/figures/candidates.png")

## ----echo = FALSE, fig.alt = "A diagram representing a 'data stack,' a specific kind of data frame. Colored 'columns' depict, in white, the true value of the outcome variable in the validation set, followed by four columns (in salmon) representing the predictions from the K-nearest neighbors model, one column (in tan) representing the linear regression model, and six (in green) representing the support vector machine model."----
knitr::include_graphics("https://raw.githubusercontent.com/tidymodels/stacks/main/man/figures/data_stack.png")

## -----------------------------------------------------------------------------
stacks()

## -----------------------------------------------------------------------------
tree_frogs_data_st <- 
  stacks() %>%
  add_candidates(knn_res) %>%
  add_candidates(lin_reg_res) %>%
  add_candidates(svm_res)

tree_frogs_data_st

## -----------------------------------------------------------------------------
as_tibble(tree_frogs_data_st)

## -----------------------------------------------------------------------------
tree_frogs_model_st <-
  tree_frogs_data_st %>%
  blend_predictions()

## ----echo = FALSE, fig.alt = "A diagram representing 'stacking coefficients,' the coefficients of the linear model combining each of the candidate member predictions to generate the ensemble's ultimate prediction. Boxes for each of the candidate members are placed besides each other, filled in with color if the coefficient for the associated candidate member is nonzero."----
knitr::include_graphics("https://raw.githubusercontent.com/tidymodels/stacks/main/man/figures/coefs.png")

## ----penalty-plot-------------------------------------------------------------
autoplot(tree_frogs_model_st)

## ----members-plot, fig.alt = "A ggplot line plot. The x axis shows the degree of penalization, ranging from 1e-06 to 1e-01, and the y axis displays the mean of three different metrics. The plots are faceted by metric type, with three facets: number of members, root mean squared error, and R squared. The plots generally show that, as penalization increases, the error decreases. There are very few proposed members in this example, so penalization doesn't drive down the number of members much at all. In this case, then, a larger penalty is acceptable."----
autoplot(tree_frogs_model_st, type = "members")

## ----weight-plot, fig.alt = "A ggplot bar plot, giving the stacking coefficient on the x axis and member on the y axis. There are three members in this ensemble, where a nearest neighbor is weighted most heavily, followed by a linear regression with a stacking coefficient about half as large, followed by a support vector machine with a very small contribution."----
autoplot(tree_frogs_model_st, type = "weights")

## -----------------------------------------------------------------------------
tree_frogs_model_st <-
  tree_frogs_model_st %>%
  fit_members()

## ----eval = FALSE, include = FALSE--------------------------------------------
#  st_print <- capture.output(print(tree_frogs_model_st))
#  
#  writeLines(st_print, con = "inst/figs/st_print.txt")

## ----echo = FALSE, fig.alt = "A diagram representing the ensemble members, where each are pentagons labeled and colored-in according to the candidate members they arose from."----
knitr::include_graphics("https://raw.githubusercontent.com/tidymodels/stacks/main/man/figures/members.png")

## ----echo = FALSE, fig.alt = "A diagram representing the 'model stack' class, which collates the stacking coefficients and members (candidate members with nonzero stacking coefficients that are trained on the full training set). The representation of the stacking coefficients and members is as before. Model stacks are a list subclass."----
knitr::include_graphics("https://raw.githubusercontent.com/tidymodels/stacks/main/man/figures/class_model_stack.png")

## -----------------------------------------------------------------------------
collect_parameters(tree_frogs_model_st, "svm_res")

## -----------------------------------------------------------------------------
tree_frogs_test <- 
  tree_frogs_test %>%
  bind_cols(predict(tree_frogs_model_st, .))

## ----fig.alt = "A ggplot scatterplot showing observed versus predicted latency values. While there is indeed a positive and roughly linear relationship, there is certainly patterned structure in the residuals."----
ggplot(tree_frogs_test) +
  aes(x = latency, 
      y = .pred) +
  geom_point() + 
  coord_obs_pred()

## -----------------------------------------------------------------------------
member_preds <- 
  tree_frogs_test %>%
  select(latency) %>%
  bind_cols(predict(tree_frogs_model_st, tree_frogs_test, members = TRUE))

## -----------------------------------------------------------------------------
map(member_preds, rmse_vec, truth = member_preds$latency) %>%
  as_tibble()

