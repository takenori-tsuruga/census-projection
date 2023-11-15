if (!require("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, data.table, caret, progress, quantregForest)


# check if data directory exists, if not, create one
if (!dir.exists("models")) {
  dir.create("models")
}

# check if data directory exists, if not, create one
if (!dir.exists("models/rf")) {
  dir.create("models/rf")
}

# load training and validation/calibration data
raw_selected_vars <- readRDS("data/raw_selected_vars.RDS")
train_ds <- readRDS("data/train_ds_ema.RDS")
calib_ds <- readRDS("data/calib_ds_ema.RDS")

# selection of variables
remove_vars <- grep("S0101_C01_001E", raw_selected_vars)
raw_selected_vars <- raw_selected_vars[-remove_vars]
select_vars <- grep("S0101_C01|S0701_C06_001E|S1301_C01_001E|S1301_C02_001E|dn", raw_selected_vars)
raw_selected_vars <- raw_selected_vars[select_vars]

remove_cols <- grep("S0101_C01_001E", colnames(train_ds))
train_ds <- train_ds[, -remove_cols, with = FALSE]
calib_ds <- calib_ds[, -remove_cols, with = FALSE]
select_cols <- grep("S0101_C01|S0701_C06_001E|S1301_C01_001E|S1301_C02_001E|dn", colnames(train_ds))
train_ds <- train_ds[, select_cols, with = FALSE]
calib_ds <- calib_ds[, select_cols, with = FALSE]

# initialize progress bar for training loop progress
pg <- progress_bar$new(
  format = "(:spin) [:bar] :percent [Elapsed time: :elapsedfull || Estimated time remaining: :eta]",
  total = length(raw_selected_vars),
  complete = "=", # Completion bar character
  incomplete = "-", # Incomplete bar character
  current = ">", # Current bar character
  clear = FALSE, # If TRUE, clears the bar when finish
  width = 100,
  show_after = 0
)

# training loop for each variable
# raw_selected_vars_loop <- raw_selected_vars[which(raw_selected_vars == "S0101_C01_014E"):length(raw_selected_vars)] %>% na.omit()
raw_selected_vars_loop <- raw_selected_vars
for (target_var in raw_selected_vars_loop) {
  pg$message(paste0("Training Random Forest model for ", target_var))

  # initialization of results storage list
  save_list <- list()
  pred_score <- list()

  # selection of variables for training
  x <- train_ds[, .SD, .SDcols = !c(grep("_y", colnames(train_ds), value = TRUE))]
  x_1 <- x %>%
      select(matches(target_var))
  x_2 <- x %>%
    select(-matches(target_var)) %>%
    select(-matches("_lag_")) %>%
    select(-matches("ema"))
  x <- cbind(x_1, x_2) %>% as.matrix()

  # selection of variables for validation/calibration
  val_x <- calib_ds[, .SD, .SDcols = !c(grep("_y", colnames(calib_ds), value = TRUE))]
  val_x_1 <- val_x %>%
    select(matches(target_var))
  val_x_2 <- val_x %>%
    select(-matches(target_var)) %>%
    select(-matches("_lag_")) %>%
    select(-matches("ema"))
  val_x <- cbind(val_x_1, val_x_2) %>% as.matrix()

  # selection of target variable for training
  y <- train_ds[[paste0(target_var, "_y")]] %>% as.matrix()
  # selection of target variable for validation/calibration
  val_y <- calib_ds[[paste0(target_var, "_y")]] %>% as.matrix()

  # train random forest model
  set.seed(123)
  model <- quantregForest(
    x = x,
    y = y,
    nthreads = 16
  )

  # prediction error recordings for each alpha
  pred <- predict(model, val_x, what = 0.05)
  pred_score[[paste0("alpha_0.05")]] <- pred - val_y
  pred <- predict(model, val_x, what = 0.1)
  pred_score[[paste0("alpha_0.1")]] <- pred - val_y
  pred <- predict(model, val_x, what = 0.15)
  pred_score[[paste0("alpha_0.15")]] <- pred - val_y
  pred <- predict(model, val_x, what = 0.2)
  pred_score[[paste0("alpha_0.2")]] <- pred - val_y
  pred <- predict(model, val_x, what = 0.25)
  pred_score[[paste0("alpha_0.25")]] <- pred - val_y
  pred <- predict(model, val_x, what = 0.3)
  pred_score[[paste0("alpha_0.3")]] <- pred - val_y
  pred <- predict(model, val_x, what = 0.35)
  pred_score[[paste0("alpha_0.35")]] <- pred - val_y
  pred <- predict(model, val_x, what = 0.4)
  pred_score[[paste0("alpha_0.4")]] <- pred - val_y
  pred <- predict(model, val_x, what = 0.45)
  pred_score[[paste0("alpha_0.45")]] <- pred - val_y

  pred <- predict(model, val_x, what = 0.55)
  pred_score[[paste0("alpha_0.55")]] <- val_y - pred
  pred <- predict(model, val_x, what = 0.6)
  pred_score[[paste0("alpha_0.6")]] <- val_y - pred
  pred <- predict(model, val_x, what = 0.65)
  pred_score[[paste0("alpha_0.65")]] <- val_y - pred
  pred <- predict(model, val_x, what = 0.7)
  pred_score[[paste0("alpha_0.7")]] <- val_y - pred
  pred <- predict(model, val_x, what = 0.75)
  pred_score[[paste0("alpha_0.75")]] <- val_y - pred
  pred <- predict(model, val_x, what = 0.8)
  pred_score[[paste0("alpha_0.8")]] <- val_y - pred
  pred <- predict(model, val_x, what = 0.85)
  pred_score[[paste0("alpha_0.85")]] <- val_y - pred
  pred <- predict(model, val_x, what = 0.9)
  pred_score[[paste0("alpha_0.9")]] <- val_y - pred
  pred <- predict(model, val_x, what = 0.95)
  pred_score[[paste0("alpha_0.95")]] <- val_y - pred

  pred <- predict(model, val_x, what = 0.5)
  eval <- postResample(pred, val_y)
  save_list[["rf_50_eval"]] <- eval

  # save model
  save_list[[paste0("rf_model")]] <- model
  # calculate calibration score
  calib_score <- list()
  calib_score[[paste0("alpha_0.05")]] <- pmax(pred_score$alpha_0.05, pred_score$alpha_0.95)
  calib_score[[paste0("alpha_0.1")]] <- pmax(pred_score$alpha_0.1, pred_score$alpha_0.9)
  calib_score[[paste0("alpha_0.15")]] <- pmax(pred_score$alpha_0.15, pred_score$alpha_0.85)
  calib_score[[paste0("alpha_0.2")]] <- pmax(pred_score$alpha_0.2, pred_score$alpha_0.8)
  calib_score[[paste0("alpha_0.25")]] <- pmax(pred_score$alpha_0.25, pred_score$alpha_0.75)
  calib_score[[paste0("alpha_0.3")]] <- pmax(pred_score$alpha_0.3, pred_score$alpha_0.7)
  calib_score[[paste0("alpha_0.35")]] <- pmax(pred_score$alpha_0.35, pred_score$alpha_0.65)
  calib_score[[paste0("alpha_0.4")]] <- pmax(pred_score$alpha_0.4, pred_score$alpha_0.6)
  calib_score[[paste0("alpha_0.45")]] <- pmax(pred_score$alpha_0.45, pred_score$alpha_0.55)
  # calculation of qhat values
  qhat <- list()
  qhat[[paste0("alpha_0.05")]] <- quantile(calib_score$alpha_0.05, probs = (length(calib_score$alpha_0.05) + 1) * (1 - 0.05) / length(calib_score$alpha_0.05))
  qhat[[paste0("alpha_0.1")]] <- quantile(calib_score$alpha_0.1, probs = (length(calib_score$alpha_0.1) + 1) * (1 - 0.1) / length(calib_score$alpha_0.1))
  qhat[[paste0("alpha_0.15")]] <- quantile(calib_score$alpha_0.15, probs = (length(calib_score$alpha_0.15) + 1) * (1 - 0.15) / length(calib_score$alpha_0.15))
  qhat[[paste0("alpha_0.2")]] <- quantile(calib_score$alpha_0.2, probs = (length(calib_score$alpha_0.2) + 1) * (1 - 0.2) / length(calib_score$alpha_0.2))
  qhat[[paste0("alpha_0.25")]] <- quantile(calib_score$alpha_0.25, probs = (length(calib_score$alpha_0.25) + 1) * (1 - 0.25) / length(calib_score$alpha_0.25))
  qhat[[paste0("alpha_0.3")]] <- quantile(calib_score$alpha_0.3, probs = (length(calib_score$alpha_0.3) + 1) * (1 - 0.3) / length(calib_score$alpha_0.3))
  qhat[[paste0("alpha_0.35")]] <- quantile(calib_score$alpha_0.35, probs = (length(calib_score$alpha_0.35) + 1) * (1 - 0.35) / length(calib_score$alpha_0.35))
  qhat[[paste0("alpha_0.4")]] <- quantile(calib_score$alpha_0.4, probs = (length(calib_score$alpha_0.4) + 1) * (1 - 0.4) / length(calib_score$alpha_0.4))
  qhat[[paste0("alpha_0.45")]] <- quantile(calib_score$alpha_0.45, probs = (length(calib_score$alpha_0.45) + 1) * (1 - 0.45) / length(calib_score$alpha_0.45))

  # store all results in save_list
  save_list[["qhat"]] <- qhat
  save_list[["pred_score"]] <- pred_score
  save_list[["calib_score"]] <- calib_score
  # save save_list to RDS for later prediction use
  saveRDS(save_list, paste0("models/rf/", target_var, "_rf.RDS"))
  pg$tick()
  rm(save_list)
  gc()
}
