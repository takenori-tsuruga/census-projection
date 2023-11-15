if (!require("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, data.table, parallel, foreach, doParallel, caret, progress, pracma)

num_cores <- detectCores()-1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# load training data
census_ds <- readRDS("data/census_training_prepped_sd_norm.rds")

# gathering column info
raw_colnames <- colnames(census_ds) %>% setdiff(c("year", "state", "place", "calib", "cluster_size", "cluster"))

# create lagged columns for number of steps specified in lookback
lookback <- 6
for (i in 1:lookback) {
  census_ds <- census_ds[, paste0(raw_colnames, "_lag_", i) := shift(.SD, i, type = "lag"), by = .(state, place), .SDcols = raw_colnames]
}
# add y columns for each value by looking 1 step ahead
census_ds <- census_ds[, paste0(raw_colnames, "_y") := shift(.SD, 1, type = "lead"), by = .(state, place), .SDcols = raw_colnames]

# separate training and validation/calibration data
train_ds <- census_ds[calib == 0 & cluster_size > 1, ]
calib_ds <- census_ds[calib == 1 & cluster_size > 1, ]

# get rid of cluster, calib, cluster_size columns
train_ds <- train_ds[, c("cluster", "calib", "cluster_size") := NULL]
calib_ds <- calib_ds[, c("cluster", "calib", "cluster_size") := NULL]
rm(census_ds)
gc()

# keep only the rows with year between 2017 and 2019 which contians the all the time seires sample slices
train_ds <- train_ds[year >= 2017 & year <= 2019, ]
calib_ds <- calib_ds[year >= 2017 & year <= 2019, ]

# gather information of features
raw_selected_vars <- colnames(train_ds)[colnames(train_ds) != c("year", "state", "place")]
raw_selected_vars <- raw_selected_vars[!grepl("_lag_", raw_selected_vars)]
raw_selected_vars <- raw_selected_vars[!grepl("_y", raw_selected_vars)]

# save raw_selected_vars to as rds for later use
saveRDS(raw_selected_vars, "data/raw_selected_vars.RDS")

ema_process <- function(ds) {
  mas <- foreach(j = seq(nrow(ds))) %dopar% {
    library(pracma)
    library(magrittr)
    mas <- movavg(as.numeric(ds[j, ]), n = 2, type = "e") %>% tail(5)
    mas
  }
  return(mas)
}

pb <- progress_bar$new(
  format = "(:spin) [:bar] :percent [Elapsed time: :elapsedfull || Estimated time remaining: :eta]",
  total = length(raw_selected_vars),
  complete = "=", # Completion bar character
  incomplete = "-", # Incomplete bar character
  current = ">", # Current bar character
  clear = FALSE, # If TRUE, clears the bar when finish
  width = 100)

for (i in raw_selected_vars) {
  target_cols <- grep(i, colnames(train_ds), value = TRUE) %>% setdiff(grep("_y", ., value = TRUE))
  target_cols_rev <- rev(target_cols)
  tmp_ds_1 <- train_ds[, target_cols_rev, with = FALSE]
  tmp_ds_2 <- calib_ds[, target_cols_rev, with = FALSE]
  train_mas <- ema_process(tmp_ds_1)
  calib_mas <- ema_process(tmp_ds_2)
  train_ds[, paste0(i, "_ema_5") := unlist(train_mas)[seq(1, length(train_mas) * 5, 5)]]
  train_ds[, paste0(i, "_ema_4") := unlist(train_mas)[seq(2, length(train_mas) * 5, 5)]]
  train_ds[, paste0(i, "_ema_3") := unlist(train_mas)[seq(3, length(train_mas) * 5, 5)]]
  train_ds[, paste0(i, "_ema_2") := unlist(train_mas)[seq(4, length(train_mas) * 5, 5)]]
  train_ds[, paste0(i, "_ema_1") := unlist(train_mas)[seq(5, length(train_mas) * 5, 5)]]
  calib_ds[, paste0(i, "_ema_5") := unlist(calib_mas)[seq(1, length(calib_mas) * 5, 5)]]
  calib_ds[, paste0(i, "_ema_4") := unlist(calib_mas)[seq(2, length(calib_mas) * 5, 5)]]
  calib_ds[, paste0(i, "_ema_3") := unlist(calib_mas)[seq(3, length(calib_mas) * 5, 5)]]
  calib_ds[, paste0(i, "_ema_2") := unlist(calib_mas)[seq(4, length(calib_mas) * 5, 5)]]
  calib_ds[, paste0(i, "_ema_1") := unlist(calib_mas)[seq(5, length(calib_mas) * 5, 5)]]
  pb$tick()
}

saveRDS(train_ds, "data/train_ds_ema.RDS")
saveRDS(calib_ds, "data/calib_ds_ema.RDS")

stopCluster(cl)
