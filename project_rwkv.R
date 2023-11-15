if (!require("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(keras, tidyverse, data.table, tensorflow, reticulate)

# tensorflow confifuration
devices <- tf$config$get_visible_devices()
tf$config$experimental$set_memory_growth(devices[[2]], TRUE)
# tf$config$set_visible_devices(devices[[1]])
tf$config$threading$set_intra_op_parallelism_threads(16L)
tf$config$threading$set_inter_op_parallelism_threads(16L)

### qhat calculation for conformal prediction
census_ds <- readRDS("data/census_training_prepped_std.rds")

# pull where calib == 1 and cluster_size > 1 as calib_ds
calib_ds <- census_ds[calib == 1 & cluster_size > 1, ]
calib_ds <- calib_ds[year >= 2011 & year <= 2019, ]
# remove census_ds from memory
rm(census_ds)

mean_value <- readRDS("data/mean_value_std.rds")
sd_value <- readRDS("data/sd_value_std.rds")
zero_value <- readRDS("data/zero_value_std.rds")
zero_value[length(zero_value)] <- -Inf

raw_colnames <- colnames(calib_ds) %>% setdiff(c("year", "state", "place", "calib", "cluster_size", "cluster"))

# generator function for prediction
generator <- function(data, lookback, delay, min_index, max_index, shuffle = Flase, batch_size = 128, step = 6, n_input, n_output, col_range) {
  index <- min_index
  random_range <- min_index:max_index
  function() {
    if (shuffle) {
      if (length(random_range) < batch_size) {
        random_range <<- min_index:max_index
      }
      rows <- sample(random_range, batch_size)
      random_range <<- setdiff(random_range, rows)
    } else {
      if (index > max_index) {
        index <<- min_index
      }
      if (index + batch_size - 1 > max_index) {
        rows <- seq(index, max_index)
        rows <- append(rows, seq(min_index, (min_index + batch_size - length(rows))))
      } else {
        rows <- seq(index, min(index + batch_size - 1, max_index))
      }
      index <<- index + batch_size
    }
    lookback_count <- lookback / step
    x <- array(rep(0, batch_size * (lookback_count + 1) * n_input), dim = c(batch_size, (lookback_count + 1), n_input))
    y <- array(rep(0, batch_size * n_output), dim = c(batch_size, n_output))

    for (j in 1:batch_size) {
      row_info <- data[rows[j], .(year, state, place)]
      indices <- data[, .I[year <= (row_info[1, year] + 1) & year >= (row_info[1, year] - lookback) & state == row_info[1, state] & place == row_info[1, place]]]
      while (length(indices) <= (lookback + 1)) {
        row_info <- data[sample(random_range, 1), .(year, state, place)]
        indices <- data[, .I[year <= (row_info[1, year] + 1) & year >= (row_info[1, year] - lookback) & state == row_info[1, state] & place == row_info[1, place]]]
      }
      x[j, , ] <- as.matrix(data[indices[1:(1 + lookback_count)], col_range, with = F])
      y[j, ] <- as.matrix(data[indices[(2 + lookback_count)], col_range, with = F])
    }
    list(x, y)
  }
}

# pull the latest place list in reiverside county
place_list <- pull(calib_ds[year == 2019, ], place)
# for each place in place_list, count the row number of the place in the data.table rivco_ds
place_count <- calib_ds[place %in% place_list, .N, by = place]
# keep the place with place_count more than or equal to 9
place_list <- place_count[N >= 9, place]
# update rivco_ds with the new place_list
calib_ds <- calib_ds[place %in% place_list, ]

# Setting up for training
lookback <- 6
step <- 1
delay <- 1
batch_size <- nrow(calib_ds[year == 2019, ]) * 3
n_input <- length(raw_colnames)
n_output <- length(raw_colnames)
col_offset <- 3
pred_col_range <- seq(col_offset + 1, col_offset + n_output)

# find minimal index where year ==  2017
pred_min_index <- which(calib_ds[, year] == 2017)[1]
# find maximal index where year ==  2019
pred_max_index <- which(calib_ds[, year] == 2019)[length(which(calib_ds[, year] == 2019))]


calib_gen <- generator(
    calib_ds,
    lookback = lookback,
    min_index = pred_min_index,
    max_index = pred_max_index,
    shuffle = FALSE,
    batch_size = batch_size,
    step = step,
    n_input = n_input,
    n_output = n_output,
    col_range = pred_col_range
)

calib_pred_x <- calib_gen()


source_python("tf_rwkv_model.py")
model <- tf$keras$models$load_model("models/rwkv_50_model/skel.keras")


model$load_weights(paste0("models/rwkv_51_model/epc_cpt"))


calib_pred <- predict(
    model,
    calib_pred_x[[1]],
    steps = 1,
    verbose = 1
)

pred_score_51 <- calib_pred_x[[2]] - calib_pred

model$load_weights(paste0("models/rwkv_49_model/epc_cpt"))

calib_pred <- predict(
    model,
    calib_pred_x[[1]],
    steps = 1,
    verbose = 1
)

pred_score_49 <- calib_pred - calib_pred_x[[2]]

# get element wise max of pred_score_51 and pred_score_49
calib_score <- pmax(pred_score_51, pred_score_49)

# apply above quantile function to each column of calib_score
qhat <- apply(calib_score, 2, function(x) quantile(x, probs = (length(x) + 1) * (1 - 0.49) / length(x)))

saveRDS(qhat, "data/rwkv_qhat_values.RDS")


## projection generation

# load data for prediction
rivco_ds <- readRDS("data/census_predbase_prepped_std.rds")

raw_colnames <- colnames(rivco_ds) %>% setdiff(c("year", "state", "place", "calib", "cluster_size", "cluster"))

# make sure placed in data has data for all years from 2015 to 2021 (7 years)
# pull the latest place list in reiverside county
place_list <- pull(rivco_ds[year == 2021, ], place)
# for each place in place_list, count the row number of the place in the data.table rivco_ds
place_count <- rivco_ds[place %in% place_list, .N, by = place]
# keep the place with place_count more than or equal to 9
place_list <- place_count[N >= 7, place]
# update rivco_ds with the new place_list
rivco_ds <- rivco_ds[place %in% place_list, ]


rivco_ds_backup <- copy(rivco_ds)

# Setting up for training
lookback <- 6
step <- 1
delay <- 1
batch_size <- nrow(rivco_ds[year == 2021, ])
n_input <- length(raw_colnames)
n_output <- length(raw_colnames)
col_offset <- 3
pred_col_range <- seq(col_offset + 1, col_offset + n_output)

# source_python("tf_rwkv_model.py")
model <- tf$keras$models$load_model("models/rwkv_50_model/skel.keras")
model$load_weights(paste0("models/rwkv_50_model/epc_cpt"))

## Prediction loop for mid quantile
for (y in seq(2022, 2060, by = 1)) {
    print(paste0("generating projection for year ", y))
    # find minimal index where year ==  y - 1
    pred_min_index <- which(rivco_ds[, year] == y - 1)[1]
    # find maximal index where year ==  y - 1
    pred_max_index <- which(rivco_ds[, year] == y - 1)[length(which(rivco_ds[, year] == y - 1))]

    # expand data frame to be filled with prediction
    tmp_tbl <- data.table(
        year = rep(y, batch_size),
        state = rep(6, batch_size),
        place = place_list %>% sort()
    )
    rivco_ds <- bind_rows(rivco_ds, tmp_tbl)

    # initialize a generator for prediction
    pred_gen <- generator(
        rivco_ds,
        lookback = lookback,
        # horizon = horizon,
        min_index = pred_min_index,
        max_index = pred_max_index,
        shuffle = FALSE,
        batch_size = batch_size,
        step = step,
        n_input = n_input,
        n_output = n_output,
        col_range = pred_col_range
    )

    # generate data for prediction
    pred_x <- pred_gen()

    # prediction
    pred <- model(
        pred_x[[1]],
        training = FALSE
    )

    pred <- as.matrix(pred)

    # make sure the prediction are not negative except for the last column
    for (i in 1:(ncol(pred) - 1)) {
        if (any(pred[, i] < zero_value[i])) {
            pred[pred[, i] < zero_value[i], i] <- zero_value[i]
        }
    }

    # fill in the prediction
    pred <- as.data.table(pred)
    colnames(pred) <- raw_colnames
    rivco_ds[year == y, pred_col_range] <- pred

    rm(tmp_tbl)
}


revert_std <- function(x, sd, mean) {
    x <- x * sd + mean
    return(x)
}


# revert standardization
for (i in raw_colnames) {
    rivco_ds[, paste0(i) := lapply(.SD, revert_std, sd = sd_value[i], mean = mean_value[i]), .SDcols = c(paste0(i))]
}

saveRDS(rivco_ds, paste0("data/rwkv_projection_50.RDS"))

# Prediction loop for lower quantile
rivco_ds <- copy(rivco_ds_backup)
model$load_weights(paste0("models/rwkv_49_model/epc_cpt"))
for (y in seq(2022, 2060, by = 1)) {
    print(paste0("generating projection for year ", y))
    # find minimal index where year ==  y - 1
    pred_min_index <- which(rivco_ds[, year] == y - 1)[1]
    # find maximal index where year ==  y - 1
    pred_max_index <- which(rivco_ds[, year] == y - 1)[length(which(rivco_ds[, year] == y - 1))]

    # expand data frame to be filled with prediction
    tmp_tbl <- data.table(
        year = rep(y, batch_size),
        state = rep(6, batch_size),
        place = place_list %>% sort()
    )
    rivco_ds <- bind_rows(rivco_ds, tmp_tbl)

    # initialize a generator for prediction
    pred_gen <- generator(
        rivco_ds,
        lookback = lookback,
        # horizon = horizon,
        min_index = pred_min_index,
        max_index = pred_max_index,
        shuffle = FALSE,
        batch_size = batch_size,
        step = step,
        n_input = n_input,
        n_output = n_output,
        col_range = pred_col_range
    )

    # generate data for prediction
    pred_x <- pred_gen()

    # prediction
    pred <- model(
        pred_x[[1]],
        training = FALSE
    )

    pred <- as.matrix(pred)

    # subtract qhat from the prediction
    pred <- pred - qhat

    # make sure the prediction are not negative except for the last column
    for (i in 1:(ncol(pred) - 1)) {
        if (any(pred[, i] < zero_value[i])) {
            pred[pred[, i] < zero_value[i], i] <- zero_value[i]
        }
    }

    # fill in the prediction
    pred <- as.data.table(pred)
    colnames(pred) <- raw_colnames
    rivco_ds[year == y, pred_col_range] <- pred

    rm(tmp_tbl)
}


revert_std <- function(x, sd, mean) {
    x <- x * sd + mean
    return(x)
}


# revert standardization
for (i in raw_colnames) {
    rivco_ds[, paste0(i) := lapply(.SD, revert_std, sd = sd_value[i], mean = mean_value[i]), .SDcols = c(paste0(i))]
}

saveRDS(rivco_ds, paste0("data/rwkv_projection_49.RDS"))

# Prediction loop for upper quantile
rivco_ds <- copy(rivco_ds_backup)
model$load_weights(paste0("models/rwkv_51_model/epc_cpt"))
for (y in seq(2022, 2060, by = 1)) {
    print(paste0("generating projection for year ", y))
    # find minimal index where year ==  y - 1
    pred_min_index <- which(rivco_ds[, year] == y - 1)[1]
    # find maximal index where year ==  y - 1
    pred_max_index <- which(rivco_ds[, year] == y - 1)[length(which(rivco_ds[, year] == y - 1))]

    # expand data frame to be filled with prediction
    tmp_tbl <- data.table(
        year = rep(y, batch_size),
        state = rep(6, batch_size),
        place = place_list %>% sort()
    )
    rivco_ds <- bind_rows(rivco_ds, tmp_tbl)

    # initialize a generator for prediction
    pred_gen <- generator(
        rivco_ds,
        lookback = lookback,
        # horizon = horizon,
        min_index = pred_min_index,
        max_index = pred_max_index,
        shuffle = FALSE,
        batch_size = batch_size,
        step = step,
        n_input = n_input,
        n_output = n_output,
        col_range = pred_col_range
    )

    # generate data for prediction
    pred_x <- pred_gen()

    # prediction
    pred <- model(
        pred_x[[1]],
        training = FALSE
    )

    pred <- as.matrix(pred)

    # add qhat to the prediction
    pred <- pred + qhat

    # make sure the prediction are not negative except for the last column
    for (i in 1:(ncol(pred) - 1)) {
        if (any(pred[, i] < zero_value[i])) {
            pred[pred[, i] < zero_value[i], i] <- zero_value[i]
        }
    }

    # fill in the prediction
    pred <- as.data.table(pred)
    colnames(pred) <- raw_colnames
    rivco_ds[year == y, pred_col_range] <- pred

    rm(tmp_tbl)
}


revert_std <- function(x, sd, mean) {
    x <- x * sd + mean
    return(x)
}


# revert standardization
for (i in raw_colnames) {
    rivco_ds[, paste0(i) := lapply(.SD, revert_std, sd = sd_value[i], mean = mean_value[i]), .SDcols = c(paste0(i))]
}

saveRDS(rivco_ds, paste0("data/rwkv_projection_51.RDS"))
