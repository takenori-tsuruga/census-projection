if (!require("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, data.table, parallel, foreach, doParallel, pracma, matrixStats, progress, quantregForest, keras, tensorflow, reticulate)

num_cores <- detectCores()-1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

### random forest prediction on 2018, 2019, 2020 data in Riverside County

# load prepped data
rivco_ds <- readRDS("data/census_predbase_prepped_sd_norm.rds")
sd_value <- readRDS("data/sd_value_sd_norm.rds")
# make use of zero value from standardization method just for frame
zero_value <- readRDS("data/zero_value_std.rds")
zero_value <- zero_value * 0
zero_value[length(zero_value)] <- -Inf

# feature selection, prep both dataset and zero_value
remove_cols <- grep("S0101_C01_001E", colnames(rivco_ds))
rivco_ds <- rivco_ds[, -remove_cols, with = FALSE]
select_cols <- grep("year|state|place|S0101_C01|S0701_C06_001E|S1301_C01_001E|S1301_C02_001E|dn", colnames(rivco_ds))
rivco_ds <- rivco_ds[, select_cols, with = FALSE]

remove_cols <- grep("S0101_C01_001E", names(zero_value))
zero_value <- zero_value[-remove_cols]
select_cols <- grep("S0101_C01|S0701_C06_001E|S1301_C01_001E|S1301_C02_001E|dn", names(zero_value))
zero_value <- zero_value[select_cols]

# make sure all the time slice is available for each place
# pull the latest place list in reiverside county
place_list <- pull(rivco_ds[year == 2021, ], place)
# for each place in place_list, count the row number of the place in the data.table rivco_ds
place_count <- rivco_ds[place %in% place_list, .N, by = place]
# keep the place with place_count more than or equal to 9
place_list <- place_count[N >= 11, place]
# update rivco_ds with the new place_list
rivco_ds <- rivco_ds[place %in% place_list, ]


# left_edge_index
lei <- which(colnames(rivco_ds) == "S0101_C01_002E")
# right_edge_index
rei <- which(colnames(rivco_ds) == "dn")

raw_colnames <- colnames(rivco_ds[, lei:rei, with = FALSE])
percentiles <- seq(from = 50, to = 50, by = 10)
colnames_percentiles_list <- list()
for (p in percentiles) {
    colnames_percentiles_list[[paste0("colnames_", p, "p")]] <- paste0(raw_colnames, "_", p, "p")
}

for (p in percentiles) {
    rivco_ds[, colnames_percentiles_list[[paste0("colnames_", p, "p")]]] <- rivco_ds[, lei:rei, with = FALSE]
}

# Setting up for training
lookback <- 6
batch_size <- nrow(rivco_ds[year == 2021, ])
n_feature <- length(lei:rei)
col_offset <- 3
pred_col_range <- seq(col_offset + 1, col_offset + n_feature)
col_range_list <- list()
base_colrange <- pred_col_range
for (p in percentiles) {
    col_range_list[[paste0("col_range_", p, "p")]] <- seq(max(base_colrange) + 1, max(base_colrange) + n_feature)
    base_colrange <- col_range_list[[paste0("col_range_", p, "p")]]
}

# function for ema augmentation
ema_process <- function(ds) {
  mas <- foreach(j = seq(nrow(ds))) %dopar% {
    library(pracma)
    library(magrittr)
    mas <- movavg(as.numeric(ds[j, ]), n = 2, type = "e") %>% tail(5)
    mas
  }
  return(mas)
}

# function for training feature preparation
x_prep <- function(x, target_var) {
    library(magrittr)
    library(dplyr)
    x_1 <- x %>% 
        select(matches(target_var))
    x_2 <- x %>%
        select(-matches(target_var)) %>%
        select(-matches("_lag_")) %>%
        select(-matches("ema"))
    x <- cbind(x_1, x_2) %>% as.matrix()
    return(x)
}


for (y in seq(2018, 2020)) {

    lagged_rivco_ds <- rivco_ds[year %in% seq(y - (lookback + 1), y - 1), ]
    lagged_raw_colnames <- colnames(lagged_rivco_ds) %>% setdiff(c("year", "state", "place"))
    for (i in 1:lookback) {
        lagged_rivco_ds <- lagged_rivco_ds[, paste0(lagged_raw_colnames, "_lag_", i) := shift(.SD, i, type = "lag"), by = .(state, place), .SDcols = lagged_raw_colnames]
    }
    lagged_rivco_ds <- lagged_rivco_ds[year == y - 1, ]

    pred_data_list <- list()
    for (p in percentiles) {
        pred_data_list[[paste0("pred_data_", p, "p")]] <- lagged_rivco_ds[, grep(paste0("_", p, "p"), colnames(lagged_rivco_ds), value = TRUE), with = FALSE]
    }

    x_list <- list()
    for (p in percentiles) {
        x_list[[paste0("x_", p, "p")]] <- pred_data_list[[paste0("pred_data_", p, "p")]] %>%
            rename_at(vars(contains(paste0("_", p, "p"))), ~ str_replace(., paste0("_", p, "p"), "")) %>%
            as.data.table()
    }
    pb <- progress_bar$new(
        format = "(:spin) [:bar] :percent [Elapsed time: :elapsedfull || Estimated time remaining: :eta]",
        total = length(raw_colnames),
        complete = "=", # Completion bar character
        incomplete = "-", # Incomplete bar character
        current = ">", # Current bar character
        clear = FALSE, # If TRUE, clears the bar when finish
        width = 100)
    print(paste0("Calculating moving average for ", y))
    for (i in raw_colnames) {
        pb$tick()
        target_cols <- grep(i, colnames(x_list[["x_50p"]]), value = TRUE)
        target_cols_rev <- rev(target_cols)
        tmp_ds_list <- list()
        for (p in percentiles) {
            tmp_ds_list[[paste0("tmp_ds_", p, "p")]] <- x_list[[paste0("x_", p, "p")]][, target_cols_rev, with = FALSE]
        }
        mas_list <- list()
        for (p in percentiles) {
            mas_list[[paste0("mas_", p, "p")]] <- ema_process(tmp_ds_list[[paste0("tmp_ds_", p, "p")]])
        }
        for (p in percentiles) {
            x_list[[paste0("x_", p, "p")]][, paste0(i, "_ema_5")] <- unlist(mas_list[[paste0("mas_", p, "p")]])[seq(1, length(mas_list[[paste0("mas_", p, "p")]]) * 5, 5)]
            x_list[[paste0("x_", p, "p")]][, paste0(i, "_ema_4")] <- unlist(mas_list[[paste0("mas_", p, "p")]])[seq(2, length(mas_list[[paste0("mas_", p, "p")]]) * 5, 5)]
            x_list[[paste0("x_", p, "p")]][, paste0(i, "_ema_3")] <- unlist(mas_list[[paste0("mas_", p, "p")]])[seq(3, length(mas_list[[paste0("mas_", p, "p")]]) * 5, 5)]
            x_list[[paste0("x_", p, "p")]][, paste0(i, "_ema_2")] <- unlist(mas_list[[paste0("mas_", p, "p")]])[seq(4, length(mas_list[[paste0("mas_", p, "p")]]) * 5, 5)]
            x_list[[paste0("x_", p, "p")]][, paste0(i, "_ema_1")] <- unlist(mas_list[[paste0("mas_", p, "p")]])[seq(5, length(mas_list[[paste0("mas_", p, "p")]]) * 5, 5)]
        }
    }


    pb2 <- progress_bar$new(
        format = "(:spin) [:bar] :percent [Elapsed time: :elapsedfull || Estimated time remaining: :eta]",
        total = length(raw_colnames),
        complete = "=", # Completion bar character
        incomplete = "-", # Incomplete bar character
        current = ">", # Current bar character
        clear = FALSE, # If TRUE, clears the bar when finish
        width = 100)
    print(paste0("Predicting ", y))
    for (target_var in raw_colnames) {
        pb2$tick()
        info_list <- readRDS(paste0("models/rf/", target_var, "_rf.RDS"))
        model <- info_list$rf_model
        qhat <- info_list$qhat
        x_target_list <- list()
        for (i in percentiles) {
            x_target_list[[paste0("x_target_", i, "p")]] <- x_prep(x_list[[paste0("x_", i, "p")]], target_var)
        }

        pred_list <- list()
        for (p in percentiles) {
            p_float <- as.numeric(p) / 100
            pred_list[[paste0("pred_", p, "p")]] <- predict(model, x_target_list[[paste0("x_target_", p, "p")]], what = p_float)
        }
        lower_percetile <- percentiles[percentiles < 50]
        for (p in lower_percetile) {
            p_float <- as.numeric(p) / 100
            pred_list[[paste0("pred_", p, "p")]] <- pred_list[[paste0("pred_", p, "p")]] - qhat[[paste0("alpha_", p_float)]]
        }
        upper_percetile <- percentiles[percentiles > 50]
        for (p in upper_percetile) {
            p_float <- as.numeric(100 - p) / 100
            pred_list[[paste0("pred_", p, "p")]] <- pred_list[[paste0("pred_", p, "p")]] + qhat[[paste0("alpha_", p_float)]]
        }

        zero_line <- zero_value[[target_var]]
        for (p in percentiles) {
            pred_list[[paste0("pred_", p, "p")]][pred_list[[paste0("pred_", p, "p")]] < zero_line] <- zero_line
        }

        if (target_var == "S0101_C01_002E") {
            pred_collection_list <- list()
            for (p in percentiles) {
                pred_collection_list[[paste0("pred_collection_", p, "p")]] <- pred_list[[paste0("pred_", p, "p")]]
            }
        } else {
            for (p in percentiles) {
                pred_collection_list[[paste0("pred_collection_", p, "p")]] <- cbind(pred_collection_list[[paste0("pred_collection_", p, "p")]], pred_list[[paste0("pred_", p, "p")]])
            }
        }
    }

    for (p in percentiles) {
        pred_collection_list[[paste0("pred_collection_", p, "p")]] <- as.data.table(pred_collection_list[[paste0("pred_collection_", p, "p")]])
        colnames(pred_collection_list[[paste0("pred_collection_", p, "p")]]) <- colnames_percentiles_list[[paste0("colnames_", p, "p")]]
        pred_collection_list[[paste0("pred_collection_", p, "p")]] <- pred_collection_list[[paste0("pred_collection_", p, "p")]] * sd_value
    }
    # store pred_collection_list to variable name rf_y
    assign(paste0("rf_", y), pred_collection_list)

}

### RWKV prediction on 2018, 2019, 2020 data in Riverside County

# tensorflow confifuration
devices <- tf$config$get_visible_devices()
tf$config$experimental$set_memory_growth(devices[[2]], TRUE)
# tf$config$set_visible_devices(devices[[1]])
tf$config$threading$set_intra_op_parallelism_threads(16L)
tf$config$threading$set_inter_op_parallelism_threads(16L)


# load data for prediction
rivco_ds <- readRDS("data/census_predbase_prepped_std.rds")

mean_value <- readRDS("data/mean_value_std.rds")
sd_value <- readRDS("data/sd_value_std.rds")
zero_value <- readRDS("data/zero_value_std.rds")

zero_value[length(zero_value)] <- -Inf

raw_colnames <- colnames(rivco_ds) %>% setdiff(c("year", "state", "place", "calib", "cluster_size", "cluster"))

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


# make sure all the time slice is available for each place
# pull the latest place list in reiverside county
place_list <- pull(rivco_ds[year == 2021, ], place)
# for each place in place_list, count the row number of the place in the data.table rivco_ds
place_count <- rivco_ds[place %in% place_list, .N, by = place]
# keep the place with place_count more than or equal to 9
place_list <- place_count[N >= 11, place]
# update rivco_ds with the new place_list
rivco_ds <- rivco_ds[place %in% place_list, ]


# Setting up for training
lookback <- 6
step <- 1
delay <- 1
batch_size <- nrow(rivco_ds[year == 2021, ])
n_input <- length(raw_colnames)
n_output <- length(raw_colnames)
col_offset <- 3
pred_col_range <- seq(col_offset + 1, col_offset + n_output)

source_python("tf_rwkv_model.py")
model <- tf$keras$models$load_model("models/rwkv_50_model/skel.keras")
model$load_weights(paste0("models/rwkv_50_model/epc_cpt"))


## Prediction loop for mid quantile
for (y in seq(2018, 2020, by = 1)) {
    print(paste0("generating projection for year ", y))
    # find minimal index where year ==  y - 1
    pred_min_index <- which(rivco_ds[, year] == y - 1)[1]
    # find maximal index where year ==  y - 1
    pred_max_index <- which(rivco_ds[, year] == y - 1)[length(which(rivco_ds[, year] == y - 1))]

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

    pred_x <- pred_gen()

    pred <- model(
        pred_x[[1]],
        training = FALSE
    )

    pred <- as.matrix(pred)

    # for each column where pred is less than corresponding zero_value, replace the value with corresponding zero_value
    for (i in 1:(ncol(pred) - 1)) {

        if (any(pred[, i] < zero_value[i])) {
            pred[pred[, i] < zero_value[i], i] <- zero_value[i]
        }
    }

    
    revert_std <- function(x, sd, mean) {
        x <- x * sd + mean
        return(x)
    }


    pred <- as.data.table(pred)
    colnames(pred) <- raw_colnames
    for (i in raw_colnames) {
        pred[, paste0(i) := lapply(.SD, revert_std, sd = sd_value[i], mean = mean_value[i]), .SDcols = c(paste0(i))]
    }
    # save pred as variable rwkv_y
    assign(paste0("rwkv_", y), pred)

}


# get rid of "_50p" in column names
rf_2018 <- rf_2018[[1]] %>% rename_at(vars(contains("_50p")), ~ str_replace(., "_50p", ""))
rf_2019 <- rf_2019[[1]] %>% rename_at(vars(contains("_50p")), ~ str_replace(., "_50p", ""))
rf_2020 <- rf_2020[[1]] %>% rename_at(vars(contains("_50p")), ~ str_replace(., "_50p", ""))

# create vector of chars from "S0101_C01_002E" to "S0101_C01_019E"
selected_columns <- paste0("S0101_C01_", str_pad(2:19, 3, pad = "0"), "E")

# keep only selected columns
rf_2018 <- rf_2018 %>% select(all_of(selected_columns))
rf_2019 <- rf_2019 %>% select(all_of(selected_columns))
rf_2020 <- rf_2020 %>% select(all_of(selected_columns))
rwkv_2018 <- rwkv_2018 %>% select(all_of(selected_columns))
rwkv_2019 <- rwkv_2019 %>% select(all_of(selected_columns))
rwkv_2020 <- rwkv_2020 %>% select(all_of(selected_columns))


eval_place_list <- place_list
rivco_ds <- readRDS("data/census_predbase_prepped_sd_norm.rds")
rivco_ds <- rivco_ds %>% filter(place %in% eval_place_list)
rivco_ds <- rivco_ds[year >= 2018 & year <= 2020, ]
rivco_ds <- rivco_ds %>% select(all_of(selected_columns))
sd_value <- readRDS("data/sd_value_sd_norm.rds")
rivco_ds <- rivco_ds * sd_value

# stack up rf_2018, rf_2019, rf_2020 with rbind
rf_df <- rbind(rf_2018, rf_2019, rf_2020)
rwkv_df <- rbind(rwkv_2018, rwkv_2019, rwkv_2020)


# create a vector of age groups
age_groups <- c(
    "0-4",
    "5-9",
    "10-14",
    "15-19",
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80-84",
    "85+"
)

# rename columns
colnames(rf_df) <- age_groups
colnames(rwkv_df) <- age_groups
colnames(rivco_ds) <- age_groups

# add a column "Total" which is sum of all the age groups
rf_df$Total <- rowSums(rf_df)
rwkv_df$Total <- rowSums(rwkv_df)
rivco_ds$Total <- rowSums(rivco_ds)

# check each df for NA
# sum(is.na(rf_df))
# sum(is.na(rwkv_df))
# sum(is.na(rivco_ds))

# get mean absolute error per column rf_df as y_pred, rivco_ds as y_true
mae_rf <- abs(rf_df - rivco_ds) %>% summarise_all(mean) %>% as.tibble(.)
mae_rwkv <- abs(rwkv_df - rivco_ds) %>% summarise_all(mean) %>% as.tibble(.)

# get mean absolute percentage error per column rf_df as y_pred, rivco_ds as y_true
mape_rf <- (abs(rf_df - rivco_ds) / (1 + rivco_ds)) %>% as.tibble(.) %>% mutate_all(~ ifelse(. > quantile(., 0.95), NA, .)) %>%summarise_all(mean, na.rm = TRUE)
mape_rwkv <- (abs(rwkv_df - rivco_ds) / (1 + rivco_ds)) %>% as.tibble(.) %>% mutate_all(~ ifelse(. > quantile(., 0.95), NA, .)) %>%summarise_all(mean, na.rm = TRUE)

# swap rows and columns
eval_table <- cbind(t(mae_rf), t(mae_rwkv), t(mape_rf), t(mape_rwkv)) %>% as.data.frame(.)

# add a row with rowname "Mean" which is mean of all the rows except the last row
mean_row <- eval_table[1:nrow(eval_table)-1, ] %>% colMeans(.) 
# insert mean_row to eval_table at 2nd from last row, name it "Mean"
eval_table <- rbind(eval_table[1:nrow(eval_table)-1, ], mean_row, eval_table[nrow(eval_table), ]) %>% as.data.frame(.)
# rename
row_names <- c(age_groups, "Mean", "Total")
rownames(eval_table) <- row_names
# rename columns
col_names <- c("MAE RF", "MAE RWKV", "MAPE RF", "MAPE RWKV")
colnames(eval_table) <- col_names

# for column 3 and 4, multiply by 100
eval_table[, 3:4] <- eval_table[, 3:4] * 100

# keep only 2 decimal places
eval_table <- eval_table %>% round(2)

saveRDS(eval_table, "data/evaluation_table.RDS")

# save as csv
write.csv(eval_table, "data/evaluation_table.csv")
