if (!require("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, data.table, parallel, foreach, doParallel, pracma, matrixStats, progress, quantregForest)

# set up parallel computing
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

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


# make sure place to make prjection has data from year 2015 to 2021 (7 years)
place_list <- pull(rivco_ds[year == 2021, ], place)
place_count <- rivco_ds[place %in% place_list, .N, by = place]
place_list <- place_count[N >= 7, place]
rivco_ds <- rivco_ds[place %in% place_list, ]

# left_edge_index
lei <- which(colnames(rivco_ds) == "S0101_C01_002E")
# right_edge_index
rei <- which(colnames(rivco_ds) == "dn")

raw_colnames <- colnames(rivco_ds[, lei:rei, with = FALSE])
# set up quantile range for prediction 40, 50, 60
percentiles <- seq(from = 40, to = 60, by = 10)
# prepare column names for prediction
colnames_percentiles_list <- list()
for (p in percentiles) {
    colnames_percentiles_list[[paste0("colnames_", p, "p")]] <- paste0(raw_colnames, "_", p, "p")
}
# create column for each quantils range
for (p in percentiles) {
    rivco_ds[, colnames_percentiles_list[[paste0("colnames_", p, "p")]]] <- rivco_ds[, lei:rei, with = FALSE]
}


# Setting up parameters for training
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

# projection generatioin loop
for (y in seq(2022, 2060)) {

    # pull out the data from year y - (lookback + 1) to y - 1
    lagged_rivco_ds <- rivco_ds[year %in% seq(y - (lookback + 1), y - 1), ]
    lagged_raw_colnames <- colnames(lagged_rivco_ds) %>% setdiff(c("year", "state", "place"))
    # create lagged columns for each column in lagged_rivco_ds
    for (i in 1:lookback) {
        lagged_rivco_ds <- lagged_rivco_ds[, paste0(lagged_raw_colnames, "_lag_", i) := shift(.SD, i, type = "lag"), by = .(state, place), .SDcols = lagged_raw_colnames]
    }
    # take out range of rows from rivco_ds where year is y-1
    lagged_rivco_ds <- lagged_rivco_ds[year == y - 1, ]

    # separate data for each quantile
    pred_data_list <- list()
    for (p in percentiles) {
        pred_data_list[[paste0("pred_data_", p, "p")]] <- lagged_rivco_ds[, grep(paste0("_", p, "p"), colnames(lagged_rivco_ds), value = TRUE), with = FALSE]
    }

    # rename columns of each data for each quantile, to have generic column names
    x_list <- list()
    for (p in percentiles) {
        # assign(paste0("x_", p, "p"), get(paste0("pred_data_list[[pred_data_", p, "p]]")) %>% rename_at(vars(contains(paste0("_", p, "p"))), ~ str_replace(., paste0("_", p, "p"), "")) %>% as.data.table())
        x_list[[paste0("x_", p, "p")]] <- pred_data_list[[paste0("pred_data_", p, "p")]] %>%
            rename_at(vars(contains(paste0("_", p, "p"))), ~ str_replace(., paste0("_", p, "p"), "")) %>%
            as.data.table()
    }

    # initialize progress bar
    pb <- progress_bar$new(
        format = "(:spin) [:bar] :percent [Elapsed time: :elapsedfull || Estimated time remaining: :eta]",
        total = length(raw_colnames),
        complete = "=", # Completion bar character
        incomplete = "-", # Incomplete bar character
        current = ">", # Current bar character
        clear = FALSE, # If TRUE, clears the bar when finish
        width = 100)
    
    # calculate moving average for each column and each quantile, then add them to the dataset
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

    # expand the dataframe for filling in the prediction
    tmp_tbl <- data.table(
        year = rep(y, batch_size),
        state = rep(6, batch_size),
        place = place_list %>% sort()
    )
    rivco_ds <- bind_rows(rivco_ds, tmp_tbl)

    pb2 <- progress_bar$new(
        format = "(:spin) [:bar] :percent [Elapsed time: :elapsedfull || Estimated time remaining: :eta]",
        total = length(raw_colnames),
        complete = "=", # Completion bar character
        incomplete = "-", # Incomplete bar character
        current = ">", # Current bar character
        clear = FALSE, # If TRUE, clears the bar when finish
        width = 100)
    print(paste0("Predicting ", y))

    # prediction loop
    for (target_var in raw_colnames) {
        pb2$tick()
        # load the model information
        info_list <- readRDS(paste0("models/rf/", target_var, "_rf.RDS"))
        model <- info_list$rf_model
        qhat <- info_list$qhat
        
        # prepare the features for prediction
        x_target_list <- list()
        for (i in percentiles) {
            x_target_list[[paste0("x_target_", i, "p")]] <- x_prep(x_list[[paste0("x_", i, "p")]], target_var)
        }

        # predict each quantile
        pred_list <- list()
        for (p in percentiles) {
            p_float <- as.numeric(p) / 100
            pred_list[[paste0("pred_", p, "p")]] <- predict(model, x_target_list[[paste0("x_target_", p, "p")]], what = p_float)
        }
        # apply conformalized quantile regression to lower and upper quantiles
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
       
        # make sure the prediction is not below zero
        for (p in percentiles) {
            pred_list[[paste0("pred_", p, "p")]][pred_list[[paste0("pred_", p, "p")]] < zero_line] <- zero_line
        }

        # store the prediction from each percentile into list
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

    # fill in the prediction into rivco_ds
    for (p in percentiles) {
        pred_collection_list[[paste0("pred_collection_", p, "p")]] <- as.data.table(pred_collection_list[[paste0("pred_collection_", p, "p")]])
        colnames(pred_collection_list[[paste0("pred_collection_", p, "p")]]) <- colnames_percentiles_list[[paste0("colnames_", p, "p")]]
        rivco_ds[year == y, col_range_list[[paste0("col_range_", p, "p")]]] <- pred_collection_list[[paste0("pred_collection_", p, "p")]]
    }
}

# get rid of the columns are in raw_colnames
rivco_ds <- rivco_ds %>% select(-all_of(raw_colnames))

revert_norm <- function(x, sd) {
    x <- x * sd
    return(x)
}

# reverse the normalization
for (i in raw_colnames) {
    for (p in percentiles) {
        rivco_ds[, paste0(i, "_", p, "p") := lapply(.SD, revert_norm, sd = sd_value), .SDcols = c(paste0(i, "_", p, "p"))]
    }
}

# save the projection
saveRDS(rivco_ds, paste0("data/rf_projection.RDS"))

