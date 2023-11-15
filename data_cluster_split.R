if (!require("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, data.table, dtwclust)


census_ds <- fread("data/census-training-pull.csv", showProgress = TRUE)

# make list of unique state-place pairs with 12 years of data
place_list <- census_ds[, .N, by = .(state, place)]
place_list <- place_list[place_list$N == 12, .(state, place)]

saveRDS(place_list, "data/place_list.RDS")

# pull the rows with state and place from place_list, and store them in a list for tsclust to use
fixed_list <- list()
for (i in seq(nrow(place_list))) {
  fixed_list[[i]] <- census_ds[state == place_list[i, state], ][place == place_list[i, place], ][, 4:ncol(census_ds)] %>% as.matrix()
}

# cluster the list of time series
clust_results <- tsclust(fixed_list, k = 1000L, distance = "sdtw", seed = 123L)
clust_assignment <- clust_results@cluster

for (i in seq(length(clust_assignment))) {
  # get the row index of mathicng state and place from place_list in census_ds
  target_index <- census_ds[, .I[state == place_list[i, state] & place == place_list[i, place]]]
  # store the cluster assignment to census_ds cluster column
  census_ds[target_index, cluster := clust_assignment[i]]
}

# gather info on size of each cluster and store in clust_info_tbl
clust_info_tbl <- matrix(nrow = 0, ncol = 2)
for (i in seq(1000)) {
  clust_info_tbl <- rbind(clust_info_tbl, c(i, nrow(census_ds[cluster == i, ]) / 12))
}
clust_info_tbl <- as.data.frame(clust_info_tbl)
colnames(clust_info_tbl) <- c("cluster", "member")


# collect information of cluster size variation
member_list <- sort(unique(clust_info_tbl$member))
# initialize columns to be filled with 0
census_ds <- census_ds[, `:=`(calib = 0, cluster_size = 0)]
# gather info on split info of validation/calibration set
calibration_tbl <- matrix(nrow = 0, ncol = 2)
for (i in member_list) {
  tmp_lst <- clust_info_tbl[clust_info_tbl$member == i, ]$cluster
  for (j in tmp_lst) {
    census_ds <- census_ds[cluster == j, cluster_size := i]
    if (i < 10) {
      set.seed(123)
      calibration_list <- unique(census_ds[cluster == j, ], by = c("state", "place")) %>%
        select(state, place) %>%
        sample_n(1)
      calibration_tbl <- rbind(calibration_tbl, as.matrix(calibration_list))
    } else {
      set.seed(123)
      calibration_list <- unique(census_ds[cluster == j, ], by = c("state", "place")) %>%
        select(state, place) %>%
        sample_n(floor(i / 5))
      calibration_tbl <- rbind(calibration_tbl, as.matrix(calibration_list))
    }
  }
}
calibration_tbl <- as.data.frame(calibration_tbl)

# write split information to data frame
for (i in seq(nrow(calibration_tbl))) {
  census_ds <- census_ds[state == calibration_tbl$state[i] & place == calibration_tbl$place[i], calib := 1]
}

census_ds[is.na(census_ds)] <- 0

saveRDS(census_ds, "data/census-training-splitted.rds")
