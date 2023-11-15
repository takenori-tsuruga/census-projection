if (!require("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, data.table)

census_ds <- readRDS("data/census-training-splitted.rds")
rivco_ds <- fread("data/census-predbase-pull.csv", showProgress = TRUE)

### apply the data processing to census_ds

raw_colnames <- colnames(census_ds)

# grep colnames with "S0701", then exclude ones with "C01"
convert_colnames <- raw_colnames[grep("S0701", raw_colnames)] %>%
    .[!grepl("C01", .)] %>%
    .[!grepl("011E", .)] %>%
    .[!grepl("048E", .)]

# convert all the convert_colnames from percentage to numeric count by multiplying with corresponding C0701_C01
for (i in seq_along(convert_colnames)) {
    census_ds[, (convert_colnames[i]) := get(convert_colnames[i]) * get(paste0("S0701_C01", substr(convert_colnames[i], 10, 14))) / 100]
}

# generate colnames for sum of migration which has format "S0701_C06_001E"
mig_colnames <- raw_colnames[grep("S0701", raw_colnames)] %>%
    .[grepl("C02", .)] %>%
    .[!grepl("011E", .)] %>%
    .[!grepl("048E", .)] %>%
    str_replace_all("C02", "C06")

# create column with mig_colunames which is sum of convert_colnames
for (i in seq_along(mig_colnames)) {
    sum_colnames <- convert_colnames[grep(substr(mig_colnames[i], 10, 14), convert_colnames)]
    census_ds[, (mig_colnames[i]) := rowSums(.SD), .SDcols = sum_colnames]
}

# remove columns include "S1301_C03" or "S1301_C04" in its name as they are just redundant different representations of S1301_C02
remove_cols <- raw_colnames[grep("S1301", raw_colnames)] %>% .[grepl("C03", .) | grepl("C04", .)]
census_ds[, (remove_cols) := NULL]

# update column "S1301_C05" by product of "S1301_C05" and "S1301_C02"
convert_colnames <- raw_colnames[grep("S1301", raw_colnames)] %>% .[grepl("C05", .)]
for (i in seq_along(convert_colnames)) {
    census_ds[, (convert_colnames[i]) := get(convert_colnames[i]) * get(paste0("S1301_C02", substr(convert_colnames[i], 10, 14))) / 100]
}

# get rid of non population count unit columns
remove_cols <- raw_colnames[grep("S0701", raw_colnames)] %>% .[grepl("011E", .) | grepl("048E", .)]
census_ds[, (remove_cols) := NULL]
remove_cols <- raw_colnames[grep("S1301_C01", raw_colnames)] %>% .[grepl("031E", .) | grepl("032E", .)]
census_ds[, (remove_cols) := NULL]


# calulate decrease in population
census_ds <- census_ds[, S0101_C01_001E_lag := shift(S0101_C01_001E, 1, type = "lag"), by = .(state, place)]
census_ds <- census_ds[, dn := (S0101_C01_001E - S0101_C01_001E_lag) - S0701_C06_001E - S1301_C02_001E, by = .(state, place)]
census_ds[, S0101_C01_001E_lag := NULL]

# get rid of rows where year == 2010 since it doesn't have dn
census_ds <- census_ds[year != 2010]


### apply the data processing to rivco_ds

raw_colnames <- colnames(rivco_ds)

# grep colnames with "S0701", then exclude ones with "C01"
convert_colnames <- raw_colnames[grep("S0701", raw_colnames)] %>%
    .[!grepl("C01", .)] %>%
    .[!grepl("011E", .)] %>%
    .[!grepl("048E", .)]

# convert all the convert_colnames from percentage to numeric count by multiplying with corresponding C0701_C01
for (i in seq_along(convert_colnames)) {
    rivco_ds[, (convert_colnames[i]) := get(convert_colnames[i]) * get(paste0("S0701_C01", substr(convert_colnames[i], 10, 14))) / 100]
}

# generate colnames for sum of migration which has format "S0701_C06_001E"
mig_colnames <- raw_colnames[grep("S0701", raw_colnames)] %>%
    .[grepl("C02", .)] %>%
    .[!grepl("011E", .)] %>%
    .[!grepl("048E", .)] %>%
    str_replace_all("C02", "C06")

# create column with mig_colunames which is sum of convert_colnames
for (i in seq_along(mig_colnames)) {
    sum_colnames <- convert_colnames[grep(substr(mig_colnames[i], 10, 14), convert_colnames)]
    rivco_ds[, (mig_colnames[i]) := rowSums(.SD), .SDcols = sum_colnames]
}

# remove columns include "S1301_C03" or "S1301_C04" in its name as they are just redundant different representations of S1301_C02
remove_cols <- raw_colnames[grep("S1301", raw_colnames)] %>% .[grepl("C03", .) | grepl("C04", .)]
rivco_ds[, (remove_cols) := NULL]

# update column "S1301_C05" my product of "S1301_C05" and "S1301_C02"
convert_colnames <- raw_colnames[grep("S1301", raw_colnames)] %>% .[grepl("C05", .)]
for (i in seq_along(convert_colnames)) {
    rivco_ds[, (convert_colnames[i]) := get(convert_colnames[i]) * get(paste0("S1301_C02", substr(convert_colnames[i], 10, 14))) / 100]
}

# get rid of non population count unit columns
remove_cols <- raw_colnames[grep("S0701", raw_colnames)] %>% .[grepl("011E", .) | grepl("048E", .)]
rivco_ds[, (remove_cols) := NULL]
remove_cols <- raw_colnames[grep("S1301_C01", raw_colnames)] %>% .[grepl("031E", .) | grepl("032E", .)]
rivco_ds[, (remove_cols) := NULL]

# calulate decrease in population
rivco_ds <- rivco_ds[, S0101_C01_001E_lag := shift(S0101_C01_001E, 1, type = "lag"), by = .(state, place)]
rivco_ds <- rivco_ds[, dn := (S0101_C01_001E - S0101_C01_001E_lag) - S0701_C06_001E - S1301_C02_001E, by = .(state, place)]
rivco_ds[, S0101_C01_001E_lag := NULL]

# get rid of rows where year == 2010 since it doesn't have dn
rivco_ds <- rivco_ds[year != 2010]


# sort out the columns order for census_ds and rivco_ds
tmp <- colnames(census_ds) %>%
    str_extract("S\\d{4}_C\\d{2}_\\d{3}E") %>%
    sort()
setcolorder(census_ds, c("year", "state", "place", tmp, "dn", "cluster", "calib", "cluster_size"))
setcolorder(rivco_ds, c("year", "state", "place", tmp, "dn"))


## normalize and standardize census_ds and rivco_ds

left_edge_index <- which(colnames(census_ds) == "S0101_C01_001E")
right_edge_index <- which(colnames(census_ds) == "dn")

census_ds[is.na(census_ds)] <- 0
rivco_ds[is.na(rivco_ds)] <- 0


# find out mean and sd for each column in census_ds for standardization
mean_value <- sapply(census_ds[, left_edge_index:right_edge_index, with = F], mean)
sd_value <- sapply(census_ds[, left_edge_index:right_edge_index, with = F], sd)

std <- function(x, sd, mean) {
    (x - mean) / sd
}


census_ds_std <- copy(census_ds)
# apply standardization to census_ds column index between left_edge_index and right_edge_index based on sd_value and mean_value
for (i in left_edge_index:right_edge_index) {
    census_ds_std[, c(i) := lapply(.SD, std, sd_value[i - left_edge_index + 1], mean_value[i - left_edge_index + 1]), .SDcols = c(i)]
}

# save census_ds_std to rds
saveRDS(census_ds_std, "data/census_training_prepped_std.rds")

rivco_ds_std <- copy(rivco_ds)
# apply standardization to rivco_ds column index between left_edge_index and right_edge_index based on sd_value and mean_value
for (i in left_edge_index:right_edge_index) {
    rivco_ds_std[, c(i) := lapply(.SD, std, sd_value[i - left_edge_index + 1], mean_value[i - left_edge_index + 1]), .SDcols = c(i)]
}

# save rivco_ds_std to rds
saveRDS(rivco_ds_std, "data/census_predbase_prepped_std.rds")


# create named vector of zeros with column name from census_ds 4:473
zero_vector <- rep(0, length(left_edge_index:right_edge_index))
names(zero_vector) <- names(census_ds[, left_edge_index:right_edge_index])
zero_value <- zero_vector - mean_value / sd_value

# save max_value, mean_value, sd_value, zero_value to rds
saveRDS(mean_value, "data/mean_value_std.rds")
saveRDS(sd_value, "data/sd_value_std.rds")
saveRDS(zero_value, "data/zero_value_std.rds")


# another normalization method for statistical ML model: simple scale by whole sd
sd_value <- sd(as.matrix(census_ds[, left_edge_index:right_edge_index, with = F]))


census_ds_std <- copy(census_ds)
census_ds_std[, left_edge_index:right_edge_index] <- census_ds_std[, left_edge_index:right_edge_index] / sd_value

# save census_ds_std to rds
saveRDS(census_ds_std, "data/census_training_prepped_sd_norm.rds")

rivco_ds_std <- copy(rivco_ds)
rivco_ds_std[, left_edge_index:right_edge_index] <- rivco_ds_std[, left_edge_index:right_edge_index] / sd_value

# save rivco_ds_std to csv and rds
saveRDS(rivco_ds_std, "data/census_predbase_prepped_sd_norm.rds")

saveRDS(sd_value, "data/sd_value_sd_norm.rds")
