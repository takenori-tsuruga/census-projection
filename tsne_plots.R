if (!require("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, Rtsne, data.table, dtwclust, viridis, randomcoloR)

# check if the plots folder exists, if not, create it
if (!dir.exists("plots")) {
    dir.create("plots")
}

census_ds <- fread("data/census-training-pull.csv", showProgress = TRUE)

# make list of unique state-place pairs with 12 years of data
place_list <- census_ds[, .N, by = .(state, place)]
place_list <- place_list[place_list$N == 12, .(state, place)]

# pull the rows with state and place from place_list, and store them in a list for tsclust to use
fixed_list <- list()
for (i in seq(nrow(place_list))) {
  fixed_list[[i]] <- census_ds[state == place_list[i, state], ][place == place_list[i, place], ][, 4:ncol(census_ds)] %>% as.matrix()
}

# clusterin into 10 clusters for tsne visualization
clust_results <- tsclust(fixed_list, k = 10L, distance = "sdtw", seed = 123L)
distmat <- clust_results@distmat
distmat <- as.matrix(distmat)

# use Rtsne to do tsne with distant matrix generated from tsclust
set.seed(123)
tsne_out <- Rtsne(
    distmat,
    check_duplicates = FALSE,
    pca = FALSE,
    perplexity = 50,
    theta = 0.5,
    eta = nrow(distmat) %/% 12,
    dims = 2,
    exaggeration_factor = 12,
    momentum = 0.5,
    final_momentum = 0.8,
    mom_switch_iter = 250L,
    max_iter = 10000L,
    verbose = TRUE,
    normalize = FALSE,
    is_distance = TRUE,
    num_threads = 16L)

# prepare color palette
n <- 10
pallette <- distinctColorPalette(n)
# prepare data frame for ggolot
tsne_out_df <- data.frame(tsne_out$Y)
tsne_out_df$cluster <- clust_results@cluster
tsne_out_df$cluster <- as.factor(tsne_out_df$cluster)
tsne_out_df$cluster <- factor(tsne_out_df$cluster, levels = unique(tsne_out_df$cluster))
# tsne plot with ggplot
ggplot(tsne_out_df, aes(x = X1, y = X2, color = cluster)) + 
    geom_point(size = 1) + 
    theme_minimal() + 
    theme(legend.position = "none") + 
    scale_color_manual(values = pallette)
ggsave("plots/tsne_k10.png", width = 10, height = 10, dpi = 300)


# gather information about split for tsne visualization
place_list <- readRDS("data/place_list.RDS")
splitted_data <- readRDS("data/census-training-splitted.rds")
splitted_data <- splitted_data[year == 2021, ]
calib <- c()
for (i in seq(nrow(place_list))) {
    calib <- append(calib, splitted_data[(state == place_list[i, state]) & (place == place_list[i, place]), calib])
}

# prepare color palette
pallette <- viridis_pal(option = "C")(2)
# prepare data frame for ggolot
tsne_out_df <- data.frame(tsne_out$Y)
tsne_out_df$calib <- calib
tsne_out_df$calib <- as.factor(tsne_out_df$calib)
tsne_out_df$calib <- factor(tsne_out_df$calib, levels = unique(tsne_out_df$calib))
# tsne plot with ggplot
ggplot(tsne_out_df, aes(x = X1, y = X2, color = calib)) + 
    geom_point(size = 1) + 
    theme_minimal() + 
    theme(legend.position = "none") + 
    scale_color_manual(values = pallette)
ggsave("plots/tsne_split.png", width = 10, height = 10, dpi = 300)
