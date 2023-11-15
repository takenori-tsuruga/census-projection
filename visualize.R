if (!require("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, openxlsx, data.table, car, parallel, foreach, doParallel, viridis)

# check if the plots folder exists, if not, create it
if (!dir.exists("plots")) {
    dir.create("plots")
}

add_total_row <- function(df) {
    sum_cols <- names(df)[!grepl("year|state|place", names(df))]
    for (y in seq(2011, 2060)) {
        total_row <- df %>% filter(year == y) %>% summarise_at(sum_cols, sum)
        total_row$year <- y
        total_row$state <- 0
        total_row$place <- 0
        total_row <- total_row[, names(df)]
        df <- rbind(df, total_row)
    }
    return(df)
}

age_bin_columns <- c(
    "S0101_C01_002E",
    "S0101_C01_003E",
    "S0101_C01_004E",
    "S0101_C01_005E",
    "S0101_C01_006E",
    "S0101_C01_007E",
    "S0101_C01_008E",
    "S0101_C01_009E",
    "S0101_C01_010E",
    "S0101_C01_011E",
    "S0101_C01_012E",
    "S0101_C01_013E",
    "S0101_C01_014E",
    "S0101_C01_015E",
    "S0101_C01_016E",
    "S0101_C01_017E",
    "S0101_C01_018E",
    "S0101_C01_019E"
)

plot_gen <- function(df, place_filter) {
    plot_table <- df %>%
        filter(place == place_filter) %>%
        select(year, all_of(age_bin_columns))
    plot_table <- plot_table %>% pivot_longer(cols = -year, names_to = "age_bin", values_to = "population")
    plot_data <- ggplot(plot_table, aes(x = year, y = population, fill = age_bin)) +
        geom_bar(stat = "identity", position = position_stack(reverse = TRUE)) +
        labs( x = "Year", y = "Population") +
        scale_fill_viridis(discrete = TRUE, direction = -1) +
        geom_vline(xintercept = 2021.5, color = "red", linetype = "solid", size = 2)  +
        theme(text = element_text(size = 30), legend.position = "none")
    return(plot_data)
}


rf_8_proj <- readRDS("data/rf_projection.RDS")
# break them into lower_quantile_df, middle_quantile_df, upper_quantile_df
lower_quantile_df <- rf_8_proj %>% select(year, state, place, contains("_40p"))
# renmae the columns to remove "_40p"
names(lower_quantile_df) <- gsub("_40p", "", names(lower_quantile_df))
# apply the same to middle_quantile_df, upper_quantile_df
middle_quantile_df <- rf_8_proj %>% select(year, state, place, contains("_50p"))
names(middle_quantile_df) <- gsub("_50p", "", names(middle_quantile_df))
upper_quantile_df <- rf_8_proj %>% select(year, state, place, contains("_60p"))
names(upper_quantile_df) <- gsub("_60p", "", names(upper_quantile_df))

# add total row for each year
rf_lq_df <- add_total_row(lower_quantile_df)
rf_mq_df <- add_total_row(middle_quantile_df)
rf_uq_df <- add_total_row(upper_quantile_df)

# generate visualization of projection for whole Riverside County for each quantile by Random Forest
place_list <- c("0")
plot_list <- list()
model_type <- "rf"
for (i in place_list) {
    p1 <- plot_gen(rf_lq_df, i)
    ggsave(p1, filename = paste0("plots/", model_type, "_", i, "_lq.png"), width = 10, height = 10, dpi = 300)
    p2 <- plot_gen(rf_mq_df, i)
    ggsave(p2, filename = paste0("plots/", model_type, "_", i, "_mq.png"), width = 10, height = 10, dpi = 300)
    p3 <- plot_gen(rf_uq_df, i)
    ggsave(p3, filename = paste0("plots/", model_type, "_", i, "_uq.png"), width = 10, height = 10, dpi = 300)
}

# load data for rwkv
lower_quantile_df <- readRDS("data/rwkv_projection_49.RDS")
middle_quantile_df <- readRDS("data/rwkv_projection_50.RDS")
upper_quantile_df <- readRDS("data/rwkv_projection_51.RDS")
# add total row 
rwkv_lq_df <- add_total_row(lower_quantile_df)
rwkv_mq_df <- add_total_row(middle_quantile_df)
rwkv_uq_df <- add_total_row(upper_quantile_df)
# generate visualization of projection for whole Riverside County for each quantile by RWKV
place_list <- c("0")
plot_list <- list()
model_type <- "rwkv"
for (i in place_list) {
    p1 <- plot_gen(rwkv_lq_df, i)
    ggsave(p1, filename = paste0("plots/", model_type, "_", i, "_lq.png"), width = 10, height = 10, dpi = 300)
    p2 <- plot_gen(rwkv_mq_df, i)
    ggsave(p2, filename = paste0("plots/", model_type, "_", i, "_mq.png"), width = 10, height = 10, dpi = 300)
    p3 <- plot_gen(rwkv_uq_df, i)
    ggsave(p3, filename = paste0("plots/", model_type, "_", i, "_uq.png"), width = 10, height = 10, dpi = 300)
}


youth_cols <- c(
    "S0101_C01_002E", # 0-4
    "S0101_C01_003E", # 5-9
    "S0101_C01_004E", # 10-14
    "S0101_C01_005E", # 15-19
    "S0101_C01_006E" # 20-24
)

working_cols <- c(
    "S0101_C01_007E", # 25-29
    "S0101_C01_008E", # 30-34
    "S0101_C01_009E", # 35-39
    "S0101_C01_010E", # 40-44
    "S0101_C01_011E", # 45-49
    "S0101_C01_012E", # 50-54
    "S0101_C01_013E", # 55-59
    "S0101_C01_014E" # 60-64
)

senior_cols <- c(
    "S0101_C01_015E", # 65-69
    "S0101_C01_016E", # 70-74
    "S0101_C01_017E", # 75-79
    "S0101_C01_018E", # 80-84
    "S0101_C01_019E" # 85+
)


add_age_group_cols <- function(df) {
    df <- df %>% mutate(
        youth = rowSums(select(., all_of(youth_cols))),
        working = rowSums(select(., all_of(working_cols))),
        senior = rowSums(select(., all_of(senior_cols))),
        total = rowSums(select(., all_of(age_bin_columns)))
    )
    return(df)
}

# add age group columns to each data frame
rf_lq_df <- add_age_group_cols(rf_lq_df)
rf_mq_df <- add_age_group_cols(rf_mq_df)
rf_uq_df <- add_age_group_cols(rf_uq_df)
rwkv_lq_df <- add_age_group_cols(rwkv_lq_df)
rwkv_mq_df <- add_age_group_cols(rwkv_mq_df)
rwkv_uq_df <- add_age_group_cols(rwkv_uq_df)

add_ratio_cols <- function(df) {
    df <- df %>% mutate(
        working_senior_ratio = working / senior,
        working_total_ratio = working / total * 100,
        senior_total_ratio = senior / total * 100
    )
    return(df)
}

# add ratio columns to each data frame
rf_lq_df <- add_ratio_cols(rf_lq_df)
rf_mq_df <- add_ratio_cols(rf_mq_df)
rf_uq_df <- add_ratio_cols(rf_uq_df)

rwkv_lq_df <- add_ratio_cols(rwkv_lq_df)
rwkv_mq_df <- add_ratio_cols(rwkv_mq_df)
rwkv_uq_df <- add_ratio_cols(rwkv_uq_df)

# make sure to set Nan to 0
rf_lq_df[is.na(rf_lq_df)] <- 0
rf_mq_df[is.na(rf_mq_df)] <- 0
rf_uq_df[is.na(rf_uq_df)] <- 0
rwkv_lq_df[is.na(rwkv_lq_df)] <- 0
rwkv_mq_df[is.na(rwkv_mq_df)] <- 0
rwkv_uq_df[is.na(rwkv_uq_df)] <- 0

# select columns reuiqred for visualization
rf_lq_df <- rf_lq_df %>% select(year, state, place, total, youth, working, senior, working_senior_ratio, working_total_ratio, senior_total_ratio)
rf_mq_df <- rf_mq_df %>% select(year, state, place, total, youth, working, senior, working_senior_ratio, working_total_ratio, senior_total_ratio)
rf_uq_df <- rf_uq_df %>% select(year, state, place, total, youth, working, senior, working_senior_ratio, working_total_ratio, senior_total_ratio)

rwkv_lq_df <- rwkv_lq_df %>% select(year, state, place, total, youth, working, senior, working_senior_ratio, working_total_ratio, senior_total_ratio)
rwkv_mq_df <- rwkv_mq_df %>% select(year, state, place, total, youth, working, senior, working_senior_ratio, working_total_ratio, senior_total_ratio)
rwkv_uq_df <- rwkv_uq_df %>% select(year, state, place, total, youth, working, senior, working_senior_ratio, working_total_ratio, senior_total_ratio)


## data frame preparation for plotting stacked age group bar chart combined with ratio line chart
prep_df <- function(df) {
    df <- df %>% 
        pivot_longer(cols = c(youth, working, senior), names_to = "age_group", values_to = "population") %>%
        pivot_longer(cols = c(working_total_ratio, senior_total_ratio), names_to = "ratio_type", values_to = "ratio")
    return(df)
}

rf_mq_plot_df <- prep_df(rf_mq_df)
rwkv_mq_plot_df <- prep_df(rwkv_mq_df)


place_list <- c("Riverside", "Blythe", "Jurupa Valley", "Hemet", "Indio", "Lake Elsinore", "Menifee", "Palm Springs", "0")

for (i in place_list) {
    max_total <- rf_mq_plot_df %>% filter(place == i) %>% select(total) %>% max() %>% as.numeric()
    max_ratio <- rf_mq_plot_df %>% filter(place == i) %>% select(ratio) %>% max() %>% as.numeric()
    ggplot(as.data.frame(rf_mq_plot_df %>% filter(place == i)), aes(x = year)) + 
        geom_bar(aes(y = population / 2, fill = age_group), stat = "identity") +
        scale_fill_viridis(name = "Age Group", labels = c("Senior", "Working", "Youth"), discrete = TRUE) +
        geom_vline(xintercept = 2021.5, color = "red", linetype = "solid", size = 2)  +
        labs(fill = "Age Group", x = "Year") +
        ylab("Population") +
        ylim(0, max_total) +
        geom_line(aes(y = ratio * max_total / 100, group = ratio_type, linetype = ratio_type), size=2, stat = "identity") +
        labs(color = "Ratio Type") +
        scale_linetype_discrete(name = "Ratio Type", labels = c("Senior", "Working")) +
        scale_y_continuous(sec.axis = sec_axis(~ . / max_total * 100, name = "Ratio")) +
        theme(text = element_text(size = 30), legend.position = "bottom", legend.text = element_text(size = 25)) +
        guides(fill=guide_legend(keywidth=1 , nrow=3, byrow=FALSE), linetype=guide_legend(keywidth=4, nrow=3, byrow=FALSE))
    ggsave(paste0("plots/rf_combination_", i, ".png"), width = 10, height = 10, dpi = 300)
}

for (i in place_list) {
    max_total <- rwkv_mq_plot_df %>% filter(place == i) %>% select(total) %>% max() %>% as.numeric()
    max_ratio <- rwkv_mq_plot_df %>% filter(place == i) %>% select(ratio) %>% max() %>% as.numeric()
    ggplot(as.data.frame(rwkv_mq_plot_df %>% filter(place == i)), aes(x = year)) + 
        geom_bar(aes(y = population / 2, fill = age_group), stat = "identity") +
        scale_fill_viridis(name = "Age Group", labels = c("Senior", "Working", "Youth"), discrete = TRUE) +
        geom_vline(xintercept = 2021.5, color = "red", linetype = "solid", size = 2)  +
        labs(fill = "Age Group", x = "Year") +
        ylab("Population") +
        ylim(0, max_total) +
        geom_line(aes(y = ratio * max_total / 100, group = ratio_type, linetype = ratio_type), size=2, stat = "identity") +
        labs(color = "Ratio Type") +
        scale_linetype_discrete(name = "Ratio Type", labels = c("Senior", "Working")) +
        scale_y_continuous(sec.axis = sec_axis(~ . / max_total * 100, name = "Ratio")) +
        theme(text = element_text(size = 30), legend.position = "bottom", legend.text = element_text(size = 25)) +
        guides(fill=guide_legend(keywidth=1 , nrow=3, byrow=FALSE), linetype=guide_legend(keywidth=4, nrow=3, byrow=FALSE))
    ggsave(paste0("plots/rwkv_combination_", i, ".png"), width = 10, height = 10, dpi = 300)
}

# pareparation fo total population quantile ribbon plots
total_pop_df <- rf_mq_df %>% select(year, state, place, total) %>% rename(total_mq_rf = total) 
total_pop_df <- total_pop_df %>% left_join(rf_lq_df %>% select(year, state, place, total), by = c("year", "state", "place")) %>% rename(total_lq_rf = total)
total_pop_df <- total_pop_df %>% left_join(rf_uq_df %>% select(year, state, place, total), by = c("year", "state", "place")) %>% rename(total_uq_rf = total)
total_pop_df <- total_pop_df %>% left_join(rwkv_mq_df %>% select(year, state, place, total), by = c("year", "state", "place")) %>% rename(total_mq_rwkv = total)
total_pop_df <- total_pop_df %>% left_join(rwkv_lq_df %>% select(year, state, place, total), by = c("year", "state", "place")) %>% rename(total_lq_rwkv = total)
total_pop_df <- total_pop_df %>% left_join(rwkv_uq_df %>% select(year, state, place, total), by = c("year", "state", "place")) %>% rename(total_uq_rwkv = total)

# plot total population, 2 lines each for rf and rwkv middle quantile, ribbons for upper and lower quantile, and vertical line for 2021
for (place_name in place_list) {
    ggplot(as.data.frame(total_pop_df %>% filter(place == place_name)), aes(x = year)) +
        geom_line(aes(y = total_mq_rf, color = "RF Mid Quantile"), size = 2, show.legend = FALSE) +
        geom_line(aes(y = total_mq_rwkv, color = "RWKV Mid Quantile"), size = 2, show.legend = FALSE) +
        geom_ribbon(aes(ymin = total_lq_rf, ymax = total_uq_rf, fill = "Random Forest"), alpha = 0.2) +
        geom_ribbon(aes(ymin = total_lq_rwkv, ymax = total_uq_rwkv, fill = "RWKV"), alpha = 0.2) +
        geom_vline(xintercept = 2021.5, color = "red", linetype = "solid", size = 2)  +
        labs(fill = "Model", x = "Year") +
        ylab("Population") +
        theme(text = element_text(size = 30), legend.position = "bottom", legend.text = element_text(size = 25))
    ggsave(paste0("plots/total_pop_", place_name, ".png"), width = 10, height = 10, dpi = 300)
}

# preparation for working_senior_ratio projection plot
ws_ratio_df <- rf_mq_df %>% select(year, state, place, working_senior_ratio) %>% mutate(model = "Random Forest", quantile = "Middle", ratio_type = "rf_mq")
ws_ratio_df <- rbind(ws_ratio_df, rf_lq_df %>% select(year, state, place, working_senior_ratio) %>% mutate(model = "Random Forest", quantile = "Lower", ratio_type = "rf_lq"))
ws_ratio_df <- rbind(ws_ratio_df, rf_uq_df %>% select(year, state, place, working_senior_ratio) %>% mutate(model = "Random Forest", quantile = "Upper", ratio_type = "rf_uq"))
ws_ratio_df <- rbind(ws_ratio_df, rwkv_mq_df %>% select(year, state, place, working_senior_ratio) %>% mutate(model = "RWKV", quantile = "Middle", ratio_type = "rwkv_mq"))
ws_ratio_df <- rbind(ws_ratio_df, rwkv_lq_df %>% select(year, state, place, working_senior_ratio) %>% mutate(model = "RWKV", quantile = "Lower", ratio_type = "rwkv_lq"))
ws_ratio_df <- rbind(ws_ratio_df, rwkv_uq_df %>% select(year, state, place, working_senior_ratio) %>% mutate(model = "RWKV", quantile = "Upper", ratio_type = "rwkv_uq"))

# set order of factor for quantile to Upper, Middle, Lower
ws_ratio_df$quantile <- factor(ws_ratio_df$quantile, levels = c("Upper", "Middle", "Lower"))

# plot working_senior_ratio, 2 lines each for rf and rwkv middle quantile, different kind of dotted line for lower and upper quantile, and horizontal dotted line for 1, vertical solid line for 2021.5
for (place_name in place_list) {
    ggplot(as.data.frame(ws_ratio_df %>% filter(place == place_name)), aes(x = year)) +
        geom_line(aes(y=working_senior_ratio, group = ratio_type, color = model, linetype = quantile, size=quantile)) +
        geom_hline(yintercept = 1, color = "black", linetype = "dashed", size = 1)  +
        geom_vline(xintercept = 2021.5, color = "red", linetype = "solid", size = 1)  +
        labs(x = "Year", color = "Model", linetype = "Quantile") +
        ylab("Working/Senior Ratio") +
        ylim(0, NA) +
        scale_linetype_manual(name = "Quantile", values = c("Middle" = "solid", "Upper" = "longdash", "Lower" = "twodash")) +
        scale_size_manual(name = "Quantile", values = c("Middle" = 2, "Lower" = 1, "Upper" = 1)) +
        theme(text = element_text(size = 30), legend.position = "bottom", legend.text = element_text(size = 25)) +
        guides(color = guide_legend(keywidth = 3, nrow = 3, byrow = FALSE), linetype = guide_legend(keywidth = 5, nrow = 3, byrow = FALSE)) 
    ggsave(paste0("plots/working_senior_ratio_", place_name, ".png"), width = 10, height = 10, dpi = 300)
}

# prepare for density plot of working_senior_ratio on 2021 and 2060
ws_ratio_df <- rf_mq_df %>% select(year, state, place, working_senior_ratio) %>% rename(working_senior_ratio_mq_rf = working_senior_ratio)
ws_ratio_df <- ws_ratio_df %>% left_join(rf_lq_df %>% select(year, state, place, working_senior_ratio), by = c("year", "state", "place")) %>% rename(working_senior_ratio_lq_rf = working_senior_ratio)
ws_ratio_df <- ws_ratio_df %>% left_join(rf_uq_df %>% select(year, state, place, working_senior_ratio), by = c("year", "state", "place")) %>% rename(working_senior_ratio_uq_rf = working_senior_ratio)

ws_ratio_df <- ws_ratio_df %>% left_join(rwkv_mq_df %>% select(year, state, place, working_senior_ratio), by = c("year", "state", "place")) %>% rename(working_senior_ratio_mq_rwkv = working_senior_ratio)
ws_ratio_df <- ws_ratio_df %>% left_join(rwkv_lq_df %>% select(year, state, place, working_senior_ratio), by = c("year", "state", "place")) %>% rename(working_senior_ratio_lq_rwkv = working_senior_ratio)
ws_ratio_df <- ws_ratio_df %>% left_join(rwkv_uq_df %>% select(year, state, place, working_senior_ratio), by = c("year", "state", "place")) %>% rename(working_senior_ratio_uq_rwkv = working_senior_ratio)

# density plot of working_senior_ratio on 2021 and 2060, place == "0"
ws_ratio_df_2021 <- ws_ratio_df %>% filter(year == 2021 & place != "0")
ws_ratio_df_2060 <- ws_ratio_df %>% filter(year == 2060 & place != "0")

# replace NA with 0
ws_ratio_df_2021[is.na(ws_ratio_df_2021)] <- 0
ws_ratio_df_2060[is.na(ws_ratio_df_2060)] <- 0

plot_df_2021 <- ws_ratio_df %>% filter(year == 2021 & place != "0") %>% select(working_senior_ratio_mq_rf) %>% rename(ratio = working_senior_ratio_mq_rf)
plot_df_2060_rf <- ws_ratio_df %>% filter(year == 2060 & place != "0") %>% select(working_senior_ratio_mq_rf) %>% rename(ratio = working_senior_ratio_mq_rf)
plot_df_2060_rwkv <- ws_ratio_df %>% filter(year == 2060 & place != "0") %>% select(working_senior_ratio_mq_rwkv) %>% rename(ratio = working_senior_ratio_mq_rwkv)

# put them together in one df with type column which has "2021", "2060_rf", "2060_rwkv"
plot_df <- rbind(plot_df_2021 %>% mutate(type = "2021"), plot_df_2060_rf %>% mutate(type = "2060_rf"), plot_df_2060_rwkv %>% mutate(type = "2060_rwkv"))

ggplot(as.data.frame(plot_df), aes(x = ratio, group = type, fill = type)) + 
    geom_density(alpha = .2) +
    xlim(-8, 10) +
    ylim(0, NA) +
    theme(text = element_text(size = 30)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "red", size = 1) + 
    scale_color_brewer(palette="Dark2") +
    labs(fill = "Type") +
    theme(legend.position = "bottom", legend.text = element_text(size = 25)) +
    labs(x = "Working/Senior Ratio", y = "Density") +
    scale_fill_discrete(labels = c("2021", "2060 RF", "2060 RWKV"))
ggsave("plots/working_senior_ratio_density_rivco_2021_2060.png", width = 10, height = 10, dpi = 300)
