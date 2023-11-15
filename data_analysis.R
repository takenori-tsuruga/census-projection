if (!require("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, data.table, viridis)

# check if data directory exists, if not, create one
if (!dir.exists("plots")) {
  dir.create("plots")
}

# read "data/census-training-pull.csv" into census_ds data.table
census_ds <- fread("data/census-training-pull.csv", showProgress = TRUE)
rivco_ds <- fread("data/census-predbase-pull.csv", showProgress = TRUE)

#list common columns
common_cols <- intersect(names(census_ds), names(rivco_ds))

# filter out with common columns
census_ds <- census_ds[, common_cols, with = FALSE]
rivco_ds <- rivco_ds[, common_cols, with = FALSE]

# merge census_ds and rivco_ds
all_ds <- rbind(census_ds, rivco_ds)

# keep only S0101_C1 which has age binned population data
select_cols <- grepl("year|state|place|S0101_C01", names(all_ds))
all_ds <- all_ds[, select_cols, with = FALSE]

# get total of each column each year, and add as new row with state = 0, place = 0, year =
sum_cols <- names(all_ds)[!grepl("year|state|place", names(all_ds))]
for (y in seq(2010, 2021)) {
    total_row <- all_ds %>% filter(year == y) %>% summarise_at(sum_cols, sum)
    total_row$year <- y
    total_row$state <- 0
    total_row$place <- 0
    total_row <- total_row[, names(all_ds)]
    all_ds <- rbind(all_ds, total_row)
}

# age group columns
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

# for each group, get total in same row, and add as new column
all_ds <- all_ds %>% mutate(
    youth = rowSums(select(., all_of(youth_cols))),
    working = rowSums(select(., all_of(working_cols))),
    senior = rowSums(select(., all_of(senior_cols)))
)

growth_rate <- function(beg, end) {
    rate <- (end - beg) / beg
    return(rate * 100)
}

# create unique state, place comination list which has all 12 years of row
place_list <- all_ds[, .N, by = .(state, place)]
place_list <- place_list[place_list$N == 12, .(state, place)]

# gather growth rate info for each place
growth_rate_info <- c()
for (i in seq(nrow(place_list))) {
    s <- place_list[i, state]
    p <- place_list[i, place]
    place_ds <- all_ds[all_ds$state == s & all_ds$place == p, ]
    place_ds <- place_ds %>% arrange(year)
    growth_rate_info <- append(growth_rate_info, growth_rate(place_ds[year == 2010, ]$S0101_C01_001E, place_ds[year == 2021, ]$S0101_C01_001E))
}

# create new column which has ratio of working age against senior age
all_ds <- all_ds %>% mutate(
    working_senior_ratio = working / senior
)

# create new column which has ratio of working age against total population
all_ds <- all_ds %>% mutate(
    working_total_ratio = working / S0101_C01_001E * 100
)

# create new column which has ratio of senior age against total population
all_ds <- all_ds %>% mutate(
    senior_total_ratio = senior / S0101_C01_001E * 100
)

# plot density plot of growth_rate for all places
plot_df <- as.data.frame(growth_rate_info)

ggplot(as.data.frame(plot_df), aes(x = growth_rate_info)) + 
    geom_histogram(aes(y=..density..), color = "darkblue", fill = "lightblue") +
    geom_density(alpha = .2, fill = "#FF6666") +
    xlim(-250, 250) +
    ylim(0, NA) +
    theme(text = element_text(size = 30)) +
    labs(x = "Growth Rate", y = "Density")

ggsave("plots/growth_rate_distribution.png", width = 10, height = 10, dpi = 300)


# plot density plot of working_senior_ratio for year 2010 and 2021 in all places
plot_df <- all_ds %>% filter(state != 0 & place != "0") %>% select(year, working_senior_ratio) %>% filter(year == 2010 | year == 2021)

ggplot(as.data.frame(plot_df), aes(x = working_senior_ratio, group = year, fill = as.factor(year))) + 
    geom_density(alpha = .2) +
    xlim(-8, 10) +
    ylim(0, NA) +
    theme(text = element_text(size = 30)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "red", size = 1) + 
    scale_color_brewer(palette="Dark2") +
    labs(fill = "Year", x = "Working/Senior Ratio", y = "Density") +
    theme(legend.position = "bottom", legend.text = element_text(size = 25))

ggsave("plots/work_senior_ratio_distribution.png", width = 10, height = 10, dpi = 300)

# plot stacked bar chart of youth, working, senior age group, for each year, along with working_total_ratio, senior_total_ratio line chart for whole us
plot_df <- all_ds %>% filter(state == 0 & place == "0") %>% select(year, youth, working, senior, working_total_ratio, senior_total_ratio)
plot_df <- plot_df %>% pivot_longer(cols = c(youth, working, senior), names_to = "age_group", values_to = "population")
plot_df <- plot_df %>% pivot_longer(cols = c(working_total_ratio, senior_total_ratio), names_to = "ratio_type", values_to = "ratio")
max_total <- all_ds %>% filter(place == "0" & state == 0) %>% select(S0101_C01_001E) %>% max() %>% as.numeric()

ggplot(as.data.frame(plot_df), aes(x = year)) + 
    geom_bar(aes(y = population / 2, fill = age_group), stat = "identity") +
    scale_fill_viridis(name = "Age Group", labels = c("Senior", "Working", "Youth"), discrete = TRUE) +
    labs(fill = "Age Group") +
    ylab("Population") +
    geom_line(aes(y = ratio * max_total / 100, group = ratio_type, linetype = ratio_type), size=2) +
    labs(color = "Ratio Type", x = "Year") +
    scale_linetype_discrete(name = "Ratio Type", labels = c("Senior", "Working")) +
    scale_y_continuous(sec.axis = sec_axis(~ . / max_total * 100, name = "Ratio")) +
    theme(text = element_text(size = 30), legend.position = "bottom", legend.text = element_text(size = 25)) +
    guides(fill=guide_legend(keywidth=1 , nrow=3, byrow=FALSE), linetype=guide_legend(keywidth=4, nrow=3, byrow=FALSE))

ggsave("plots/population_combination_plot_us.png", width = 10, height = 10, dpi = 300)

