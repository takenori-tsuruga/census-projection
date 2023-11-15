if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, censusapi)

# check if data directory exists, if not, create one
if (!dir.exists("data")) {
  dir.create("data")
}

# prepare census data information about city and town in Riverside County
# which will be separated from the whole dataset later as test data
rivco_city_town <- read_csv("rivco_info/rivco_city_town.csv", col_names = F, show_col_types = F)
colnames(rivco_city_town) <- c("Name")

state_code <- read_csv("rivco_info/state_code.csv", show_col_types = F)

county_code <- read_csv("rivco_info/county_code.csv", col_names = F, show_col_types = F)[, 3:4]
colnames(county_code) <- c("FIPS", "Name")

place_code <- read_delim("rivco_info/place_code.csv", delim = "|", show_col_types = F)[, 2:3]
colnames(place_code) <- c("FIPS", "Name")
place_code <- place_code[which(place_code$Name %in% rivco_city_town$Name), ]

cdp_code <- read_delim("rivco_info/cdp_code.csv", delim = "|", show_col_types = F)[, 2:3]
colnames(cdp_code) <- c("FIPS", "Name")
cdp_code <- cdp_code[which(cdp_code$Name %in% rivco_city_town$Name), ]

city_town_code <- rbind(place_code, cdp_code)
# delete entry from conflicting city name from northern CAL "El Cerrito" "El Sobrante"
city_town_code <- filter(city_town_code, FIPS != "21796" & FIPS != "22454")

# Add key to .Renviron
Sys.setenv(CENSUS_KEY=PASTEYOURKEYHERE)
# Reload .Renviron
readRenviron("~/.Renviron")
# Check to see that the expected key is output in your R console
Sys.getenv("CENSUS_KEY")

# construction of query variables
# year before 2017 has different column format than after 2017 for S0101
pop_st <- "S0101"
pop_cols <- sprintf("C%02d", c(1, 3, 5))
pop_cols_16 <- sprintf("C%02d", seq(1:3))
pop_vars <- sprintf("%03dE", seq(1, 19))
pop_table <- expand.grid(pop_st, pop_cols, pop_vars)
pop_table_16 <- expand.grid(pop_st, pop_cols_16, pop_vars)
population_vars <- paste0(pop_table[, 1], "_", pop_table[, 2], "_", pop_table[, 3])
population_vars_16 <- paste0(pop_table_16[, 1], "_", pop_table_16[, 2], "_", pop_table_16[, 3])
population_vars <- population_vars[sort.list(population_vars)]
population_vars_16 <- population_vars_16[sort.list(population_vars_16)]
mig_st <- "S0701"
mig_cols <- sprintf("C%02d", seq(1:5))
mig_vars <- sprintf("%03dE", seq(1, 53))
mig_table <- expand.grid(mig_st, mig_cols, mig_vars)
migration_vars <- paste0(mig_table[, 1], "_", mig_table[, 2], "_", mig_table[, 3])
migration_vars <- migration_vars[sort.list(migration_vars)]
fer_st <- "S1301"
fer_cols <- sprintf("C%02d", seq(1:5))
fer_vars <- sprintf("%03dE", seq(1, 32))
fer_table <- expand.grid(fer_st, fer_cols, fer_vars)
fertility_vars <- paste0(fer_table[, 1], "_", fer_table[, 2], "_", fer_table[, 3])
fertility_vars <- fertility_vars[sort.list(fertility_vars)]

# loop to get census data from 2010 to 2021
for (y in seq(2010, 2021)) {
  if (y < 2017) {
    query_vars <- c(population_vars_16, migration_vars, fertility_vars)
    acs_5_population_city_town <- getCensus(
      name = "acs/acs5/subject",
      vintage = y,
      vars = query_vars,
      region = "place:*",
      regionin = "state:*"
    )
  } else {
    query_vars <- c(population_vars, migration_vars, fertility_vars)
    acs_5_population_city_town <- getCensus(
      name = "acs/acs5/subject",
      vintage = y,
      vars = query_vars,
      region = "place:*",
      regionin = "state:*"
    )
  }

  # most of data in S0101 before 2017 requires conversion from percentage to number
  if (y < 2017) {
    tmp <- transmute(rowwise(acs_5_population_city_town), across("S0101_C01_002E":"S0101_C01_019E", ~ round(S0101_C01_001E * .x / 100)))
    convert_col <- grep("_C01_", population_vars_16, value = T)
    convert_col <- convert_col[which(convert_col != "S0101_C01_001E")]
    for (i in convert_col) {
      acs_5_population_city_town[, `i`] <- tmp[, `i`]
    }
    tmp <- transmute(rowwise(acs_5_population_city_town), across("S0101_C02_002E":"S0101_C02_019E", ~ round(S0101_C02_001E * .x / 100)))
    convert_col <- grep("_C02_", population_vars_16, value = T)
    convert_col <- convert_col[which(convert_col != "S0101_C02_001E")]
    for (i in convert_col) {
      acs_5_population_city_town[, `i`] <- tmp[, `i`]
    }
    tmp <- transmute(rowwise(acs_5_population_city_town), across("S0101_C03_002E":"S0101_C03_019E", ~ round(S0101_C03_001E * .x / 100)))
    convert_col <- grep("_C03_", population_vars_16, value = T)
    convert_col <- convert_col[which(convert_col != "S0101_C03_001E")]
    for (i in convert_col) {
      acs_5_population_city_town[, `i`] <- tmp[, `i`]
    }
  }

  # separates data for Riverside County (test data)
  acs_5 <- filter(acs_5_population_city_town, !((place %in% pull(city_town_code, FIPS)) & (state == "06")))
  rivco_acs_5 <- filter(acs_5_population_city_town, ((place %in% pull(city_town_code, FIPS)) & (state == "06")))
  rm(acs_5_population_city_town)

  tmp <- c()
  for (i in seq(nrow(acs_5))) {
    tmp <- append(tmp, y)
  }
  acs_5$year <- tmp
  tmp <- c()
  for (i in seq(nrow(rivco_acs_5))) {
    tmp <- append(tmp, y)
  }
  rivco_acs_5$year <- tmp
  rm(tmp)

  acs_5 <- relocate(acs_5, year, .before = 1)
  rivco_acs_5 <- relocate(rivco_acs_5, year, .before = 1)

  # store 1 year data in a variable named acs_5_2010, acs_5_2011, etc.
  assign(paste0("acs_5_", y), acs_5)
  assign(paste0("rivco_acs_5_", y), rivco_acs_5)
  rm(acs_5)
  rm(rivco_acs_5)
}

# rename some column names of acs_5_2010 to acs_5_2016 to align with acs_5_2017 and later
colnames(acs_5_2010) <- gsub("S0101_C03_", "S0101_C05_", colnames(acs_5_2010))
colnames(acs_5_2010) <- gsub("S0101_C02_", "S0101_C03_", colnames(acs_5_2010))
colnames(acs_5_2011) <- gsub("S0101_C03_", "S0101_C05_", colnames(acs_5_2011))
colnames(acs_5_2011) <- gsub("S0101_C02_", "S0101_C03_", colnames(acs_5_2011))
colnames(acs_5_2012) <- gsub("S0101_C03_", "S0101_C05_", colnames(acs_5_2012))
colnames(acs_5_2012) <- gsub("S0101_C02_", "S0101_C03_", colnames(acs_5_2012))
colnames(acs_5_2013) <- gsub("S0101_C03_", "S0101_C05_", colnames(acs_5_2013))
colnames(acs_5_2013) <- gsub("S0101_C02_", "S0101_C03_", colnames(acs_5_2013))
colnames(acs_5_2014) <- gsub("S0101_C03_", "S0101_C05_", colnames(acs_5_2014))
colnames(acs_5_2014) <- gsub("S0101_C02_", "S0101_C03_", colnames(acs_5_2014))
colnames(acs_5_2015) <- gsub("S0101_C03_", "S0101_C05_", colnames(acs_5_2015))
colnames(acs_5_2015) <- gsub("S0101_C02_", "S0101_C03_", colnames(acs_5_2015))
colnames(acs_5_2016) <- gsub("S0101_C03_", "S0101_C05_", colnames(acs_5_2016))
colnames(acs_5_2016) <- gsub("S0101_C02_", "S0101_C03_", colnames(acs_5_2016))

# stack all data from 2010 to 2021
out_df <- bind_rows(acs_5_2010, acs_5_2011)
out_df <- bind_rows(out_df, acs_5_2012)
out_df <- bind_rows(out_df, acs_5_2013)
out_df <- bind_rows(out_df, acs_5_2014)
out_df <- bind_rows(out_df, acs_5_2015)
out_df <- bind_rows(out_df, acs_5_2016)
out_df <- bind_rows(out_df, acs_5_2017)
out_df <- bind_rows(out_df, acs_5_2018)
out_df <- bind_rows(out_df, acs_5_2019)
out_df <- bind_rows(out_df, acs_5_2020)
out_df <- bind_rows(out_df, acs_5_2021)

out_df <- out_df[order(out_df$year, out_df$state, out_df$place), ]

# # set value to 0 if if a cell contains number less than or equal to -111111111
out_df[is.na(out_df)] <- 0
out_df[out_df <= -111111111] <- 0

# find each column average value for columns 4 to ncol(out_df)
col_ave <- out_df %>% select(4:ncol(out_df)) %>% summarise_all(mean)
# get column names with average value of 0 from col_ave
zero_cols <- colnames(col_ave)[col_ave == 0]
# remove columns with name in zero_cols
out_df <- out_df[, !names(out_df) %in% zero_cols]


# save out_df to csv for training
write.csv(out_df, "data/census-training-pull.csv", row.names = FALSE)

# apply the same porces to test data
colnames(rivco_acs_5_2010) <- gsub("S0101_C03_", "S0101_C05_", colnames(rivco_acs_5_2010))
colnames(rivco_acs_5_2010) <- gsub("S0101_C02_", "S0101_C03_", colnames(rivco_acs_5_2010))
colnames(rivco_acs_5_2011) <- gsub("S0101_C03_", "S0101_C05_", colnames(rivco_acs_5_2011))
colnames(rivco_acs_5_2011) <- gsub("S0101_C02_", "S0101_C03_", colnames(rivco_acs_5_2011))
colnames(rivco_acs_5_2012) <- gsub("S0101_C03_", "S0101_C05_", colnames(rivco_acs_5_2012))
colnames(rivco_acs_5_2012) <- gsub("S0101_C02_", "S0101_C03_", colnames(rivco_acs_5_2012))
colnames(rivco_acs_5_2013) <- gsub("S0101_C03_", "S0101_C05_", colnames(rivco_acs_5_2013))
colnames(rivco_acs_5_2013) <- gsub("S0101_C02_", "S0101_C03_", colnames(rivco_acs_5_2013))
colnames(rivco_acs_5_2014) <- gsub("S0101_C03_", "S0101_C05_", colnames(rivco_acs_5_2014))
colnames(rivco_acs_5_2014) <- gsub("S0101_C02_", "S0101_C03_", colnames(rivco_acs_5_2014))
colnames(rivco_acs_5_2015) <- gsub("S0101_C03_", "S0101_C05_", colnames(rivco_acs_5_2015))
colnames(rivco_acs_5_2015) <- gsub("S0101_C02_", "S0101_C03_", colnames(rivco_acs_5_2015))
colnames(rivco_acs_5_2016) <- gsub("S0101_C03_", "S0101_C05_", colnames(rivco_acs_5_2016))
colnames(rivco_acs_5_2016) <- gsub("S0101_C02_", "S0101_C03_", colnames(rivco_acs_5_2016))

out_df <- bind_rows(rivco_acs_5_2010, rivco_acs_5_2011)
out_df <- bind_rows(out_df, rivco_acs_5_2012)
out_df <- bind_rows(out_df, rivco_acs_5_2013)
out_df <- bind_rows(out_df, rivco_acs_5_2014)
out_df <- bind_rows(out_df, rivco_acs_5_2015)
out_df <- bind_rows(out_df, rivco_acs_5_2016)
out_df <- bind_rows(out_df, rivco_acs_5_2017)
out_df <- bind_rows(out_df, rivco_acs_5_2018)
out_df <- bind_rows(out_df, rivco_acs_5_2019)
out_df <- bind_rows(out_df, rivco_acs_5_2020)
out_df <- bind_rows(out_df, rivco_acs_5_2021)

# convert place code with place name for test data
for (i in seq(nrow(out_df))) {
  out_df$place[i] <- city_town_code$Name[which(city_town_code$FIPS %in% out_df$place[i])]
}

out_df <- out_df[order(out_df$year, out_df$state, out_df$place), ]

# # set value to 0 if if a cell contains number less than or equal to -111111111
out_df[is.na(out_df)] <- 0
out_df[out_df <= -111111111] <- 0

# find each column average value for columns 4 to ncol(out_df)
col_ave <- out_df %>% select(4:ncol(out_df)) %>% summarise_all(mean)
# get column names with average value of 0 from col_ave
zero_cols <- colnames(col_ave)[col_ave == 0]
# remove columns with name in zero_cols
out_df <- out_df[, !names(out_df) %in% zero_cols]

# save out_df to csv as test data for prediction
write.csv(out_df, "data/census-predbase-pull.csv", row.names = FALSE)
