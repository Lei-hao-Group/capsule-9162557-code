# =========================================================================
# --- 0_prep_data ---
# =========================================================================

# Load necessary packages
library(readxl)
library(dplyr)
library(lubridate)
library(MatchIt)
library(survival)
library(purrr)  # For loop operations
library(ggplot2)
library(gridExtra)
library(openxlsx)
library(stringr)
library(MatchIt)
library(survival)
library(parallel)
library(pbapply)
library(tidyr)
library(patchwork)

# Get all Excel filenames (assuming filenames are sequentially named)
file_names <- sprintf("2023%02d.xlsx", 1:12)  
file_names <- c(file_names, sprintf("2024%02d.xlsx", 1:12))  

# Initialize an empty dataframe to store the combined data
combined_data <- data.frame()

# Iterate through each file and read data
for (i in seq_along(file_names)) {
  file <- file_names[i]
  
  # Read Excel file
  data <- read_excel(file)
  
  # If it's the first file, keep column names
  if (i == 1) {
    combined_data <- data
  } else {
    # Remove column names (header row), only merge content
    combined_data <- bind_rows(combined_data, data[-1, ])
  }
}

# View combined data
print(combined_data)

table1 <- read_excel("D:/data.xlsx")
table2 <- combined_data

# Ensure ID and OUTPATIENT field types are consistent
table1 <- table1 %>%
  mutate(ID = as.character(ID))  # If ID is numeric, convert to character

table2 <- table2 %>%
  mutate(OUTPATIENT = as.character(OUTPATIENT))  # If OUTPATIENT is numeric, convert to character

# Standardize date fields and convert to Date type
table1 <- table1 %>%
  mutate(Date = as.Date(Date, format = "%Y.%m.%d"))  # Date format for table 1

table2 <- table2 %>%
  mutate(CHECK_TIME = as.Date(CHECK_TIME, format = "%Y/%m/%d"))  # Date format for table 2

# Match and merge based on ID and Date
merged_table <- table1 %>%
  left_join(table2, by = c("ID" = "OUTPATIENT", "Date" = "CHECK_TIME"))

# View merged table
print(merged_table)


##########Preliminary Data Cleaning##########

# List column names to process
columns_to_process <- c("InfA", "InfB", "HPIV", "HADV", "Ch", "HRV", "HMPV", "HCOV", "HRSV", "Mp", "Boca")
merged_table <- merged_table %>%
  mutate(across(all_of(columns_to_process), ~ ifelse(. == "Positive(+)", 1, 0)))



# Convert date to Date type
merged_table <- merged_table %>%
  mutate(Date = ymd(Date)) %>%                     
  mutate(YearMonth = format(Date, "%Y-%m"))       



# Conversion function
convert_to_months <- function(age) {
  if (str_detect(age, "Year")) {
    years <- as.numeric(str_extract(age, "\\d+(?=Year)"))
    if (str_detect(age, "Month")) {
      months <- as.numeric(str_extract(age, "\\d+(?=Month)"))
    } else if (str_detect(age, "Day")) {
      days <- as.numeric(str_extract(age, "\\d+(?=Day)"))
      months <- round(days / 30, 1)  # Estimate as 30 days = 1 month
    } else if (str_detect(age, "Hour")) {
      hours <- as.numeric(str_extract(age, "\\d+(?=Hour)"))
      months <- round(hours / (24*30), 2)  # Estimate as 720 hours = 1 month
    } else {
      months <- 0
    }
    total <- years * 12 + months
  } else if (str_detect(age, "Year")) {
    total <- as.numeric(str_extract(age, "\\d+(?=Year)"))
  } else if (str_detect(age, "Day")) {
    days <- as.numeric(str_extract(age, "\\d+(?=Day)"))
    total <- round(days / 30, 1)
  } else {
    total <- NA
  }
  return(total)
}

# Apply conversion
merged_table$Age_months <- sapply(merged_table$Age, convert_to_months)



# Only keep age <= 18 years (216 months)
merged_table <- subset(merged_table, Age_months <= 216)



head(merged_table)



merged_table <- merged_table %>%
  group_by(ID) %>%           # Group by ID
  filter(n() == 1) %>%       # Keep IDs that appear only once
  ungroup()                  # Ungroup
# merged_table <- merged_table %>%
#  group_by(ID) %>%
#  arrange(desc(Date)) %>%  # Sort by date in descending order (newest on top)
#  slice(1) %>%             # Take the first row
#  ungroup()



# Calculate the count of 1s per row (i.e., number of infected pathogens)
merged_table$Infection_count <- rowSums(merged_table[, columns_to_process] == 1)



# Keep only single infections
merged_table <- subset(merged_table, Infection_count <= 1)


########### Merge with Vaccine Data ##########

data_vaccine <- read_excel("D:/data_vaccine.xlsx")



All_data <- merge(merged_table, data_vaccine, by = "ID_CARD", all.x = TRUE)



All_data <- All_data %>%
  filter(
    !is.na(ID_CARD),                     # Remove missing values
    grepl("^\\d{17}[0-9Xx]$", ID_CARD)   # Match 18-digit ID card format
  )


########## Add Necessary Columns ##########

# Define influenza vaccination date column names
flu_cols <- paste0("Influenza", 1:8, "Vaccination time")

# 1. Ensure all date columns are in date format (safe conversion to avoid errors)
All_data <- All_data %>%
  mutate(
    Date = as.Date(Date, tryFormats = c("%Y-%m-%d", "%Y/%m/%d")),
    across(all_of(flu_cols), ~ as.Date(.x, tryFormats = c("%Y-%m-%d", "%Y/%m/%d")))
  )

# 2. Calculate Pre_vaccination_num
All_data <- All_data %>%
  mutate(
    Pre_vaccination_num = case_when(
      year(Date) == 2023 ~ rowSums(
        across(all_of(flu_cols), ~ ifelse(is.na(.x), 0, year(.x) == 2022)),
        na.rm = TRUE
      ),
      year(Date) == 2024 ~ rowSums(
        across(all_of(flu_cols), ~ ifelse(is.na(.x), 0, year(.x) == 2023)),
        na.rm = TRUE
      ),
      TRUE ~ NA_integer_
    )
  )

# Create new column Pre_vaccination_days and initialize as NA
All_data$Pre_vaccination_days <- NA
# Create new column Pre_vaccination_type and initialize as NA
All_data$Pre_vaccination_type <- NA
All_data$Pre_vaccination_date <- as.Date(NA)

# Iterate through each row of the dataset
for (i in 1:nrow(All_data)) {
  current_date <- All_data$Date[i]
  last_vaccine_date <- NULL
  last_vaccine_type <- NULL
  
  # Check vaccination dates for Influenza 1 to 8
  for (j in 1:8) {
    col_name_date <- paste0("INF", j, "Vaccination time")
    col_name_type <- paste0("INF", j, "Product Name")
    
    # Check if date column exists and is not NA
    if (col_name_date %in% names(All_data) && !is.na(All_data[i, col_name_date])) {
      vaccine_date <- All_data[i, col_name_date]
      
      # If vaccination date is earlier than current date, update last vaccination date and type
      if (vaccine_date < current_date) {
        last_vaccine_date <- vaccine_date
        # Check if product name column exists and is not NA
        if (col_name_type %in% names(All_data) && !is.na(All_data[i, col_name_type])) {
          last_vaccine_type <- All_data[i, col_name_type]
        } else {
          last_vaccine_type <- NA
        }
      }
    } else {
      # If date column does not exist or is NA, stop comparison
      break
    }
  }
  
  # If a valid vaccination date is found, calculate day difference and record vaccine type
  if (!is.null(last_vaccine_date)) {
    All_data$Pre_vaccination_days[i] <- as.numeric(current_date - last_vaccine_date)
    All_data$Pre_vaccination_type[i] <- last_vaccine_type
    All_data$Pre_vaccination_date[i] <- last_vaccine_date
  }
}



# Create new column Pre_vaccination_days_cate
All_data$Pre_vaccination_days_cate <- NA

# Use cut function to classify days
All_data$Pre_vaccination_days_cate <- cut(
  All_data$Pre_vaccination_days,
  breaks = c(-Inf, 0, 180, 365, Inf),
  labels = c(NA, 1, 2, 3),  # Corresponding classification labels
  right = FALSE,  # Left-closed, right-open interval [a,b)
  include.lowest = TRUE  # Include lowest value
)

# Convert factor to numeric (if needed)
All_data$Pre_vaccination_days_cate <- as.numeric(as.character(All_data$Pre_vaccination_days_cate))



Vacc2year <- subset(All_data, 
                    is.na(Pre_vaccination_days) | Pre_vaccination_days <= 730)



Vacc2year <- subset(Vacc2year, 
                    is.na(Pre_vaccination_days) | Pre_vaccination_days > 14)



Vacc2year <- Vacc2year %>%
  mutate(Pre_vaccination = ifelse(is.na(Pre_vaccination_days), 0, 1))



# Encode gender as numeric
Vacc2year$Gender_num <- ifelse(Vacc2year$Gender == "Male", 1, 0)

Vacc2year$Month <- month(Vacc2year$Date, label = TRUE, abbr = TRUE)  # label=TRUE returns factor, abbr=TRUE abbreviates month names
Vacc2year$Month_num <- as.numeric(gsub("Month", "", Vacc2year$Month))  # Remove "Month" char and convert to number

Vacc2year$Age_years <- Vacc2year$Age_months %/% 12
