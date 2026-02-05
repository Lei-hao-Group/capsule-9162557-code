# =========================================================================
# --- 1:1 bootstrap matching repeated over 1,000 iterations ---
# =========================================================================

# List column names to process
columns_to_process <- c("InfA", "InfB", "HPIV", "HADV", "Ch", "HRV", "HMPV", "HCOV", "HRSV", "Mp", "Boca")
# Set number of parallel cores
n_cores <- detectCores() - 1

analyze_once <- function(i, pathogen) {
  # Shuffle data order
  shuffled_data <- Vacc2year[sample(nrow(Vacc2year)), ]
  
  # Matching
  matched_data <- matchit(
    as.formula(paste(pathogen, "~ Gender_num + Age_years + Month_num + Type")),
    data = shuffled_data,
    method = "nearest",
    distance = "glm",
    ratio = 1,
    exact = ~ Gender_num + Age_years + Month_num + Type,
    discard = "both",
    replace = FALSE,
    caliper = 0.01
  )
  
  ## "optimal" matching method
  # matched_data <- matchit(
  # as.formula(paste(pathogen, "~ Gender_num + Age_years + Month_num + Type")),
  # data = shuffled_data,
  # method = "optimal",
  # ratio = 1,
  # exact = ~ Gender_num + Age_years + Month_num + Type,
  # discard = "both"
  # ) 
  
  # Extract matched data
  matched_df <- match.data(matched_data)
  
  # If matching fails, return NA
  if (nrow(matched_df) == 0) {
    return(data.frame(
      Iteration = i,
      Pathogen = pathogen,
      OR = NA,
      Lower_CI = NA,
      Upper_CI = NA,
      P_value = NA,
      N_cases = NA,
      N_controls = NA
    ))
  }
  
  # Keep strictly 1:1 subclasses
  matched_df <- matched_df %>%
    group_by(subclass) %>%
    filter(n() == 2, sum(!!as.name(pathogen)) == 1) %>%  # 一例为1，一例为0
    ungroup()
  
  # Check again if data still exists
  if (nrow(matched_df) == 0) {
    return(data.frame(
      Iteration = i,
      Pathogen = pathogen,
      OR = NA,
      Lower_CI = NA,
      Upper_CI = NA,
      P_value = NA,
      N_cases = NA,
      N_controls = NA
    ))
  }
  
# =========================================================================
# --- Conditional logistic regression ---
# =========================================================================
  
  model <- clogit(
    as.formula(paste(pathogen, "~ Pre_vaccination + strata(subclass)")),
    data = matched_df
  )
  
  # Extract results
  OR <- exp(coef(model))
  OR_CI <- exp(confint(model))
  p_val <- summary(model)$coefficients["Pre_vaccination", "Pr(>|z|)"]
  
  return(data.frame(
    Iteration = i,
    Pathogen = pathogen,
    OR = OR,
    Lower_CI = OR_CI[1],
    Upper_CI = OR_CI[2],
    P_value = p_val,
    N_cases = sum(matched_df[[pathogen]] == 1),
    N_controls = sum(matched_df[[pathogen]] == 0)
  ))
}

# Run analysis for all pathogens
all_results <- list()

for (pathogen in columns_to_process) {
  message("Processing: ", pathogen)
  
  results <- pblapply(1:1000, function(i) analyze_once(i, pathogen), cl = n_cores)
  
  all_results[[pathogen]] <- do.call(rbind, results)
}

# Combine all pathogen results into one data.frame
final_results <- do.call(rbind, all_results)