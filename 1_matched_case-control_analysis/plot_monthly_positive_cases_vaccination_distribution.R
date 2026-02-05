# =========================================================================
# --- Figure S3 ---
# =========================================================================

# 1. Map original column names -> Display names
pathogen_labels <- c(
  "InfA" = "IAV",
  "InfB" = "IBV",
  "Mp"   = "MP",
  "HCOV" = "HCoV",
  "HADV" = "HAdV",
  "HRSV" = "RSV",
  "HMPV" = "HMPV",
  "Ch"   = "Ch",
  "HRV"  = "HRV",
  "HPIV" = "HPIV",
  "Boca" = "HBoV"
)

# Specify order
pathogen_order <- names(pathogen_labels)

# Data preprocessing
proportion_df <- proportion_df %>%
  mutate(
    YearMonth = as.Date(paste0(YearMonth, "-01")),
    Pathogen = factor(Pathogen, levels = pathogen_order),
    Pathogen_label = factor(pathogen_labels[Pathogen], levels = pathogen_labels)  # Labels for display
  )

# Plot heatmap
ggplot(proportion_df, aes(x = YearMonth, y = Pathogen_label, fill = Proportion)) +
  geom_tile(color = "white") +  # Color tiles
  geom_text(aes(label = sprintf("%.2f%%", Proportion * 100)), size = 3, color = "black") +  # Add numbers on each tile
  scale_fill_gradientn(
    colors = c("#f7fbff", "#6baed6", "#08306b"),
    name = "Positive cases (%)",
    limits = c(0, max(proportion_df$Proportion, na.rm = TRUE)),
    labels = scales::label_percent(accuracy = 1)  # Format percentage
  ) +
  scale_x_date(
    date_breaks = "1 month",
    date_labels = "%Y-%m",
    expand = c(0, 0)
  ) +
  labs(
    x = "Month",
    y = "Pathogen"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_blank(),
    panel.grid = element_blank()
  )

ggsave("heatmap.png", width = 14, height = 6, dpi = 4000)


# =========================================================================
# --- Figure S4 ---
# =========================================================================

# Ensure date format is correct
Vacc2year$Pre_vaccination_date <- as.Date(Vacc2year$Pre_vaccination_date)

# Extract Year-Month
Vacc2year <- Vacc2year %>%
  mutate(YearMonth = format(Pre_vaccination_date, "%Y-%m"))

# 1. Total count per month
month_count <- Vacc2year %>%
  filter(!is.na(Pre_vaccination_date)) %>%
  count(YearMonth, name = "Total_Count")

# 2. Count of each vaccine type per month
month_type_count <- Vacc2year %>%
  filter(!is.na(Pre_vaccination_date) & !is.na(Pre_vaccination_type)) %>%
  count(YearMonth, Pre_vaccination_type, name = "Type_Count")

# 3. Convert to wide format (each vaccine type becomes a column)
wide_type_count <- month_type_count %>%
  pivot_wider(
    names_from = Pre_vaccination_type,
    values_from = Type_Count,
    values_fill = 0
  )

# 4. Merge total count column
final_result <- left_join(wide_type_count, month_count, by = "YearMonth")

# View results
print(final_result)

##Calculate weekly vaccination counts and proportions##
# Ensure date format is correct
Vacc2year$Pre_vaccination_date <- as.Date(Vacc2year$Pre_vaccination_date)

# Extract Year-Week (ISO week number)
Vacc2year <- Vacc2year %>%
  mutate(YearWeek = paste0(year(Pre_vaccination_date), "-W", sprintf("%02d", isoweek(Pre_vaccination_date))))

# 1. Total count per week
week_count <- Vacc2year %>%
  filter(!is.na(Pre_vaccination_date)) %>%
  count(YearWeek, name = "Total_Count")

# 2. Count of each vaccine type per week
week_type_count <- Vacc2year %>%
  filter(!is.na(Pre_vaccination_date) & !is.na(Pre_vaccination_type)) %>%
  count(YearWeek, Pre_vaccination_type, name = "Type_Count")

# 3. Convert to wide format (each vaccine type becomes a column)
wide_type_count <- week_type_count %>%
  pivot_wider(
    names_from = Pre_vaccination_type,
    values_from = Type_Count,
    values_fill = 0
  )

# 4. Merge total count column
final_week_result <- left_join(wide_type_count, week_count, by = "YearWeek")

# 5. Calculate proportion columns
final_week_result <- final_week_result %>%
  mutate(across(-c(YearWeek, Total_Count), ~ .x / Total_Count, .names = "{.col}_prop"))

# View results
print(final_week_result)

# Identify the three categories of columns
quadrivalent_cols <- grep("Quadrivalent", names(final_result), value = TRUE)
trivalent_cols <- grep("Trivalent", names(final_result), value = TRUE)
other_cols <- setdiff(names(final_result),
                      c("YearMonth", "Total_Count", quadrivalent_cols, trivalent_cols))

# Aggregate the three categories of vaccine columns
final_result <- final_result %>%
  rowwise() %>%
  mutate(
    Quadrivalent_Total = sum(c_across(all_of(quadrivalent_cols)), na.rm = TRUE),
    Trivalent_Total = sum(c_across(all_of(trivalent_cols)), na.rm = TRUE),
    Other_Total = sum(c_across(all_of(other_cols)), na.rm = TRUE)
  ) %>%
  ungroup() %>%
  select(YearMonth, Quadrivalent_Total, Trivalent_Total, Other_Total, Total_Count)

# View results
print(final_result)

# 确保 YearMonth 是 factor（按时间排序）
final_result$YearMonth <- factor(final_result$YearMonth, levels = final_result$YearMonth)
# 计算累积数量
final_result <- final_result %>%
  mutate(Cumulative_Count = cumsum(Total_Count))

p1 <- ggplot(final_result, aes(x = YearMonth, y = Cumulative_Count, group = 1)) +
  geom_line(color = "darkred", size = 0.5) +
  geom_point(color = "darkred", size = 1) +
  labs(
    y = "Cumulative Count",
    x = NULL
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_blank(),  # Do not display X-axis labels (will be shown in the bottom plot)
    axis.title.x = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  )

# Calculate proportions for each type and convert to long format
plot_data <- final_result %>%
  mutate(
    Trivalent_Prop = Trivalent_Total / Total_Count,
    Quadrivalent_Prop = Quadrivalent_Total / Total_Count,
    Other_Prop = Other_Total / Total_Count
  ) %>%
  select(YearMonth, Total_Count, Trivalent_Prop, Quadrivalent_Prop, Other_Prop) %>%
  pivot_longer(cols = ends_with("Prop"), names_to = "Vaccine_Type", values_to = "Proportion")

# Set stacking order (bottom to top) and legend order
plot_data$Vaccine_Type <- factor(
  plot_data$Vaccine_Type,
  levels = c("Trivalent_Prop", "Quadrivalent_Prop", "Other_Prop"),
  labels = c("Trivalent", "Quadrivalent", "Other types")
)

p2 <- ggplot(plot_data, aes(x = YearMonth)) +
  geom_col(aes(y = Proportion * max(final_result$Total_Count), fill = Vaccine_Type), position = "stack") +
  geom_line(data = final_result, aes(x = YearMonth, y = Total_Count, group = 1), 
            color = "black", size = 0.5) +
  geom_point(data = final_result, aes(x = YearMonth, y = Total_Count), 
             color = "black", size = 1) +
  scale_y_continuous(
    name = "Total Count",
    sec.axis = sec_axis(
      transform = ~ . / max(final_result$Total_Count),
      name = "Proportion (%)",
      labels = percent_format(accuracy = 1)
    )
  ) +
  scale_fill_manual(
    values = c(
      "Trivalent" = "#E7F2EE",      
      "Quadrivalent" = "#C8E0D9",   
      "Other types" = "#83BDB0"     
    )
  ) +
  labs(x = "Month", fill = "Last vaccine type") +
  theme_minimal() +
  theme(
    panel.grid.major.x = element_blank(),  # Remove major X-axis grid lines
    panel.grid.minor.x = element_blank(),  # Remove minor X-axis grid lines
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.title.y.left = element_text(color = "black"),
    axis.title.y.right = element_text(color = "black")
  )

# Top plot is cumulative curve, bottom plot is stacked chart
p1 / p2 + plot_layout(heights = c(1, 2))

ggsave("VaccType_num.png", width = 20, height = 12, dpi = 2000)

