# =========================================================================
# --- Figure 3 ---
# =========================================================================

# Define data frame list and corresponding names
df_list <- list(
  df_InfA, df_InfB, df_Mp, df_HCOV, df_HADV, df_HRSV, 
  df_HMPV, df_Ch, df_HRV, df_HPIV, df_Boca
)
pathogen_names <- c(
  "IAV", "IBV", "MP", "HCoV", "HAdV", "RSV", 
  "HMPV", "Ch", "HRV", "HPIV", "HBoV"
)

# Use a combination of dplyr and purrr to efficiently combine data
combined_df <- purrr::map2_dfr(
  df_list, 
  pathogen_names,
  ~ .x %>% mutate(Pathogen = .y)
)

# Convert Pathogen column to factor and specify order
combined_df$Pathogen <- factor(combined_df$Pathogen, levels = pathogen_names)

# This function calculates custom quantiles for each group
prepare_boxplot_data <- function(df) {
  df %>%
    group_by(Pathogen) %>%
    summarise(
      ymin = quantile(OR, 0.025, na.rm = TRUE),
      lower = quantile(OR, 0.25, na.rm = TRUE),
      middle = quantile(OR, 0.5, na.rm = TRUE),
      upper = quantile(OR, 0.75, na.rm = TRUE),
      ymax = quantile(OR, 0.975, na.rm = TRUE)
    )
}

# Prepare boxplot data for plotting
box_df <- prepare_boxplot_data(combined_df)

# Custom colors (11 types)
my_11_colors <- c(
  "#284179", "#E86976", "#314036", "#d62728", "#9467bd", 
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#5CB85C"
)

# Start plotting
p_wide <- ggplot(combined_df, aes(x = Pathogen, y = OR, fill = Pathogen, color = Pathogen)) +
  
  # Add reference dashed line for OR=1
  geom_hline(yintercept = 1, linetype = "dashed", color = "black", linewidth = 0.6) + # Modification: Make line black and slightly thicker
  
  # Violin plot (right side)
  gghalves::geom_half_violin(
    side = "r", 
    alpha = 0.3, 
    width = 0.9, 
    trim = TRUE,
    position = position_nudge(x = 0.07)
  ) +
  
  # Custom boxplot
  geom_boxplot(
    data = box_df,
    aes(
      x = Pathogen, ymin = ymin, lower = lower, middle = middle,
      upper = upper, ymax = ymax, fill = Pathogen, color = Pathogen
    ),
    stat = "identity",
    position = position_nudge(x = -0.2),
    alpha = 0.6,
    width = 0.25, 
    inherit.aes = FALSE
  ) +
  
  # Mean points
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 23,
    size = 2,
    fill = "white",
    color = "black",
  ) +
  
  # Scatter plot (jitter)
  geom_point(
    position = position_jitter(width = 0.05, height = 0), 
    size = 0.5,
    alpha = 0.5
  ) +
  
  # Apply custom colors
  scale_fill_manual(values = my_11_colors) +
  scale_color_manual(values = my_11_colors) +
  
  # Theme and label settings
  theme_bw(base_size = 18) + 
  
  theme(
    # Remove X-axis title
    axis.title.x = element_blank(),
    
    # X-axis text: black, size 18
    axis.text.x = element_text(size = 18, color = "black", face = "plain"), 
    
    # Y-axis title: black, bold, size 18
    axis.title.y = element_text(size = 18, color = "black", face = "bold"),
    
    # Y-axis text: black, size 18
    axis.text.y = element_text(size = 18, color = "black"),
    
    # Set tick marks and border color to pure black
    axis.ticks = element_line(color = "black"),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    
    # Remove legend
    legend.position = "none"
  ) +
  ylab("OR") 

# Print the plot
print(p_wide)


# =========================================================================
# --- Figure 4 ---
# =========================================================================

# Ensure necessary libraries are loaded
library(ggplot2)
library(ggpubr)
library(gghalves)

# Calculate custom quantiles (2.5%, 25%, 50%, 75%, 97.5%) for the boxplot
prepare_or_box_data <- function(df, time_label) {
  or_vals <- df$OR
  qs <- quantile(or_vals, probs = c(0.025, 0.25, 0.5, 0.75, 0.975), na.rm = TRUE)
  data.frame(
    Time = time_label,
    ymin = qs[1],   # 2.5%
    lower = qs[2],  # 25%
    middle = qs[3], # 50%
    upper = qs[4],  # 75%
    ymax = qs[5]    # 97.5%
  )
}

# Create OR plotting function
create_or_plot <- function(data6, data12, data24, title, colors) {
  
  # Print means and quantiles
  summary_6 <- list(
    mean = mean(data6$OR, na.rm = TRUE),
    quantiles = quantile(data6$OR, probs = c(0.025, 0.975), na.rm = TRUE)
  )
  summary_12 <- list(
    mean = mean(data12$OR, na.rm = TRUE),
    quantiles = quantile(data12$OR, probs = c(0.025, 0.975), na.rm = TRUE)
  )
  summary_24 <- list(
    mean = mean(data24$OR, na.rm = TRUE),
    quantiles = quantile(data24$OR, probs = c(0.025, 0.975), na.rm = TRUE)
  )
  
  cat(paste0("\n--- [Statistical Results for ", title, "] ---\n"))
  cat(sprintf("0-6 months:  Mean = %.4f;  2.5th = %.4f;  97.5th = %.4f\n", summary_6$mean, summary_6$quantiles[1], summary_6$quantiles[2]))
  cat(sprintf("6-12 months: Mean = %.4f;  2.5th = %.4f;  97.5th = %.4f\n", summary_12$mean, summary_12$quantiles[1], summary_12$quantiles[2]))
  cat(sprintf("12-24 months: Mean = %.4f;  2.5th = %.4f;  97.5th = %.4f\n", summary_24$mean, summary_24$quantiles[1], summary_24$quantiles[2]))
  cat("----------------------------------\n")
  
  # Plotting section
  df_6 <- data.frame(OR = data6$OR, Time = "0-6")
  df_12 <- data.frame(OR = data12$OR, Time = "6-12")
  df_24 <- data.frame(OR = data24$OR, Time = "12-24")
  
  df_all <- rbind(df_6, df_12, df_24)
  df_all$Time <- factor(df_all$Time, levels = c("0-6", "6-12", "12-24"))
  
  # Scatter jitter
  df_all$x_num <- as.numeric(df_all$Time)
  df_all$x_num_jittered <- df_all$x_num + 0.2
  
  # Custom boxplot data
  box_df <- rbind(
    prepare_or_box_data(data6, "0-6"),
    prepare_or_box_data(data12, "6-12"),
    prepare_or_box_data(data24, "12-24")
  )
  box_df$Time <- factor(box_df$Time, levels = c("0-6", "6-12", "12-24"))
  
  # Plot the graph
  ggplot(df_all, aes(x = Time, y = OR, fill = Time, color = Time)) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
    gghalves::geom_half_violin(
      side = "r",
      position = position_nudge(x = 0.23),
      alpha = 0.3,
      width = 0.6,
      trim = TRUE
    ) +
    geom_boxplot(
      data = box_df,
      aes(
        x = Time,
        ymin = ymin,
        lower = lower,
        middle = middle,
        upper = upper,
        ymax = ymax,
        fill = Time,
        color = Time
      ),
      stat = "identity",
      position = position_nudge(x = 0.05),
      alpha = 0.6,
      width = 0.15,
      inherit.aes = FALSE
    ) +
    stat_summary(
      aes(color = Time),
      fun = mean,
      geom = "point",
      shape = 23,
      size = 1.5,
      fill = "white",
      position = position_nudge(x = 0.05)
    ) +
    geom_point(
      aes(x = x_num_jittered),
      position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.15),
      size = 0.5,
      alpha = 0.7
    ) +
    scale_fill_manual(values = colors) +
    scale_color_manual(values = colors) +
    theme_bw() +
    theme(
      axis.title.x = element_text(size = 14, color = "black", margin = margin(t = 2)),
      axis.title.y = element_text(size = 14, color = "black", margin = margin(r = 2)),
      axis.text.x = element_text(size = 14, color = "black"),
      axis.text.y = element_text(size = 14, color = "black"),
      legend.position = "none",
      plot.title = element_text(hjust = 0.5, size = 14, color = "black")
    ) +
    ylab("OR") +
    xlab("Time (Month)") +
    ggtitle(title) +
    stat_compare_means(
      method = "t.test",
      comparisons = list(
        c("0-6", "6-12"),
        c("0-6", "12-24"),
        c("6-12", "12-24")
      ),
      label = "p.signif"
    )
}

# Run and output the plot
my_colors <- c("#5CB85C", "#337AB7", "#F0AD4E")

p1 <- create_or_plot(Loop_results_InfA_6months, Loop_results_InfA_12months, Loop_results_InfA_24months, "IAV", my_colors)
p2 <- create_or_plot(Loop_results_InfB_6months, Loop_results_InfB_12months, Loop_results_InfB_24months, "IBV", my_colors)
p3 <- create_or_plot(Loop_results_Mp_6months, Loop_results_Mp_12months, Loop_results_Mp_24months, "MP", my_colors)

png("OR_Time_InfA_InfB_Mp.png", width = 12, height = 3.9, units = "in", res = 1200)
ggpubr::ggarrange(p1, p2, p3, ncol = 3, nrow = 1, labels = NULL, align = "hv")
dev.off()
