## code to generate plot for overlapping egeds between networks (Figure 3 and Figure S3)

library(ggalluvial)
library(ggplot2)
library(tidyverse)

# issue w directories...need to write them out fully each time
fig_dir <- "plots/"

cortisol_color ='#e06545'
placebo_color = '#b44582' 

store_plot <- list()

###### FULL NETWORK SIZES
net_size <- read.csv("CPM code/result/all_cpm_results_with_pvalues.csv")

pp <- net_size %>% select(c(pheno, stim, pill, pos.network, neg.network)) %>%
  mutate(pheno = ifelse(grepl("Arous", pheno), "Arousal", "Memory"),
         stim = ifelse(grepl("alc", stim), "Emotional", "Neutral")) %>%
  pivot_longer(c(4:5), names_to = "net_dir", values_to = "num_edges") %>%
  mutate(net_dir = ifelse(grepl("neg", net_dir), "Negative", "Positive")) %>%
  ggplot(aes(x = stim, y = num_edges, fill = pill)) + 
  geom_bar(stat = "identity", position = "stack") + facet_grid(net_dir~pheno) +
  coord_flip() +
  labs(y = "Number of edges in network", x = "Run type") +
  scale_fill_manual(values = c(cortisol_color, placebo_color)) +
  guides(fill = "none") +
  theme_classic()

pdf(file = paste0(fig_dir, 'num_edges_network.pdf'),
    width = 8, height = 2, useDingbats = F)
print(pp)
dev.off()



###### MEM V AROUS

# read in data
overlap_dat <- read.csv("results/network_overlap/memory&arousal_network_overlap_percent.txt", header = F)

# function: make sure each network has a bar of length = 1
fill_network_edges = function(df, strat1, strat2, countme) {
  dummy_df <- data.frame(grp = df[[strat1]],
                         extra = df[[strat2]],
                         countvar = df[[countme]])
  
  df2 <- dummy_df %>%
    group_by(grp) %>%
    summarise(tot_edge = sum(countvar),
              outvar = 1 - tot_edge) %>%
    mutate(extra = NA) %>%
    select(c(grp, extra, outvar)) %>%
    rename(!!strat1 := grp,
           !!strat2 := extra,
           !!countme := outvar)
}


# rename columns
order_vars <- c('cort_alc_pos', 'cort_alc_neg', 'cort_tool_pos', 'cort_tool_neg', 
                'plac_alc_pos', 'plac_alc_neg', 'plac_tool_pos', 'plac_tool_neg')

rownames(overlap_dat) <- paste0(order_vars,'_mem')
colnames(overlap_dat) <- paste0(order_vars, '_arous')

overlap_long <- overlap_dat %>%
  mutate(memvar = rownames(overlap_dat)) %>%
  pivot_longer(cols = c(1:8), names_to = 'arousvar', values_to = 'overlap') %>%
  mutate(compar = ifelse(grepl("cort", memvar) & grepl("cort", arousvar), 'Cortisol',
                         ifelse(grepl("plac", memvar) & grepl("plac", arousvar), 'Placebo', 
                                'Across Sessions')),
         direc = ifelse(grepl("pos", memvar) & grepl("pos", arousvar), 'pos2pos',
                        ifelse(grepl("neg", memvar) & grepl("neg", arousvar), 'neg2neg', 
                               'mixed'))) %>%
  mutate(compar = factor(compar, levels = c('Across Sessions', 'Cortisol', 'Placebo')))

### POSITIVE & NEGATIVE SEPARATELY
create_alluvial_plot <- function(direc_use, overlap_long, cortisol_color, placebo_color, fig_dir, store_plot = NULL) {
  # Filter data based on direction
  df_use <- overlap_long %>% filter(direc == direc_use)
  
  # Create edges for memory and arousal
  edges_diff.mem <- fill_network_edges(df_use, "memvar", "arousvar", "overlap") %>%
    mutate(compar = NA,
           direc = direc_use)
  
  edges_diff.arous <- fill_network_edges(df_use, "arousvar", "memvar", "overlap") %>%
    mutate(compar = NA,
           direc = direc_use)
  
  # Combine and prepare data for plotting
  df_use.plot <- rbind(df_use, edges_diff.arous) %>% rbind(edges_diff.mem) %>%
    mutate(
      memstim = case_when(
        grepl('cort_alc', memvar) ~ 'Cortisol: Emotional',
        grepl('cort_tool', memvar) ~ 'Cortisol: Neutral',
        grepl('plac_alc', memvar) ~ 'Placebo: Emotional',
        grepl('plac_tool', memvar) ~ 'Placebo: Neutral',
        TRUE ~ NA_character_
      ),
      arousstim = case_when(
        grepl('cort_alc', arousvar) ~ 'Cortisol: Emotional',
        grepl('cort_tool', arousvar) ~ 'Cortisol: Neutral',
        grepl('plac_alc', arousvar) ~ 'Placebo: Emotional',
        grepl('plac_tool', arousvar) ~ 'Placebo: Neutral',
        TRUE ~ NA_character_
      )
    )
  
  # Create plot
  pp <- ggplot(as.data.frame(df_use.plot),
               aes(y = overlap, axis1 = memstim, axis2 = arousstim)) +
    geom_alluvium(aes(fill = compar), width = 1/20, alpha = .7) +
    geom_stratum(width = 1/20, fill = "gray", color = "white") +
    scale_x_discrete(limits = c("Memory", "Arousal")) +
    scale_y_continuous(limits = c(0,4)) +
    scale_fill_manual(values = c("gray", cortisol_color, placebo_color)) +
    theme_classic(base_size = 12) + 
    labs(x = "", y = "") + 
    theme(axis.text.y = element_blank(), 
          axis.ticks.y = element_blank(), 
          axis.line = element_blank()) +
    guides(fill = "none")
  
  # Store plot if store_plot is provided
  if (!is.null(store_plot)) {
    store_plot[[paste0('mem2arous_', direc_use)]] <- pp
  }
  
  # Save plot to PDF
  pdf(file = paste0(fig_dir, 'alluvial_mem2arous_', direc_use, '.pdf'),
      width = 4, height = 3, useDingbats = FALSE)
  print(pp)
  dev.off()
  
  # Return both the plot and updated store_plot list
  return(list(plot = pp, store_plot = store_plot))
}
## pos networks
result <- create_alluvial_plot(
  direc_use = "pos2pos",
  overlap_long = overlap_long,
  cortisol_color = cortisol_color,
  placebo_color = placebo_color,
  fig_dir = fig_dir,
  store_plot = store_plot
)
# Access the plot and updated store_plot
pp <- result$plot
store_plot <- result$store_plot

## neg networks
result <- create_alluvial_plot(
  direc_use = "neg2neg",
  overlap_long = overlap_long,
  cortisol_color = cortisol_color,
  placebo_color = placebo_color,
  fig_dir = fig_dir,
  store_plot = store_plot
)
# Access the plot and updated store_plot
pp <- result$plot
store_plot <- result$store_plot

###### SAME CONSTRUCT
create_alluvial_plot_same_construct <- function(compar, direc_use = 'pos2pos', fig_dir = NULL) {
  # read in data
  overlap_dat <- read.csv(paste0("results/network_overlap/", compar, "_network_overlap_percent.txt"), header = F)
  
  # helper function: make sure each network has a bar of length = 1
  fill_network_edges = function(df, strat1, strat2, countme) {
    dummy_df <- data.frame(grp = df[[strat1]],
                           extra = df[[strat2]],
                           countvar = df[[countme]])
    
    df2 <- dummy_df %>%
      group_by(grp) %>%
      summarise(tot_edge = sum(countvar),
                outvar = 1 - tot_edge) %>%
      mutate(extra = NA) %>%
      select(c(grp, extra, outvar)) %>%
      rename(!!strat1 := grp,
             !!strat2 := extra,
             !!countme := outvar)
  }
  
  # rename columns
  order_vars <- c('cort_alc_pos', 'cort_alc_neg', 'cort_tool_pos', 'cort_tool_neg', 
                  'plac_alc_pos', 'plac_alc_neg', 'plac_tool_pos', 'plac_tool_neg')
  rownames(overlap_dat) <- order_vars
  colnames(overlap_dat) <- order_vars
  
  # data preparation
  overlap_long <- overlap_dat %>%
    mutate(strat1 = rownames(overlap_dat)) %>%
    pivot_longer(cols = c(1:8), names_to = 'strat2', values_to = 'overlap') %>%
    mutate(compar = ifelse(grepl("cort", strat1) & grepl("cort", strat2), 'Cortisol',
                           ifelse(grepl("plac", strat1) & grepl("plac", strat2), 'Placebo', 
                                  'Across Sessions')),
           direc = ifelse(grepl("pos", strat1) & grepl("pos", strat2), 'pos2pos',
                          ifelse(grepl("neg", strat1) & grepl("neg", strat2), 'neg2neg', 
                                 'mixed'))) %>%
    mutate(compar = factor(compar, levels = c('Across Sessions', 'Cortisol', 'Placebo')))
  
  overlap_long_emosplit <- overlap_long %>%
    filter(grepl("alc", strat1) & grepl("tool", strat2))
  
  # filter by direction
  df_use <- overlap_long_emosplit %>% filter(direc == direc_use)
  
  # prepare edges
  edges_diff.s1 <- fill_network_edges(df_use, "strat1", "strat2", "overlap") %>%
    mutate(compar = NA,
           direc = direc_use)
  edges_diff.s2 <- fill_network_edges(df_use, "strat2", "strat1", "overlap") %>%
    mutate(compar = NA,
           direc = direc_use)
  
  # combine data
  df_use.plot <- rbind(df_use, edges_diff.s1) %>% rbind(edges_diff.s2) %>%
    mutate(s1 = ifelse(grepl('cort_alc', strat1), 'Cortisol: Emotional', 
                       ifelse(grepl('plac_alc', strat1), 'Placebo: Emotional', NA)),
           s2 = ifelse(grepl('cort_tool', strat2), 'Cortisol: Neutral',
                       ifelse(grepl('plac_tool', strat2), 'Placebo: Neutral', NA)))
  
  # create plot
  pp <- ggplot(as.data.frame(df_use.plot),
               aes(y = overlap, axis1 = s1, axis2 = s2)) +
    geom_alluvium(aes(fill = compar), width = 1/20, alpha = .7) +
    geom_stratum(width = 1/20, fill = "gray", color = "white") +
    scale_y_continuous(limits = c(0,2)) +
    scale_fill_manual(values = c("gray", cortisol_color, placebo_color)) +
    theme_classic(base_size = 12) + 
    labs(x = "", y = "") + 
    guides(fill = "none") +
    theme(axis.text = element_blank(), 
          axis.ticks = element_blank(), 
          axis.line = element_blank())
  
  # save plot if fig_dir is provided
  if (!is.null(fig_dir)) {
    pdf(file = paste0(fig_dir, 'alluvial_', compar, '_', direc_use, '.pdf'),
        width = 2, height = 3, useDingbats = F)
    print(pp)
    dev.off()
  }
  
  return(pp)
}

plot_mem_pos <- create_alluvial_plot_same_construct(compar = "memory", direc_use = "pos2pos", fig_dir = fig_dir)
plot_mem_neg <- create_alluvial_plot_same_construct(compar = "memory", direc_use = "neg2neg", fig_dir = fig_dir)
plot_arous_pos <- create_alluvial_plot_same_construct(compar = "arousal", direc_use = "pos2pos", fig_dir = fig_dir)
plot_arous_neg <- create_alluvial_plot_same_construct(compar = "arousal", direc_use = "neg2neg", fig_dir = fig_dir)