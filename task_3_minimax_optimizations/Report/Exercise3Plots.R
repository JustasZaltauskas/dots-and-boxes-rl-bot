library(readxl)
library(ggplot2)
library(reshape2)
rm(list = ls())
Results <- read_excel("Documents/Master/MLProject/Project/Code/rl-project-KULeuven/Exercise3/Results.xlsx")
attach(Results)
detach(Results)



# TIME

df <- data.frame(Results$Dimension, "Optimized" = Results$TimeOpt * 1000, Naive = Results$TimeUnopt * 1000)
# Melt the data frame to long format
df_long <- melt(df, id.vars = "Results.Dimension", variable.name = "Algorithm", value.name = "Running_Time")

# Plotting
ggplot(df_long, aes(x = Results.Dimension, y = Running_Time, fill = Algorithm)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Input Dimension", y = "Running Time (µs) (log scale)") +
  ggtitle("Running Times of naive vs optimised Minimax") +
  scale_y_log10() +
  theme(plot.title = element_text(hjust = 0.5))

ggplot(df_long, aes(x = Results.Dimension, y = Running_Time, fill = Algorithm)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Input Dimension", y = "Running Time (µs) (normal scale)") +
  ggtitle("Running Times of naive vs optimised Minimax") +
  #scale_y_log10() +
  theme(plot.title = element_text(hjust = 0.5))


# SIZE
df <- data.frame(Results$Dimension, "Optimized" = Results$StatesOpt, Naive = Results$StatesUnopt)
# Melt the data frame to long format
df_long <- melt(df, id.vars = "Results.Dimension", variable.name = "Algorithm", value.name = "Running_Time")

# Plotting
ggplot(df_long, aes(x = Results.Dimension, y = Running_Time, fill = Algorithm)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Input Dimension", y = "States searched (log scale)") +
  ggtitle("States searched of naive vs optimised Minimax") +
  scale_y_log10() +
  theme(plot.title = element_text(hjust = 0.5))

ggplot(df_long, aes(x = Results.Dimension, y = Running_Time, fill = Algorithm)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Input Dimension", y = "States searched (normal scale)") +
  ggtitle("States searched of naive vs optimised Minimax") +
#  scale_y_log10() +
  theme(plot.title = element_text(hjust = 0.5))



# SIZE
df <- data.frame(Results$Dimension, "Optimized" = Results$TableSizeOpt)
# Melt the data frame to long format
df_long <- melt(df, id.vars = "Results.Dimension", variable.name = "Algorithm", value.name = "Running_Time")

# Plotting
ggplot(df_long, aes(x = Results.Dimension, y = Running_Time, fill = Algorithm)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Input Dimension", y = "Entries in transposition table") +
  ggtitle("Size of transposition table") +
  #scale_y_log10() +
  theme(plot.title = element_text(hjust = 0.5))





# Data collection
n <- c(10, 100, 1000, 10000) # Input sizes
algo1_time <- c(0.1, 2.3, 27.4, 347.8) # Running times of algorithm 1
algo2_time <- c(0.2, 4.5, 50.1, 619.2) # Running times of algorithm 2

# Create a data frame
df <- data.frame(n = n, algo1_time = algo1_time, algo2_time = algo2_time)

# Melt the data frame to long format
df_long <- melt(df, id.vars = "n", variable.name = "Algorithm", value.name = "Running_Time")

# Plotting
ggplot(df_long, aes(x = n, y = Running_Time, fill = Algorithm)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Input Size", y = "Running Time (seconds)") +
  ggtitle("Comparing Running Times of Two Algorithms") +
  theme(plot.title = element_text(hjust = 0))
                                  
                                  