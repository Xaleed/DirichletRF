#setwd("C:/Users/29827094/Documents/GitHub/DirichletForestParallel")
setwd("C:\\Users\\Khaled\\Documents\\GitHub\\DirichletRF")
# Load devtools
library(devtools)

# Build binary - it will be saved in the parent directory (GitHub folder)
pkg_file <- devtools::build(binary = TRUE)
print(pkg_file)



