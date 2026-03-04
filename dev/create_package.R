setwd("C:\\Users\\29827094\\Documents\\GitHub\\DirichletRF\\")  
getwd()  # Verify path

remove.packages("DirichletRF")


#_______________________________________________________________________________
# 1. Recompile Rcpp bindings from src/
Rcpp::compileAttributes()

# 2. Regenerate NAMESPACE + man/ documentation
devtools::document()

# 3. Check for any issues
devtools::check()

# 4. Install the package
devtools::install()

library(DirichletRF)
ls("package:DirichletRF")



