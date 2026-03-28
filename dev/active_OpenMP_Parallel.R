





src_path <- "C:/Users/29827094/Documents/GitHub/DirichletRF/src"

# ============================================================
# Overwrite Makevars
# ============================================================
makevars_path <- file.path(src_path, "Makevars")

con <- file(makevars_path, "w")
writeLines(
  c("PKG_CXXFLAGS = $(SHLIB_OPENMP_CXXFLAGS)",
    "PKG_LIBS = $(SHLIB_OPENMP_CFLAGS)"),
  con
)
close(con)

# ============================================================
# Overwrite Makevars.win
# ============================================================
makevars_win_path <- file.path(src_path, "Makevars.win")

con <- file(makevars_win_path, "w")
writeLines(
  c("PKG_CXXFLAGS = $(SHLIB_OPENMP_CXXFLAGS)",
    "PKG_LIBS = $(SHLIB_OPENMP_CFLAGS)"),
  con
)
close(con)

# ============================================================
# Verify — should show NEW contents
# ============================================================
cat("Makevars:\n")
cat(readLines(makevars_path), sep = "\n")

cat("\n\nMakevars.win:\n")
cat(readLines(makevars_win_path), sep = "\n")

