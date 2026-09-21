# Regenerate the R-authored fixtures under tests/fixtures/.
#
# These are the ground truth for tests/test_r_export_parity.py: a real
# export_sklearn_model() file, plus the predictions R produces for a held-out
# block of spectra. Running this needs an R session with the proximetricsR
# branch that provides export_sklearn_model() installed; see
# tests/fixtures/README.md.
#
# Usage, from the repo root:
#   Rscript scripts/build_r_fixture.R
#
# The recipe below is chosen to exercise every part of the export that has gone
# wrong before, in one pipeline:
#   prep_wav_trim   -> RangeCut, whose start/end resolve to a half-open slice
#   prep_transform  -> IntensityConversion
#   prep_derivative -> SavitzkyGolay *plus* the appended RangeCut reproducing the
#                      edge points proximetricsR drops and chemotools keeps
#   prep_snv        -> StandardNormalVariate, whose sample-vs-population SD
#                      difference is folded into x_means_/coef_

library(proximetricsR)

set.seed(2026)

data("NIRcannabis", package = "proximetricsR")

cal <- NIRcannabis[1:60, ]
val <- NIRcannabis[61:80, ]

recipe <- preprocess_recipe(
  prep_wav_trim(band = c(1100, 1600)),
  prep_transform(to = "absorbance"),
  prep_derivative(m = 1, w = 11, p = 2, algorithm = "savitzky-golay"),
  prep_snv(),
  device = "unspecified"
)

X <- cal$spc
rownames(X) <- NULL
Y <- matrix(cal$CBDA, dimnames = list(rownames(cal), "CBDA"))

model <- calibrate(
  X, Y,
  data = cal,
  preprocess = recipe,
  method = fit_plsr(ncomp = 5, type = "modified"),
  control = calibration_control("none"),
  verbose = FALSE
)

out <- file.path("tests", "fixtures")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

export_sklearn_model(model, file = file.path(out, "r_exported_pipeline.json"))

X_val <- val$spc
rownames(X_val) <- NULL
y_hat <- predict(model, newdata = X_val, verbose = FALSE)

write.csv(
  X_val,
  file.path(out, "r_reference_spectra.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(r_prediction = as.vector(y_hat$predictions)),
  file.path(out, "r_reference_predictions.csv"),
  row.names = FALSE
)

cat("Wrote fixtures to", out, "\n")
cat("proximetricsR", as.character(packageVersion("proximetricsR")), "|", R.version.string, "\n")
