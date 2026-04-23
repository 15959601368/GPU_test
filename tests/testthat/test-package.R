# Basic package loading and structure tests

test_that("SAIGEQTL package loads correctly", {
    expect_true(requireNamespace("SAIGEQTL", quietly = TRUE))
})

test_that("Package has expected exported functions", {
    # Main workflow functions
    expect_true(exists("fitNULLGLMM_multiV"))
    expect_true(exists("SPAGMMATtest"))
    expect_true(exists("CCT"))
    expect_true(exists("createSparseGRM"))
    expect_true(exists("makeGroupFileforRegions"))
})

test_that("Package dependencies are available", {
    expect_true(requireNamespace("Rcpp", quietly = TRUE))
    expect_true(requireNamespace("Matrix", quietly = TRUE))
    expect_true(requireNamespace("data.table", quietly = TRUE))
})
