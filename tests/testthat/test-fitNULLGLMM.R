# Tests for fitNULLGLMM_multiV function (Step 1)
# Based on: https://weizhou0.github.io/SAIGE-QTL-doc/docs/step1.html

test_that("fitNULLGLMM_multiV exists and is a function", {
    expect_true(exists("fitNULLGLMM_multiV"))
    expect_true(is.function(fitNULLGLMM_multiV))
})

test_that("fitNULLGLMM_multiV has expected parameters", {
    fn_args <- names(formals(fitNULLGLMM_multiV))

    # Check key parameters from documentation
    expect_true("phenoFile" %in% fn_args)
    expect_true("phenoCol" %in% fn_args)
    expect_true("covarColList" %in% fn_args)
    expect_true("sampleIDColinphenoFile" %in% fn_args)
    expect_true("traitType" %in% fn_args)
    expect_true("outputPrefix" %in% fn_args)
    expect_true("plinkFile" %in% fn_args)
    expect_true("tol" %in% fn_args)
    expect_true("isCovariateTransform" %in% fn_args)
    expect_true("skipVarianceRatioEstimation" %in% fn_args)
    expect_true("useSparseGRMtoFitNULL" %in% fn_args)
    expect_true("offsetCol" %in% fn_args)
})

test_that("fitNULLGLMM_multiV has correct default values", {
    defaults <- formals(fitNULLGLMM_multiV)

    expect_equal(defaults$traitType, "binary")
    expect_equal(defaults$tol, 0.02)
    expect_equal(defaults$maxiter, 20)
    expect_equal(defaults$nThreads, 1)
    expect_equal(defaults$isCovariateTransform, TRUE)
})

test_that("fitNULLGLMM_multiV errors on missing phenoFile", {
    skip_on_cran()  # Skip expensive tests on CRAN

    expect_error(
        fitNULLGLMM_multiV(
            phenoFile = "nonexistent_file.txt",
            phenoCol = "gene_1",
            outputPrefix = tempfile()
        )
    )
})

test_that("fitNULLGLMM_multiV errors on missing output directory", {
    skip_on_cran()

    expect_error(
        fitNULLGLMM_multiV(
            phenoFile = "",
            phenoCol = "gene_1",
            outputPrefix = "/nonexistent/path/output"
        )
    )
})
