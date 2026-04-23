# Tests for SPAGMMATtest function (Step 2)
# Based on: https://weizhou0.github.io/SAIGE-QTL-doc/docs/single_step2.html

test_that("SPAGMMATtest exists and is a function", {
    expect_true(exists("SPAGMMATtest"))
    expect_true(is.function(SPAGMMATtest))
})
test_that("SPAGMMATtest has expected parameters", {
    fn_args <- names(formals(SPAGMMATtest))

    # Check key parameters for PLINK input
    expect_true("bedFile" %in% fn_args)
    expect_true("bimFile" %in% fn_args)
    expect_true("famFile" %in% fn_args)

    # Check key parameters for BGEN input
    expect_true("bgenFile" %in% fn_args)

    # Check key parameters for VCF input
    expect_true("vcfFile" %in% fn_args)
    expect_true("vcfField" %in% fn_args)

    # Check key output and model parameters
    expect_true("SAIGEOutputFile" %in% fn_args)
    expect_true("GMMATmodelFile" %in% fn_args)
    expect_true("varianceRatioFile" %in% fn_args)

    # Check filtering parameters
    expect_true("minMAF" %in% fn_args)
    expect_true("minMAC" %in% fn_args)
    expect_true("chrom" %in% fn_args)

    # Check analysis options
    expect_true("LOCO" %in% fn_args)
    expect_true("SPAcutoff" %in% fn_args)
})

test_that("SPAGMMATtest has correct default values", {
    defaults <- formals(SPAGMMATtest)

    expect_equal(defaults$minMAF, 0)
    expect_equal(defaults$minMAC, 0.5)
    expect_equal(defaults$LOCO, FALSE)
    expect_equal(defaults$vcfField, "DS")
    expect_equal(defaults$SPAcutoff, 2)
})

test_that("SPAGMMATtest errors on missing model file", {
    skip_on_cran()

    expect_error(
        SPAGMMATtest(
            GMMATmodelFile = "nonexistent_model.rda",
            SAIGEOutputFile = tempfile()
        )
    )
})
