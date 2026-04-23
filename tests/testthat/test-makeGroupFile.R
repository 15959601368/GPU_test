# Tests for makeGroupFileforRegions function

test_that("makeGroupFileforRegions exists and is a function", {
    expect_true(exists("makeGroupFileforRegions"))
    expect_true(is.function(makeGroupFileforRegions))
})

test_that("makeGroupFileforRegions has expected parameters", {
    fn_args <- names(formals(makeGroupFileforRegions))

    # Check genotype input parameters
    expect_true("bgenFile" %in% fn_args)
    expect_true("vcfFile" %in% fn_args)
    expect_true("bedFile" %in% fn_args)
    expect_true("bimFile" %in% fn_args)
    expect_true("famFile" %in% fn_args)

    # Check region and output parameters
    expect_true("regionFile" %in% fn_args)
    expect_true("outputPrefix" %in% fn_args)

    # Check allele order parameter
    expect_true("AlleleOrder" %in% fn_args)
})

test_that("makeGroupFileforRegions has correct default values", {
    defaults <- formals(makeGroupFileforRegions)

    expect_equal(defaults$AlleleOrder, "alt-first")
    expect_equal(defaults$vcfField, "DS")
})

test_that("makeGroupFileforRegions errors on missing region file", {
    skip_on_cran()

    expect_error(
        makeGroupFileforRegions(
            regionFile = "nonexistent_region.txt",
            outputPrefix = tempfile()
        )
    )
})
