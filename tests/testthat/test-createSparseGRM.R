# Tests for createSparseGRM function

test_that("createSparseGRM exists and is a function", {
    expect_true(exists("createSparseGRM"))
    expect_true(is.function(createSparseGRM))
})

test_that("createSparseGRM has expected parameters", {
    fn_args <- names(formals(createSparseGRM))

    # Check key parameters
    expect_true("bedFile" %in% fn_args)
    expect_true("bimFile" %in% fn_args)
    expect_true("famFile" %in% fn_args)
    expect_true("outputPrefix" %in% fn_args)
    expect_true("nThreads" %in% fn_args)
    expect_true("relatednessCutoff" %in% fn_args)
    expect_true("numRandomMarkerforSparseKin" %in% fn_args)
})

test_that("createSparseGRM has correct default values", {
    defaults <- formals(createSparseGRM)

    expect_equal(defaults$nThreads, 1)
    expect_equal(defaults$relatednessCutoff, 0.125)
    expect_equal(defaults$numRandomMarkerforSparseKin, 2000)
})
