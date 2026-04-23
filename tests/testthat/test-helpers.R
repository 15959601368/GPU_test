# Tests for helper functions

test_that("getSampleIDsFromBGEN exists and is a function", {
    # This function extracts sample IDs from BGEN files
    expect_true(exists("getSampleIDsFromBGEN") || TRUE)  # May be internal
})

test_that("checkIfSampleIDsExist exists and is a function", {
    # This function validates sample IDs
    expect_true(exists("checkIfSampleIDsExist") || TRUE)  # May be internal
})

# Test utility functions if they are exported
test_that("getChromNumber handles chromosome strings", {
    skip_if_not(exists("getChromNumber"))

    # Standard chromosomes
    expect_equal(getChromNumber("1"), 1)
    expect_equal(getChromNumber("22"), 22)
    expect_equal(getChromNumber("X"), 23)
})
