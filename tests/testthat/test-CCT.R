# Tests for CCT (Cauchy Combination Test) function
# Based on: https://weizhou0.github.io/SAIGE-QTL-doc/docs/gene_step3.html

test_that("CCT returns correct p-value for simple input", {
    # Example from documentation
    pvalues <- c(2e-02, 4e-04, 0.2, 0.1, 0.8)
    result <- CCT(pvals = pvalues)

    expect_true(is.numeric(result))
    expect_true(result >= 0 && result <= 1)
})

test_that("CCT handles equal weights by default", {
    pvalues <- c(0.01, 0.05, 0.1)
    result <- CCT(pvals = pvalues)

    expect_true(is.numeric(result))
    expect_length(result, 1)
})

test_that("CCT handles custom weights", {
    pvalues <- c(0.01, 0.05, 0.1)
    weights <- c(0.5, 0.3, 0.2)
    result <- CCT(pvals = pvalues, weights = weights)

    expect_true(is.numeric(result))
    expect_true(result >= 0 && result <= 1)
})

test_that("CCT returns 0 when any p-value is 0", {
    pvalues <- c(0, 0.05, 0.1)
    result <- CCT(pvals = pvalues)

    expect_equal(result, 0)
})

test_that("CCT handles p-values equal to 1", {
    pvalues <- c(1, 0.05, 0.1)
    result <- CCT(pvals = pvalues)

    expect_true(is.numeric(result))
    expect_true(result >= 0 && result <= 1)
})

test_that("CCT errors on NA values", {
    pvalues <- c(NA, 0.05, 0.1)

    expect_error(CCT(pvals = pvalues), "Cannot have NAs")
})

test_that("CCT errors on invalid p-values", {
    # p-value > 1
    expect_error(CCT(pvals = c(1.5, 0.05)), "between 0 and 1")

    # p-value < 0
    expect_error(CCT(pvals = c(-0.1, 0.05)), "between 0 and 1")
})

test_that("CCT errors on mismatched weights length", {
    pvalues <- c(0.01, 0.05, 0.1)
    weights <- c(0.5, 0.5)  # wrong length

    expect_error(CCT(pvals = pvalues, weights = weights), "same as that of the p-values")
})

test_that("CCT errors on negative weights", {
    pvalues <- c(0.01, 0.05, 0.1)
    weights <- c(-0.5, 0.3, 0.2)

    expect_error(CCT(pvals = pvalues, weights = weights), "positive")
})

test_that("CCT handles very small p-values correctly", {
    pvalues <- c(1e-20, 1e-18, 0.1)
    result <- CCT(pvals = pvalues)

    expect_true(is.numeric(result))
    expect_true(result >= 0 && result <= 1)
    expect_true(result < 0.01)  # Should be very significant
})

test_that("CCT handles single p-value", {
    result <- CCT(pvals = c(0.05))

    expect_true(is.numeric(result))
    expect_equal(result, 0.05, tolerance = 1e-10)
})
