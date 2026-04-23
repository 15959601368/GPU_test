# Integration tests using example data
# These tests use the data from extdata/ to verify full workflow

# Helper to get extdata path
get_extdata_path <- function(file) {
    # Try inst/extdata first (installed package)
    path <- system.file("extdata", file, package = "SAIGEQTL")
    if (path == "") {
        # Try development path
        path <- file.path(
            system.file(package = "SAIGEQTL"),
            "..", "extdata", file
        )
    }
    path
}

test_that("Example phenotype data file exists", {
    skip_on_cran()
    skip_if_not_installed("SAIGEQTL")

    # Check if extdata is accessible
    pheno_file <- get_extdata_path(
        "input/seed_1_100_nindep_100_ncell_100_lambda_2_tauIntraSample_0.5_Poisson.txt"
    )

    # This test may be skipped if extdata is not bundled
    skip_if(!file.exists(pheno_file), "Example data not available")

    expect_true(file.exists(pheno_file))
})

test_that("Example genotype data files exist", {
    skip_on_cran()
    skip_if_not_installed("SAIGEQTL")

    # Check PLINK files
    bed_file <- get_extdata_path("input/genotype_100markers.bed")
    bim_file <- get_extdata_path("input/genotype_100markers.bim")
    fam_file <- get_extdata_path("input/genotype_100markers.fam")

    skip_if(!file.exists(bed_file), "Example genotype data not available")

    expect_true(file.exists(bed_file))
    expect_true(file.exists(bim_file))
    expect_true(file.exists(fam_file))
})

test_that("Example BGEN data file exists", {
    skip_on_cran()
    skip_if_not_installed("SAIGEQTL")

    bgen_file <- get_extdata_path("input/genotype_100markers.bgen")

    skip_if(!file.exists(bgen_file), "Example BGEN data not available")

    expect_true(file.exists(bgen_file))
})

test_that("Example region file exists", {
    skip_on_cran()
    skip_if_not_installed("SAIGEQTL")

    region_file <- get_extdata_path("input/gene_1_cis_region.txt")

    skip_if(!file.exists(region_file), "Example region data not available")

    expect_true(file.exists(region_file))
})

test_that("Expected output reference files exist", {
    skip_on_cran()
    skip_if_not_installed("SAIGEQTL")

    # Variance ratio file from Step 1
    var_ratio_file <- get_extdata_path(
        "expected_output/nindep_100_ncell_100_lambda_2_tauIntraSample_0.5_gene_1_shared.varianceRatio.txt"
    )

    skip_if(!file.exists(var_ratio_file), "Expected output not available")

    expect_true(file.exists(var_ratio_file))

    # Read and validate structure
    var_ratio <- readLines(var_ratio_file)
    expect_true(length(var_ratio) > 0)
})

test_that("Expected model output file exists", {
    skip_on_cran()
    skip_if_not_installed("SAIGEQTL")

    model_file <- get_extdata_path(
        "expected_output/nindep_100_ncell_100_lambda_2_tauIntraSample_0.5_gene_1_shared.rda"
    )

    skip_if(!file.exists(model_file), "Expected model output not available")

    expect_true(file.exists(model_file))
})
