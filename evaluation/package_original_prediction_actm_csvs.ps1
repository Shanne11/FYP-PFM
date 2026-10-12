$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$packageRoot = Join-Path $repoRoot 'outputs\original_prediction_actm_csvs'
$zipPath = Join-Path $repoRoot 'outputs\Original_Prediction_ACTM_CSVs_Seeds_42_52_62.zip'
$checksumPath = "$zipPath.sha256.txt"
$seeds = @(42, 52, 62)

$sourcePaths = @(
    'outputs\actm_selection_baselines\actm_selection_contract.json',
    'outputs\actm_selection_baselines\actm_selection_transaction_outcomes.csv',
    'outputs\actm_selection_baselines\actm_selection_runs.csv',
    'outputs\actm_selection_baselines\actm_selection_summary.csv'
)

foreach ($seed in $seeds) {
    foreach ($variant in @('full', 'without_heuristic_adjustment')) {
        $base = "outputs\ablations\seed_$seed\$variant"
        $sourcePaths += @(
            "$base\predictions.csv",
            "$base\test_record_ids.csv",
            "$base\overall_metrics.csv",
            "$base\prompt_metrics.csv"
        )
    }
}

if (Test-Path -LiteralPath $packageRoot) {
    $resolvedPackage = (Resolve-Path -LiteralPath $packageRoot).Path
    $resolvedOutputs = (Resolve-Path -LiteralPath (Join-Path $repoRoot 'outputs')).Path
    if (-not $resolvedPackage.StartsWith($resolvedOutputs, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to replace package outside the repository outputs directory: $resolvedPackage"
    }
    Remove-Item -LiteralPath $packageRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $packageRoot | Out-Null

$manifestFiles = @()
foreach ($relativePath in $sourcePaths) {
    $source = Join-Path $repoRoot $relativePath
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Missing original output: $source"
    }
    $destination = Join-Path $packageRoot $relativePath
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination

    $sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    $copyHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($sourceHash -ne $copyHash) {
        throw "Copied file hash mismatch: $relativePath"
    }

    $entry = [ordered]@{
        packaged_path = $relativePath.Replace('\', '/')
        original_path = $relativePath.Replace('\', '/')
        sha256 = $copyHash
        bytes = (Get-Item -LiteralPath $destination).Length
    }
    if ([IO.Path]::GetExtension($destination) -eq '.csv') {
        $entry.data_rows = @(Import-Csv -LiteralPath $destination).Count
    }
    $manifestFiles += [pscustomobject]$entry
}

$matchedValidation = [ordered]@{}
foreach ($seed in $seeds) {
    $fullPath = Join-Path $packageRoot "outputs\ablations\seed_$seed\full\predictions.csv"
    $ablationPath = Join-Path $packageRoot "outputs\ablations\seed_$seed\without_heuristic_adjustment\predictions.csv"
    $full = @(Import-Csv -LiteralPath $fullPath)
    $ablation = @(Import-Csv -LiteralPath $ablationPath)
    $fullIds = @($full.transaction_id)
    $ablationIds = @($ablation.transaction_id)
    $idsMatch = $fullIds.Count -eq $ablationIds.Count
    if ($idsMatch) {
        for ($index = 0; $index -lt $fullIds.Count; $index++) {
            if ($fullIds[$index] -ne $ablationIds[$index]) {
                $idsMatch = $false
                break
            }
        }
    }
    $uniqueIds = @($fullIds | Sort-Object -Unique).Count
    if ($full.Count -ne 2308 -or $ablation.Count -ne 2308 -or -not $idsMatch) {
        throw "Matched prediction verification failed for seed $seed"
    }
    $matchedValidation["$seed"] = [ordered]@{
        full_rows = $full.Count
        without_heuristic_adjustment_rows = $ablation.Count
        transaction_ids_identical_and_ordered = $idsMatch
        unique_transaction_ids = $uniqueIds
    }
}

$actmPath = Join-Path $packageRoot 'outputs\actm_selection_baselines\actm_selection_transaction_outcomes.csv'
$actm = @(Import-Csv -LiteralPath $actmPath)
$actmSeeds = @($actm.seed | ForEach-Object { [int]$_ } | Sort-Object -Unique)
$strategies = @($actm.strategy | Sort-Object -Unique)
$expectedActmRows = 2308 * 3 * 5
if ($actm.Count -ne $expectedActmRows) {
    throw "ACTM transaction outcome row count is incomplete"
}

$manifest = [ordered]@{
    purpose = 'Original transaction-level ACTM outcomes and seed 42/52/62 matched-ablation prediction CSVs for independent reproduction.'
    copy_policy = 'Files under outputs/ are byte-for-byte copies of original experiment outputs; hashes were compared after copying.'
    actm_validation = [ordered]@{
        rows = $actm.Count
        expected_rows = $expectedActmRows
        seeds = $actmSeeds
        strategies = $strategies
    }
    matched_prediction_validation = $matchedValidation
    files = $manifestFiles
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $packageRoot 'MANIFEST.json') -Encoding utf8

$readme = @'
# Original prediction and ACTM CSV package

This package contains byte-for-byte copies of the original experimental outputs used for the Section 16 evidence audit.

## ACTM selection evidence

- `actm_selection_transaction_outcomes.csv`: transaction-level outcomes for seeds 42, 52 and 62 across Random, Entropy-only, Margin-only, Conflict-only and Proposed ACTM.
- `actm_selection_runs.csv`: one result row per seed and strategy.
- `actm_selection_summary.csv`: saved three-seed summary.
- `actm_selection_contract.json`: selection and evaluation contract.

The transaction file has 34,620 rows: 2,308 held-out records x 5 strategies x 3 seeds.

## Matched-ablation predictions

For each seed, `full/predictions.csv` is paired with `without_heuristic_adjustment/predictions.csv`. Each file contains 2,308 rows, and the ordered transaction identifiers match within each seed. Test record IDs, overall metrics and prompt metrics are included beside the predictions.

## Integrity

`MANIFEST.json` records each original path, row count, byte size and SHA-256 hash. The packaging script verifies that copied hashes match the original files before creating the archive.
'@
$readme | Set-Content -LiteralPath (Join-Path $packageRoot 'README.md') -Encoding utf8

if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -Path (Join-Path $packageRoot '*') -DestinationPath $zipPath -CompressionLevel Optimal
$zipHash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
"$zipHash  $([IO.Path]::GetFileName($zipPath))" | Set-Content -LiteralPath $checksumPath -Encoding ascii

[pscustomobject]@{
    PackageDirectory = $packageRoot
    Zip = $zipPath
    ZipSHA256 = $zipHash
    OriginalFiles = $sourcePaths.Count
    ACTMRows = $actm.Count
    Strategies = $strategies -join '; '
    Seed42Matched = $matchedValidation['42'].transaction_ids_identical_and_ordered
    Seed52Matched = $matchedValidation['52'].transaction_ids_identical_and_ordered
    Seed62Matched = $matchedValidation['62'].transaction_ids_identical_and_ordered
} | Format-List
