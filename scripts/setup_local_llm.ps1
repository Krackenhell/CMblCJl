param(
    [switch]$SkipModelDownload,
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modelDirectory = Join-Path $projectRoot "models"
$toolDirectory = Join-Path $projectRoot "tools\llama"
$vulkanDirectory = Join-Path $projectRoot "tools\llama-vulkan"
$downloadDirectory = Join-Path $projectRoot ".local-llm-download"

$llamaVersion = "b9637"
$llamaArchiveUrl = "https://github.com/ggml-org/llama.cpp/releases/download/b9637/llama-b9637-bin-win-cpu-x64.zip"
$llamaArchiveSha256 = "f7783c2b8c007f95e710ac40f26a24861a80b603b0b739fc54d7c926a4716c1e"
$vulkanArchiveUrl = "https://github.com/ggml-org/llama.cpp/releases/download/b9637/llama-b9637-bin-win-vulkan-x64.zip"
$vulkanArchiveSha256 = "a353945604cffdac3d0d6da6392de78ca565a531a6f2ff3521f44b9b7c6e553f"
$modelCommit = "bb5d59e06d9551d752d08b292a50eb208b07ab1f"
$modelBaseUrl = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/$modelCommit"
$modelPart1Name = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
$modelPart2Name = "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"
$modelPart1Sha256 = "dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db"
$modelPart2Sha256 = "539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a"
$modelSha256 = "1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49"
$modelFile = Join-Path $modelDirectory "qwen2.5-7b-instruct-q4_k_m.gguf"
$fastModelUrl = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/7dabda4d13d513e3e842b20f0d435c732f172cbe/qwen2.5-3b-instruct-q4_k_m.gguf"
$fastModelSha256 = "626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d"
$fastModelFile = Join-Path $modelDirectory "qwen2.5-3b-instruct-q4_k_m.gguf"
$serverFile = Join-Path $toolDirectory "llama-server.exe"
$mergeTool = Join-Path $toolDirectory "llama-gguf-split.exe"

New-Item -ItemType Directory -Force -Path $modelDirectory, $toolDirectory, $vulkanDirectory, $downloadDirectory | Out-Null

function Get-VerifiedFile {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$ExpectedSha256,
        [string]$Label
    )
    if (Test-Path -LiteralPath $Destination) {
        $existingHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Destination).Hash.ToLowerInvariant()
        if ($existingHash -eq $ExpectedSha256) {
            Write-Host "$Label already exists and passed SHA-256 verification."
            return
        }
        $backup = "$Destination.invalid-$(Get-Date -Format 'yyyyMMddHHmmss')"
        Move-Item -LiteralPath $Destination -Destination $backup
        Write-Warning "Existing file failed verification and was moved to $backup"
    }
    Write-Host "Downloading: $Label"
    & curl.exe -L --fail --progress-bar --output $Destination $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to download $Label"
    }
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Destination).Hash.ToLowerInvariant()
    if ($actualHash -ne $ExpectedSha256) {
        throw "SHA-256 mismatch for $Label. Actual: $actualHash"
    }
}

$archive = Join-Path $downloadDirectory "llama-$llamaVersion-win-cpu.zip"
Get-VerifiedFile -Url $llamaArchiveUrl -Destination $archive -ExpectedSha256 $llamaArchiveSha256 -Label "llama.cpp $llamaVersion"

if (-not (Test-Path -LiteralPath $serverFile)) {
    $extractDirectory = Join-Path $downloadDirectory "llama-$llamaVersion"
    New-Item -ItemType Directory -Force -Path $extractDirectory | Out-Null
    Expand-Archive -LiteralPath $archive -DestinationPath $extractDirectory -Force
    $server = Get-ChildItem -Path $extractDirectory -Recurse -Filter "llama-server.exe" | Select-Object -First 1
    if (-not $server) {
        throw "llama-server.exe was not found in the llama.cpp archive"
    }
    Copy-Item -Path (Join-Path $server.Directory.FullName "*") -Destination $toolDirectory -Recurse -Force
}

$vulkanServerFile = Join-Path $vulkanDirectory "llama-server.exe"
if (-not $CpuOnly) {
    $vulkanArchive = Join-Path $downloadDirectory "llama-$llamaVersion-win-vulkan.zip"
    Get-VerifiedFile -Url $vulkanArchiveUrl -Destination $vulkanArchive -ExpectedSha256 $vulkanArchiveSha256 -Label "llama.cpp $llamaVersion Vulkan"
    if (-not (Test-Path -LiteralPath $vulkanServerFile)) {
        $vulkanExtractDirectory = Join-Path $downloadDirectory "llama-$llamaVersion-vulkan"
        New-Item -ItemType Directory -Force -Path $vulkanExtractDirectory | Out-Null
        Expand-Archive -LiteralPath $vulkanArchive -DestinationPath $vulkanExtractDirectory -Force
        $vulkanServer = Get-ChildItem -Path $vulkanExtractDirectory -Recurse -Filter "llama-server.exe" | Select-Object -First 1
        if (-not $vulkanServer) {
            throw "Vulkan llama-server.exe was not found in the archive"
        }
        Copy-Item -Path (Join-Path $vulkanServer.Directory.FullName "*") -Destination $vulkanDirectory -Recurse -Force
    }
}

if (-not $SkipModelDownload) {
    $modelIsValid = (Test-Path -LiteralPath $modelFile) -and `
        ((Get-FileHash -Algorithm SHA256 -LiteralPath $modelFile).Hash.ToLowerInvariant() -eq $modelSha256)
    if (-not $modelIsValid) {
        $part1 = Join-Path $modelDirectory $modelPart1Name
        $part2 = Join-Path $modelDirectory $modelPart2Name
        Get-VerifiedFile -Url "$modelBaseUrl/$modelPart1Name" -Destination $part1 -ExpectedSha256 $modelPart1Sha256 -Label "Qwen2.5-7B Q4_K_M part 1/2"
        Get-VerifiedFile -Url "$modelBaseUrl/$modelPart2Name" -Destination $part2 -ExpectedSha256 $modelPart2Sha256 -Label "Qwen2.5-7B Q4_K_M part 2/2"
        if (-not (Test-Path -LiteralPath $mergeTool)) {
            throw "llama-gguf-split.exe was not found in the llama.cpp archive"
        }
        & $mergeTool --merge $part1 $modelFile
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to merge the Qwen2.5-7B GGUF parts"
        }
        $mergedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $modelFile).Hash.ToLowerInvariant()
        if ($mergedHash -ne $modelSha256) {
            throw "SHA-256 mismatch for the merged Qwen2.5-7B model. Actual: $mergedHash"
        }
    } else {
        Write-Host "Qwen2.5-7B-Instruct Q4_K_M already exists and passed SHA-256 verification."
    }
    Get-VerifiedFile -Url $fastModelUrl -Destination $fastModelFile -ExpectedSha256 $fastModelSha256 -Label "Qwen2.5-3B-Instruct Q4_K_M fast model"
}

if ((Test-Path -LiteralPath $serverFile) -and (Test-Path -LiteralPath $modelFile)) {
    $manifest = [ordered]@{
        model = "Qwen2.5-7B-Instruct-Q4_K_M"
        model_file = (Split-Path -Leaf $modelFile)
        model_sha256 = $modelSha256
        model_source_commit = $modelCommit
        fast_model = "Qwen2.5-3B-Instruct-Q4_K_M"
        fast_model_file = (Split-Path -Leaf $fastModelFile)
        fast_model_sha256 = $fastModelSha256
        runtime = "llama.cpp"
        runtime_version = $llamaVersion
        runtime_sha256 = $llamaArchiveSha256
        vulkan_runtime_sha256 = $vulkanArchiveSha256
        vulkan_available = (Test-Path -LiteralPath $vulkanServerFile)
        installed_at = (Get-Date).ToUniversalTime().ToString("o")
    }
    $manifest | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $modelDirectory "local-model-manifest.json")
    Write-Host "Local LLM installed. No API key or internet connection is required at runtime."
} elseif ($SkipModelDownload) {
    Write-Host "llama.cpp installed; model download was skipped."
}
