$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$downloadDir = Join-Path $projectRoot ".local-voice-download"
$runtimeDir = Join-Path $projectRoot "tools\whisper"
$jreDir = Join-Path $projectRoot "tools\jre"
$languageToolDir = Join-Path $projectRoot "tools\languagetool"
$modelDir = Join-Path $projectRoot "models"
$archivePath = Join-Path $downloadDir "whisper-bin-x64-v1.9.1.zip"
$jreArchivePath = Join-Path $downloadDir "temurin-jre17-windows-x64.zip"
$languageToolArchivePath = Join-Path $downloadDir "LanguageTool-6.6.zip"
$modelPath = Join-Path $modelDir "ggml-base.en.bin"
$vadModelPath = Join-Path $modelDir "ggml-silero-v6.2.0.bin"
$manifestPath = Join-Path $modelDir "voice-model-manifest.json"
$expectedHashes = @{
    $archivePath = "7d8be46ecd31828e1eb7a2ecdd0d6b314feafd82163038ab6092594b0a063539"
    $modelPath = "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002"
    $vadModelPath = "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987"
    $jreArchivePath = "79a598e1fbb4e16582d92c4ee22280a3c4d72fd52606e1e46b1223c0fe53b0da"
    $languageToolArchivePath = "53600506b399bb5ffe1e4c8dec794fd378212f14aaf38ccef9b6f89314d11631"
}

function Assert-ExpectedHash([string]$Path) {
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expectedHashes[$Path]) {
        throw "Контрольная сумма не совпала: $Path"
    }
}

New-Item -ItemType Directory -Force -Path `
    $downloadDir, $runtimeDir, $jreDir, $languageToolDir, $modelDir | Out-Null

if (-not (Test-Path -LiteralPath $archivePath)) {
    Invoke-WebRequest -UseBasicParsing `
        "https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.1/whisper-bin-x64.zip" `
        -OutFile $archivePath
}
Expand-Archive -LiteralPath $archivePath -DestinationPath $runtimeDir -Force

if (-not (Test-Path -LiteralPath $modelPath)) {
    Invoke-WebRequest -UseBasicParsing `
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin" `
        -OutFile $modelPath
}
if (-not (Test-Path -LiteralPath $vadModelPath)) {
    Invoke-WebRequest -UseBasicParsing `
        "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin" `
        -OutFile $vadModelPath
}

if (-not (Test-Path -LiteralPath $jreArchivePath)) {
    Invoke-WebRequest -UseBasicParsing `
        "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.19%2B10/OpenJDK17U-jre_x64_windows_hotspot_17.0.19_10.zip" `
        -OutFile $jreArchivePath
}
if (-not (Get-ChildItem -LiteralPath $jreDir -Recurse -Filter java.exe -ErrorAction SilentlyContinue)) {
    Expand-Archive -LiteralPath $jreArchivePath -DestinationPath $jreDir -Force
}

if (-not (Test-Path -LiteralPath $languageToolArchivePath)) {
    Invoke-WebRequest -UseBasicParsing `
        "https://languagetool.org/download/LanguageTool-6.6.zip" `
        -OutFile $languageToolArchivePath
}
if (-not (Get-ChildItem -LiteralPath $languageToolDir -Recurse `
        -Filter languagetool-commandline.jar -ErrorAction SilentlyContinue)) {
    Expand-Archive -LiteralPath $languageToolArchivePath -DestinationPath $languageToolDir -Force
}

$runtimeExe = Join-Path $runtimeDir "Release\whisper-cli.exe"
if (-not (Test-Path -LiteralPath $runtimeExe)) {
    throw "whisper-cli.exe не найден после распаковки runtime."
}
$javaExe = Get-ChildItem -LiteralPath $jreDir -Recurse -Filter java.exe | `
    Select-Object -First 1 -ExpandProperty FullName
$languageToolJar = Get-ChildItem -LiteralPath $languageToolDir -Recurse `
    -Filter languagetool-commandline.jar | Select-Object -First 1 -ExpandProperty FullName
if (-not $javaExe -or -not $languageToolJar) {
    throw "Локальный LanguageTool или Java runtime не найден после распаковки."
}

foreach ($file in $expectedHashes.Keys) {
    Assert-ExpectedHash $file
}

$manifest = [ordered]@{
    asr_model = "Whisper base.en"
    asr_model_file = "ggml-base.en.bin"
    asr_model_sha256 = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    vad_model = "Silero VAD 6.2 for whisper.cpp"
    vad_model_file = "ggml-silero-v6.2.0.bin"
    vad_model_sha256 = (Get-FileHash -LiteralPath $vadModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    runtime = "whisper.cpp"
    runtime_version = "v1.9.1"
    runtime_archive_sha256 = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    grammar_checker = "LanguageTool 6.6 offline"
    grammar_archive_sha256 = (Get-FileHash -LiteralPath $languageToolArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    java_runtime = "Eclipse Temurin JRE 17"
    java_archive_sha256 = (Get-FileHash -LiteralPath $jreArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    tts = "Windows SAPI offline"
    installed_at = [DateTime]::UtcNow.ToString("o")
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host "Локальный голосовой runtime готов."
Write-Host "ASR: $modelPath"
Write-Host "VAD: $vadModelPath"
Write-Host "Runtime: $runtimeExe"
Write-Host "Grammar: $languageToolJar"
