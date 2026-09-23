# Installs the GR2 Camera Raw presets for the current Windows user.
# Run: powershell -ExecutionPolicy Bypass -File .\Install_GR2_CameraRaw_Presets.ps1

$ErrorActionPreference = "Stop"
$sourceDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "Presets"))
$destinationDirectory = [IO.Path]::GetFullPath((Join-Path $env:APPDATA "Adobe\CameraRaw\Settings"))

if (-not (Test-Path -LiteralPath $sourceDirectory)) {
    throw "Preset source folder not found: $sourceDirectory"
}

if (-not (Test-Path -LiteralPath $destinationDirectory)) {
    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
}

$presets = Get-ChildItem -LiteralPath $sourceDirectory -File -Filter "*.xmp"
if ($presets.Count -eq 0) {
    throw "No .xmp presets found in: $sourceDirectory"
}

foreach ($preset in $presets) {
    $target = Join-Path $destinationDirectory $preset.Name
    Copy-Item -LiteralPath $preset.FullName -Destination $target -Force
    Write-Output "Installed: $($preset.Name)"
}

Write-Output ""
Write-Output "Camera Raw preset folder:"
Write-Output $destinationDirectory
Write-Output ""
Write-Output "Restart Photoshop or Camera Raw, then open the Presets panel and find 'GR2 Film Looks'."