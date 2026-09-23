# Creates a synthetic visual comparison sheet for the GR3 looks.
# Run: powershell -ExecutionPolicy Bypass -File .\generate_preview.ps1

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
Add-Type -TypeDefinition ([IO.File]::ReadAllText((Join-Path $PSScriptRoot "GR3Preview.cs"), [Text.Encoding]::UTF8)) -ReferencedAssemblies System.Drawing

$packageRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$lutDirectory = Join-Path $packageRoot "LUT"
$previewDirectory = Join-Path $packageRoot "Preview"
if (-not (Test-Path -LiteralPath $previewDirectory)) {
    New-Item -ItemType Directory -Path $previewDirectory -Force | Out-Null
}

$outputPath = Join-Path $previewDirectory "GR3_Look_Comparison.png"
[RicohGr3Preview]::Generate($lutDirectory, $outputPath)
Write-Output "Generated preview: $outputPath"