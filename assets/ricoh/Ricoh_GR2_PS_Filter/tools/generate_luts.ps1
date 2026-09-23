# Generates Photoshop-compatible .cube LUTs for the Ricoh GR2 look pack.
# Run: powershell -ExecutionPolicy Bypass -File .\generate_luts.ps1

$ErrorActionPreference = "Stop"
$outputDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\LUT"))
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
}

$profiles = @(
    @{
        File = "GR2_Positive_Film.cube"
        Title = "Ricoh GR2 Positive Film"
        Curve = @(@(0,3), @(34,28), @(78,72), @(128,132), @(184,195), @(230,229), @(255,248))
        Brightness = -1.0
        Contrast = 9.0
        Saturation = -6.0
        Shadows = @(-5,0,4)
        Midtones = @(2,-1,-2)
        Highlights = @(5,-1,-5)
    },
    @{
        File = "GR2_Negative_Film.cube"
        Title = "Ricoh GR2 Negative Film"
        Curve = @(@(0,17), @(34,42), @(92,96), @(145,150), @(198,198), @(235,225), @(255,238))
        Brightness = 1.0
        Contrast = -4.0
        Saturation = -11.0
        Shadows = @(-7,3,6)
        Midtones = @(5,2,-4)
        Highlights = @(3,1,-6)
    },
    @{
        File = "GR2_Hi_BW.cube"
        Title = "Ricoh GR2 High Contrast BW"
        Curve = @(@(0,0), @(28,15), @(72,61), @(126,126), @(178,192), @(226,229), @(255,255))
        Brightness = 0.0
        Contrast = 18.0
        Saturation = -100.0
        Shadows = @(0,0,0)
        Midtones = @(0,0,0)
        Highlights = @(0,0,0)
    },
    @{
        File = "GR2_Street_Positive.cube"
        Title = "Ricoh GR2 Street Positive"
        Curve = @(@(0,0), @(30,20), @(76,69), @(128,134), @(183,199), @(229,230), @(255,251))
        Brightness = -2.0
        Contrast = 15.0
        Saturation = -3.0
        Shadows = @(-6,1,5)
        Midtones = @(3,-2,-3)
        Highlights = @(6,-1,-6)
    }
)

function Clamp01([double]$value) {
    if ($value -lt 0.0) { return 0.0 }
    if ($value -gt 1.0) { return 1.0 }
    return $value
}

function CurveValue([double]$value, [object[]]$points) {
    $x = $value * 255.0
    for ($i = 0; $i -lt ($points.Count - 1); $i++) {
        $p0x = [double]$points[$i][0]
        $p0y = [double]$points[$i][1]
        $p1x = [double]$points[$i + 1][0]
        $p1y = [double]$points[$i + 1][1]
        if ($x -le $p1x) {
            $t = ($x - $p0x) / ($p1x - $p0x)
            $y = $p0y + $t * ($p1y - $p0y)
            return (Clamp01 ($y / 255.0))
        }
    }
    return (Clamp01 ([double]$points[$points.Count - 1][1] / 255.0))
}

function ColorBalanceBias([double]$value, [int]$shadow, [int]$midtone, [int]$highlight) {
    $shadowWeight = [Math]::Pow(1.0 - $value, 2.0)
    $highlightWeight = [Math]::Pow($value, 2.0)
    $midtoneWeight = 1.0 - $shadowWeight - $highlightWeight
    $weighted = $shadow * $shadowWeight + $midtone * $midtoneWeight + $highlight * $highlightWeight
    return (($weighted / 100.0) * 0.18)
}

function GradePixel([double]$r, [double]$g, [double]$b, [hashtable]$profile) {
    $r = CurveValue $r $profile.Curve
    $g = CurveValue $g $profile.Curve
    $b = CurveValue $b $profile.Curve

    $brightness = $profile.Brightness / 100.0
    $contrastFactor = 1.0 + ($profile.Contrast / 100.0)
    $r = ($r + $brightness - 0.5) * $contrastFactor + 0.5
    $g = ($g + $brightness - 0.5) * $contrastFactor + 0.5
    $b = ($b + $brightness - 0.5) * $contrastFactor + 0.5

    $r += ColorBalanceBias $r $profile.Shadows[0] $profile.Midtones[0] $profile.Highlights[0]
    $g += ColorBalanceBias $g $profile.Shadows[1] $profile.Midtones[1] $profile.Highlights[1]
    $b += ColorBalanceBias $b $profile.Shadows[2] $profile.Midtones[2] $profile.Highlights[2]

    $luma = 0.2126 * $r + 0.7152 * $g + 0.0722 * $b
    $saturationFactor = 1.0 + ($profile.Saturation / 100.0)
    $r = $luma + ($r - $luma) * $saturationFactor
    $g = $luma + ($g - $luma) * $saturationFactor
    $b = $luma + ($b - $luma) * $saturationFactor

    return @((Clamp01 $r), (Clamp01 $g), (Clamp01 $b))
}

$size = 33
$lastIndex = $size - 1
$culture = [Globalization.CultureInfo]::InvariantCulture

foreach ($profile in $profiles) {
    $builder = New-Object System.Text.StringBuilder
    [void]$builder.AppendLine("TITLE `"$($profile.Title)`"")
    [void]$builder.AppendLine("LUT_3D_SIZE $size")
    [void]$builder.AppendLine("DOMAIN_MIN 0.000000 0.000000 0.000000")
    [void]$builder.AppendLine("DOMAIN_MAX 1.000000 1.000000 1.000000")
    [void]$builder.AppendLine("")

    for ($blueIndex = 0; $blueIndex -lt $size; $blueIndex++) {
        $blue = $blueIndex / $lastIndex
        for ($greenIndex = 0; $greenIndex -lt $size; $greenIndex++) {
            $green = $greenIndex / $lastIndex
            for ($redIndex = 0; $redIndex -lt $size; $redIndex++) {
                $red = $redIndex / $lastIndex
                $pixel = GradePixel $red $green $blue $profile
                $line = [string]::Format($culture, "{0:F6} {1:F6} {2:F6}", $pixel[0], $pixel[1], $pixel[2])
                [void]$builder.AppendLine($line)
            }
        }
    }

    $target = Join-Path $outputDirectory $profile.File
    [IO.File]::WriteAllText($target, $builder.ToString(), (New-Object System.Text.UTF8Encoding($false)))
    Write-Output "Generated $($profile.File)"
}