# Generates Ricoh GR III-inspired Photoshop LUTs and Camera Raw presets.
# Run: powershell -ExecutionPolicy Bypass -File .\generate_assets.ps1

$ErrorActionPreference = "Stop"

$packageRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$lutDirectory = Join-Path $packageRoot "LUT"
$presetDirectory = Join-Path $packageRoot "CameraRaw_Presets"

foreach ($directory in @($lutDirectory, $presetDirectory)) {
    if (-not (Test-Path -LiteralPath $directory)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
}

$profiles = @(
    @{
        File = "GR3_Standard.cube"
        Title = "Ricoh GR III Standard"
        PresetName = "GR3 标准 Standard"
        Description = "GR III-inspired natural street look with a restrained tonal palette."
        Curve = @(@(0,1), @(32,30), @(92,94), @(156,160), @(220,223), @(255,253))
        Brightness = 0.0
        Contrast = 7.0
        Saturation = -4.0
        Vibrance = 3.0
        Shadows = @(-2,1,2)
        Midtones = @(1,1,-1)
        Highlights = @(3,0,-3)
        ShadowHue = 195.0
        ShadowSat = 8.0
        HighlightHue = 45.0
        HighlightSat = 6.0
        SplitBalance = 0.0
        Grain = 8.0
        Sharpness = 34.0
        Vignette = 7.0
        ConvertToGrayscale = $false
    },
    @{
        File = "GR3_Positive_Film.cube"
        Title = "Ricoh GR III Positive Film"
        PresetName = "GR3 正片 Positive Film"
        Description = "Teal shadows, warm highlights, deeper blues and restrained greens."
        Curve = @(@(0,2), @(30,24), @(72,66), @(128,133), @(190,199), @(232,231), @(255,248))
        Brightness = -2.0
        Contrast = 12.0
        Saturation = -7.0
        Vibrance = 2.0
        Shadows = @(-7,1,6)
        Midtones = @(4,-1,-5)
        Highlights = @(8,0,-7)
        ShadowHue = 190.0
        ShadowSat = 18.0
        HighlightHue = 43.0
        HighlightSat = 14.0
        SplitBalance = -5.0
        Grain = 12.0
        Sharpness = 48.0
        Vignette = 10.0
        ConvertToGrayscale = $false
    },
    @{
        File = "GR3_Negative_Film.cube"
        Title = "Ricoh GR III Negative Film"
        PresetName = "GR3 负片 Negative Film"
        Description = "Lifted blacks, gentle contrast, muted color and a soft highlight roll-off."
        Curve = @(@(0,16), @(32,38), @(84,88), @(132,139), @(186,185), @(230,220), @(255,241))
        Brightness = 0.0
        Contrast = -5.0
        Saturation = -12.0
        Vibrance = -3.0
        Shadows = @(-4,2,5)
        Midtones = @(5,0,-5)
        Highlights = @(4,1,-6)
        ShadowHue = 205.0
        ShadowSat = 13.0
        HighlightHue = 45.0
        HighlightSat = 8.0
        SplitBalance = -2.0
        Grain = 15.0
        Sharpness = 28.0
        Vignette = 8.0
        ConvertToGrayscale = $false
    },
    @{
        File = "GR3_Vivid_Street.cube"
        Title = "Ricoh GR III Vivid Street"
        PresetName = "GR3 鲜明街头 Vivid Street"
        Description = "Hard contrast, stronger blue, crisp street detail and controlled skin tones."
        Curve = @(@(0,0), @(24,15), @(72,70), @(128,137), @(192,204), @(236,237), @(255,250))
        Brightness = -3.0
        Contrast = 18.0
        Saturation = 6.0
        Vibrance = 8.0
        Shadows = @(-6,1,4)
        Midtones = @(4,-2,-2)
        Highlights = @(5,-1,-8)
        ShadowHue = 210.0
        ShadowSat = 16.0
        HighlightHue = 42.0
        HighlightSat = 10.0
        SplitBalance = -4.0
        Grain = 11.0
        Sharpness = 56.0
        Vignette = 14.0
        ConvertToGrayscale = $false
    },
    @{
        File = "GR3_Bleach_Bypass.cube"
        Title = "Ricoh GR III Bleach Bypass"
        PresetName = "GR3 漂白负冲 Bleach Bypass"
        Description = "High contrast, low saturation and metallic gray tonality."
        Curve = @(@(0,0), @(20,8), @(60,48), @(124,131), @(188,204), @(232,238), @(255,255))
        Brightness = -2.0
        Contrast = 23.0
        Saturation = -40.0
        Vibrance = -20.0
        Shadows = @(-2,1,3)
        Midtones = @(1,0,-1)
        Highlights = @(2,-1,-2)
        ShadowHue = 205.0
        ShadowSat = 7.0
        HighlightHue = 45.0
        HighlightSat = 5.0
        SplitBalance = 0.0
        Grain = 14.0
        Sharpness = 45.0
        Vignette = 12.0
        ConvertToGrayscale = $false
    },
    @{
        File = "GR3_High_Contrast_BW.cube"
        Title = "Ricoh GR III High Contrast B&W"
        PresetName = "GR3 高反差黑白 High Contrast B&W"
        Description = "Deep blacks, bright whites and visible monochrome grain."
        Curve = @(@(0,0), @(18,5), @(58,44), @(126,129), @(182,202), @(226,235), @(255,255))
        Brightness = -1.0
        Contrast = 22.0
        Saturation = -100.0
        Vibrance = 0.0
        Shadows = @(0,0,0)
        Midtones = @(0,0,0)
        Highlights = @(0,0,0)
        ShadowHue = 0.0
        ShadowSat = 0.0
        HighlightHue = 0.0
        HighlightSat = 0.0
        SplitBalance = 0.0
        Grain = 20.0
        Sharpness = 62.0
        Vignette = 15.0
        ConvertToGrayscale = $true
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

function XmlEscape([string]$value) {
    if ($null -eq $value) { return "" }
    return $value.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace('"', "&quot;")
}

function Get-DeterministicId([string]$value) {
    $md5 = [Security.Cryptography.MD5]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($value)
        $hash = $md5.ComputeHash($bytes)
        return (($hash | ForEach-Object { $_.ToString("X2") }) -join "")
    } finally {
        $md5.Dispose()
    }
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

    $lutTarget = Join-Path $lutDirectory $profile.File
    [IO.File]::WriteAllText($lutTarget, $builder.ToString(), (New-Object System.Text.UTF8Encoding($false)))
    Write-Output "Generated LUT: $($profile.File)"
}

foreach ($profile in $profiles) {
    $id = Get-DeterministicId $profile.Title
    $presetName = XmlEscape $profile.PresetName
    $description = XmlEscape $profile.Description
    $grayscale = if ($profile.ConvertToGrayscale) { "True" } else { "False" }
    $vignetteAmount = -1.0 * [double]$profile.Vignette

    $builder = New-Object System.Text.StringBuilder
    [void]$builder.AppendLine('<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0-c000 1.000000, 0000/00/00-00:00:00        ">')
    [void]$builder.AppendLine(' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">')
    [void]$builder.AppendLine('  <rdf:Description rdf:about=""')
    [void]$builder.AppendLine('    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"')
    [void]$builder.AppendLine('   crs:PresetType="Normal"')
    [void]$builder.AppendLine('   crs:Cluster="Ricoh GR3 Look"')
    [void]$builder.AppendLine("   crs:UUID=`"$id`"")
    [void]$builder.AppendLine('   crs:SupportsAmount="False"')
    [void]$builder.AppendLine('   crs:SupportsColor="True"')
    [void]$builder.AppendLine('   crs:SupportsMonochrome="True"')
    [void]$builder.AppendLine('   crs:SupportsHighDynamicRange="True"')
    [void]$builder.AppendLine('   crs:SupportsNormalDynamicRange="True"')
    [void]$builder.AppendLine('   crs:SupportsSceneReferred="True"')
    [void]$builder.AppendLine('   crs:SupportsOutputReferred="True"')
    [void]$builder.AppendLine('   crs:RequiresRGBTables="False"')
    [void]$builder.AppendLine('   crs:CameraRawVersion="17.5"')
    [void]$builder.AppendLine('   crs:ProcessVersion="15.4"')
    [void]$builder.AppendLine("   crs:Name=`"$presetName`"")
    [void]$builder.AppendLine("   crs:ShortName=`"$presetName`"")
    [void]$builder.AppendLine('   crs:Group="Ricoh GR3 Look"')
    [void]$builder.AppendLine("   crs:Description=`"$description`"")
    [void]$builder.AppendLine('   crs:Version="17.5"')
    [void]$builder.AppendLine('   crs:WhiteBalance="As Shot"')
    [void]$builder.AppendLine('   crs:IncrementalTemperature="0"')
    [void]$builder.AppendLine('   crs:IncrementalTint="0"')
    [void]$builder.AppendLine('   crs:Exposure2012="0.00"')
    [void]$builder.AppendLine("   crs:Contrast2012=`"$($profile.Contrast.ToString('0', $culture))`"")
    [void]$builder.AppendLine('   crs:Highlights2012="0"')
    [void]$builder.AppendLine('   crs:Shadows2012="0"')
    [void]$builder.AppendLine('   crs:Whites2012="0"')
    [void]$builder.AppendLine('   crs:Blacks2012="0"')
    [void]$builder.AppendLine('   crs:Texture="0"')
    [void]$builder.AppendLine('   crs:Clarity2012="0"')
    [void]$builder.AppendLine('   crs:Dehaze="0"')
    [void]$builder.AppendLine("   crs:Vibrance=`"$($profile.Vibrance.ToString('0', $culture))`"")
    [void]$builder.AppendLine("   crs:Saturation=`"$($profile.Saturation.ToString('0', $culture))`"")
    [void]$builder.AppendLine('   crs:ParametricShadows="0"')
    [void]$builder.AppendLine('   crs:ParametricDarks="0"')
    [void]$builder.AppendLine('   crs:ParametricLights="0"')
    [void]$builder.AppendLine('   crs:ParametricHighlights="0"')
    [void]$builder.AppendLine('   crs:ParametricShadowSplit="25"')
    [void]$builder.AppendLine('   crs:ParametricMidtoneSplit="50"')
    [void]$builder.AppendLine('   crs:ParametricHighlightSplit="75"')
    [void]$builder.AppendLine("   crs:Sharpness=`"$($profile.Sharpness.ToString('0', $culture))`"")
    [void]$builder.AppendLine('   crs:SharpenRadius="1.0"')
    [void]$builder.AppendLine('   crs:SharpenDetail="25"')
    [void]$builder.AppendLine('   crs:SharpenEdgeMasking="0"')
    [void]$builder.AppendLine('   crs:LuminanceSmoothing="0"')
    [void]$builder.AppendLine('   crs:ColorNoiseReduction="0"')
    [void]$builder.AppendLine('   crs:SplitToningShadowHue="' + $profile.ShadowHue.ToString('0', $culture) + '"')
    [void]$builder.AppendLine('   crs:SplitToningShadowSaturation="' + $profile.ShadowSat.ToString('0', $culture) + '"')
    [void]$builder.AppendLine('   crs:SplitToningHighlightHue="' + $profile.HighlightHue.ToString('0', $culture) + '"')
    [void]$builder.AppendLine('   crs:SplitToningHighlightSaturation="' + $profile.HighlightSat.ToString('0', $culture) + '"')
    [void]$builder.AppendLine('   crs:SplitToningBalance="' + $profile.SplitBalance.ToString('0', $culture) + '"')
    [void]$builder.AppendLine('   crs:ColorGradeMidtoneHue="0"')
    [void]$builder.AppendLine('   crs:ColorGradeMidtoneSat="0"')
    [void]$builder.AppendLine('   crs:ColorGradeShadowLum="0"')
    [void]$builder.AppendLine('   crs:ColorGradeMidtoneLum="0"')
    [void]$builder.AppendLine('   crs:ColorGradeHighlightLum="0"')
    [void]$builder.AppendLine('   crs:ColorGradeBlending="50"')
    [void]$builder.AppendLine('   crs:ColorGradeGlobalHue="0"')
    [void]$builder.AppendLine('   crs:ColorGradeGlobalSat="0"')
    [void]$builder.AppendLine('   crs:ColorGradeGlobalLum="0"')
    [void]$builder.AppendLine('   crs:LensProfileEnable="0"')
    [void]$builder.AppendLine('   crs:VignetteAmount="0"')
    [void]$builder.AppendLine("   crs:GrainAmount=`"$($profile.Grain.ToString('0', $culture))`"")
    [void]$builder.AppendLine('   crs:GrainSize="25"')
    [void]$builder.AppendLine('   crs:GrainFrequency="50"')
    [void]$builder.AppendLine("   crs:PostCropVignetteAmount=`"$($vignetteAmount.ToString('0', $culture))`"")
    [void]$builder.AppendLine('   crs:PostCropVignetteMidpoint="50"')
    [void]$builder.AppendLine('   crs:PostCropVignetteFeather="55"')
    [void]$builder.AppendLine('   crs:PostCropVignetteRoundness="0"')
    [void]$builder.AppendLine('   crs:PostCropVignetteStyle="1"')
    [void]$builder.AppendLine('   crs:PostCropVignetteHighlightContrast="0"')
    [void]$builder.AppendLine("   crs:ConvertToGrayscale=`"$grayscale`"")
    [void]$builder.AppendLine('   crs:OverrideLookVignette="False"')
    [void]$builder.AppendLine('   crs:ToneCurveName2012="Custom"')
    [void]$builder.AppendLine('   crs:HasSettings="True"')
    [void]$builder.AppendLine('   crs:HasCrop="False"')
    [void]$builder.AppendLine('   crs:CropTop="0"')
    [void]$builder.AppendLine('   crs:CropLeft="0"')
    [void]$builder.AppendLine('   crs:CropBottom="1"')
    [void]$builder.AppendLine('   crs:CropRight="1"')
    [void]$builder.AppendLine('   crs:CropAngle="0"')
    [void]$builder.AppendLine('   crs:CropConstrainToWarp="0"')
    [void]$builder.AppendLine('   crs:CropConstrainToUnitSquare="1"')
    [void]$builder.AppendLine('   crs:ClipboardOrientation="1"')
    [void]$builder.AppendLine('   crs:ClipboardAspectRatio="1.0"')
    [void]$builder.AppendLine('   crs:SubsetLensBlur="False"')
    [void]$builder.AppendLine('   crs:SubsetRetouch="False"')
    [void]$builder.AppendLine('   crs:MaskMergeOption="1"')
    [void]$builder.AppendLine('   crs:MaskDeleteOption="1">')
    [void]$builder.AppendLine('   <crs:ToneCurvePV2012>')
    [void]$builder.AppendLine('    <rdf:Seq>')
    foreach ($point in $profile.Curve) {
        [void]$builder.AppendLine("     <rdf:li>$($point[0]), $($point[1])</rdf:li>")
    }
    [void]$builder.AppendLine('    </rdf:Seq>')
    [void]$builder.AppendLine('   </crs:ToneCurvePV2012>')
    [void]$builder.AppendLine('  </rdf:Description>')
    [void]$builder.AppendLine(' </rdf:RDF>')
    [void]$builder.AppendLine('</x:xmpmeta>')

    $presetFile = [IO.Path]::ChangeExtension($profile.File, ".xmp")
    $presetTarget = Join-Path $presetDirectory $presetFile
    [IO.File]::WriteAllText($presetTarget, $builder.ToString(), (New-Object System.Text.UTF8Encoding($false)))
    Write-Output "Generated Camera Raw preset: $presetFile"
}