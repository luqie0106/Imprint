# Generates Adobe Camera Raw / Photoshop preset .xmp files for the GR2 look pack.
# Run: powershell -ExecutionPolicy Bypass -File .\generate_presets.ps1

$ErrorActionPreference = "Stop"
$outputDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\Presets"))
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
}

$baseFields = [ordered]@{
    PresetType = "Normal"
    Cluster = ""
    UUID = ""
    SupportsAmount = "False"
    SupportsColor = "True"
    SupportsMonochrome = "True"
    SupportsHighDynamicRange = "True"
    SupportsNormalDynamicRange = "True"
    SupportsSceneReferred = "True"
    SupportsOutputReferred = "True"
    RequiresRGBTables = "False"
    Copyright = ""
    ContactInfo = ""
    Version = "17.5"
    ProcessVersion = "15.4"
    WhiteBalance = "As Shot"
    Temperature = "0"
    Tint = "0"
    Exposure2012 = "0"
    Contrast2012 = "0"
    Highlights2012 = "0"
    Shadows2012 = "0"
    Whites2012 = "0"
    Blacks2012 = "0"
    Texture = "0"
    Clarity2012 = "0"
    Dehaze = "0"
    Vibrance = "0"
    Saturation = "0"
    ParametricShadows = "0"
    ParametricDarks = "0"
    ParametricLights = "0"
    ParametricHighlights = "0"
    ParametricShadowSplit = "25"
    ParametricMidtoneSplit = "50"
    ParametricHighlightSplit = "75"
    Sharpness = "40"
    SharpenRadius = "+1.0"
    SharpenDetail = "25"
    SharpenEdgeMasking = "0"
    LuminanceSmoothing = "0"
    ColorNoiseReduction = "25"
    ColorNoiseReductionDetail = "50"
    ColorNoiseReductionSmoothness = "50"
    HueAdjustmentRed = "0"
    HueAdjustmentOrange = "0"
    HueAdjustmentYellow = "0"
    HueAdjustmentGreen = "0"
    HueAdjustmentAqua = "0"
    HueAdjustmentBlue = "0"
    HueAdjustmentPurple = "0"
    HueAdjustmentMagenta = "0"
    SaturationAdjustmentRed = "0"
    SaturationAdjustmentOrange = "0"
    SaturationAdjustmentYellow = "0"
    SaturationAdjustmentGreen = "0"
    SaturationAdjustmentAqua = "0"
    SaturationAdjustmentBlue = "0"
    SaturationAdjustmentPurple = "0"
    SaturationAdjustmentMagenta = "0"
    LuminanceAdjustmentRed = "0"
    LuminanceAdjustmentOrange = "0"
    LuminanceAdjustmentYellow = "0"
    LuminanceAdjustmentGreen = "0"
    LuminanceAdjustmentAqua = "0"
    LuminanceAdjustmentBlue = "0"
    LuminanceAdjustmentPurple = "0"
    LuminanceAdjustmentMagenta = "0"
    SplitToningShadowHue = "0"
    SplitToningShadowSaturation = "0"
    SplitToningHighlightHue = "0"
    SplitToningHighlightSaturation = "0"
    SplitToningBalance = "0"
    ColorGradeMidtoneHue = "0"
    ColorGradeMidtoneSat = "0"
    ColorGradeShadowLum = "0"
    ColorGradeMidtoneLum = "0"
    ColorGradeHighlightLum = "0"
    ColorGradeBlending = "50"
    ColorGradeGlobalHue = "0"
    ColorGradeGlobalSat = "0"
    ColorGradeGlobalLum = "0"
    AutoLateralCA = "1"
    LensProfileEnable = "0"
    LensManualDistortionAmount = "0"
    VignetteAmount = "0"
    DefringePurpleAmount = "0"
    DefringePurpleHueLo = "30"
    DefringePurpleHueHi = "70"
    DefringeGreenAmount = "0"
    DefringeGreenHueLo = "40"
    DefringeGreenHueHi = "60"
    GrainAmount = "0"
    GrainSize = "20"
    GrainFrequency = "30"
    PostCropVignetteAmount = "0"
    PostCropVignetteMidpoint = "50"
    PostCropVignetteFeather = "50"
    PostCropVignetteRoundness = "0"
    PostCropVignetteStyle = "1"
    ShadowTint = "0"
    RedHue = "0"
    RedSaturation = "0"
    GreenHue = "0"
    GreenSaturation = "0"
    BlueHue = "0"
    BlueSaturation = "0"
    ConvertToGrayscale = "False"
    HDREditMode = "0"
    OverrideLookVignette = "False"
    ToneCurveName2012 = "Custom"
    HasSettings = "True"
    CameraProfile = "Adobe Standard"
}

$profiles = @(
    @{
        File = "GR2_Positive_Film.xmp"
        Name = "GR2 正片 Positive Film"
        Description = "GR2-inspired positive film look: deep greens, cyan shadows, warm highlights and restrained saturation."
        Group = "GR2 Film Looks"
        UUIDSeed = "GR2_Positive_Film_v1"
        Curve = @(@(0,3), @(34,28), @(78,72), @(128,132), @(184,195), @(230,229), @(255,248))
        Fields = [ordered]@{
            Contrast2012 = "12"
            Highlights2012 = "-26"
            Shadows2012 = "14"
            Whites2012 = "8"
            Blacks2012 = "-9"
            Texture = "5"
            Clarity2012 = "8"
            Dehaze = "0"
            Vibrance = "-5"
            Saturation = "-5"
            Sharpness = "42"
            SharpenRadius = "+0.6"
            SharpenDetail = "28"
            SharpenEdgeMasking = "10"
            LuminanceSmoothing = "3"
            ColorNoiseReduction = "22"
            HueAdjustmentRed = "-3"
            HueAdjustmentOrange = "1"
            HueAdjustmentYellow = "-5"
            HueAdjustmentGreen = "5"
            HueAdjustmentAqua = "1"
            HueAdjustmentBlue = "-2"
            HueAdjustmentMagenta = "2"
            SaturationAdjustmentRed = "6"
            SaturationAdjustmentOrange = "-2"
            SaturationAdjustmentYellow = "-12"
            SaturationAdjustmentGreen = "-18"
            SaturationAdjustmentAqua = "-4"
            SaturationAdjustmentBlue = "4"
            SaturationAdjustmentPurple = "-2"
            SaturationAdjustmentMagenta = "-8"
            LuminanceAdjustmentRed = "4"
            LuminanceAdjustmentOrange = "3"
            LuminanceAdjustmentYellow = "-8"
            LuminanceAdjustmentGreen = "-14"
            LuminanceAdjustmentAqua = "-6"
            LuminanceAdjustmentBlue = "-6"
            LuminanceAdjustmentMagenta = "-2"
            SplitToningShadowHue = "205"
            SplitToningShadowSaturation = "8"
            SplitToningHighlightHue = "42"
            SplitToningHighlightSaturation = "7"
            SplitToningBalance = "-10"
            ColorGradeMidtoneHue = "35"
            ColorGradeMidtoneSat = "3"
            GrainAmount = "14"
            GrainSize = "18"
            GrainFrequency = "42"
            PostCropVignetteAmount = "-8"
            PostCropVignetteMidpoint = "50"
            PostCropVignetteFeather = "72"
            PostCropVignetteRoundness = "15"
        }
    },
    @{
        File = "GR2_Negative_Film.xmp"
        Name = "GR2 负片 Negative Film"
        Description = "GR2-inspired negative film look: lifted blacks, softer contrast, muted color and warm midtones."
        Group = "GR2 Film Looks"
        UUIDSeed = "GR2_Negative_Film_v1"
        Curve = @(@(0,17), @(34,42), @(92,96), @(145,150), @(198,198), @(235,225), @(255,238))
        Fields = [ordered]@{
            Contrast2012 = "-6"
            Highlights2012 = "-18"
            Shadows2012 = "26"
            Whites2012 = "-12"
            Blacks2012 = "13"
            Texture = "-3"
            Clarity2012 = "-8"
            Dehaze = "-3"
            Vibrance = "-7"
            Saturation = "-8"
            Sharpness = "25"
            SharpenRadius = "+0.8"
            SharpenDetail = "20"
            SharpenEdgeMasking = "5"
            LuminanceSmoothing = "8"
            ColorNoiseReduction = "28"
            HueAdjustmentRed = "4"
            HueAdjustmentOrange = "-2"
            HueAdjustmentYellow = "-10"
            HueAdjustmentGreen = "-6"
            HueAdjustmentAqua = "2"
            HueAdjustmentBlue = "1"
            HueAdjustmentPurple = "-4"
            HueAdjustmentMagenta = "-6"
            SaturationAdjustmentRed = "-8"
            SaturationAdjustmentOrange = "-12"
            SaturationAdjustmentYellow = "-15"
            SaturationAdjustmentGreen = "-14"
            SaturationAdjustmentAqua = "-8"
            SaturationAdjustmentBlue = "-10"
            SaturationAdjustmentPurple = "-5"
            SaturationAdjustmentMagenta = "-10"
            LuminanceAdjustmentRed = "6"
            LuminanceAdjustmentOrange = "5"
            LuminanceAdjustmentYellow = "0"
            LuminanceAdjustmentGreen = "-10"
            LuminanceAdjustmentAqua = "-5"
            LuminanceAdjustmentBlue = "-3"
            LuminanceAdjustmentMagenta = "-3"
            SplitToningShadowHue = "185"
            SplitToningShadowSaturation = "10"
            SplitToningHighlightHue = "48"
            SplitToningHighlightSaturation = "6"
            SplitToningBalance = "-15"
            ColorGradeMidtoneHue = "55"
            ColorGradeMidtoneSat = "2"
            GrainAmount = "17"
            GrainSize = "24"
            GrainFrequency = "32"
            PostCropVignetteAmount = "-5"
            PostCropVignetteMidpoint = "50"
            PostCropVignetteFeather = "80"
            PostCropVignetteRoundness = "0"
        }
    },
    @{
        File = "GR2_Hi_BW.xmp"
        Name = "GR2 高对比黑白 Hi-BW"
        Description = "GR2-inspired high contrast monochrome: dense blacks, crisp highlights, firm contrast and visible grain."
        Group = "GR2 Film Looks"
        UUIDSeed = "GR2_Hi_BW_v1"
        Curve = @(@(0,0), @(28,15), @(72,61), @(126,126), @(178,192), @(226,229), @(255,255))
        Fields = [ordered]@{
            ConvertToGrayscale = "True"
            Contrast2012 = "20"
            Highlights2012 = "-18"
            Shadows2012 = "8"
            Whites2012 = "5"
            Blacks2012 = "-10"
            Texture = "10"
            Clarity2012 = "15"
            Dehaze = "3"
            Sharpness = "52"
            SharpenRadius = "+0.8"
            SharpenDetail = "32"
            SharpenEdgeMasking = "12"
            LuminanceSmoothing = "2"
            ColorNoiseReduction = "20"
            GrayMixerRed = "34"
            GrayMixerOrange = "24"
            GrayMixerYellow = "-8"
            GrayMixerGreen = "-20"
            GrayMixerAqua = "-10"
            GrayMixerBlue = "-14"
            GrayMixerPurple = "0"
            GrayMixerMagenta = "5"
            GrainAmount = "18"
            GrainSize = "22"
            GrainFrequency = "45"
            PostCropVignetteAmount = "-11"
            PostCropVignetteMidpoint = "50"
            PostCropVignetteFeather = "70"
            PostCropVignetteRoundness = "0"
        }
    },
    @{
        File = "GR2_Street_Positive.xmp"
        Name = "GR2 街头正片 Street Positive"
        Description = "GR2-inspired hard street positive: stronger contrast, crisp detail, restrained color and a slight cool shadow cast."
        Group = "GR2 Film Looks"
        UUIDSeed = "GR2_Street_Positive_v1"
        Curve = @(@(0,0), @(30,20), @(76,69), @(128,134), @(183,199), @(229,230), @(255,251))
        Fields = [ordered]@{
            Contrast2012 = "18"
            Highlights2012 = "-32"
            Shadows2012 = "18"
            Whites2012 = "9"
            Blacks2012 = "-14"
            Texture = "8"
            Clarity2012 = "12"
            Dehaze = "3"
            Vibrance = "-4"
            Saturation = "-4"
            Sharpness = "55"
            SharpenRadius = "+0.6"
            SharpenDetail = "32"
            SharpenEdgeMasking = "8"
            LuminanceSmoothing = "1"
            ColorNoiseReduction = "18"
            HueAdjustmentRed = "-4"
            HueAdjustmentOrange = "1"
            HueAdjustmentYellow = "-6"
            HueAdjustmentGreen = "6"
            HueAdjustmentAqua = "2"
            HueAdjustmentBlue = "-4"
            HueAdjustmentMagenta = "1"
            SaturationAdjustmentRed = "3"
            SaturationAdjustmentOrange = "-4"
            SaturationAdjustmentYellow = "-14"
            SaturationAdjustmentGreen = "-20"
            SaturationAdjustmentAqua = "-6"
            SaturationAdjustmentBlue = "2"
            SaturationAdjustmentPurple = "-4"
            SaturationAdjustmentMagenta = "-10"
            LuminanceAdjustmentRed = "3"
            LuminanceAdjustmentOrange = "2"
            LuminanceAdjustmentYellow = "-10"
            LuminanceAdjustmentGreen = "-16"
            LuminanceAdjustmentAqua = "-8"
            LuminanceAdjustmentBlue = "-7"
            LuminanceAdjustmentMagenta = "-4"
            SplitToningShadowHue = "210"
            SplitToningShadowSaturation = "7"
            SplitToningHighlightHue = "44"
            SplitToningHighlightSaturation = "6"
            SplitToningBalance = "-8"
            ColorGradeMidtoneHue = "38"
            ColorGradeMidtoneSat = "2"
            GrainAmount = "12"
            GrainSize = "16"
            GrainFrequency = "48"
            PostCropVignetteAmount = "-12"
            PostCropVignetteMidpoint = "50"
            PostCropVignetteFeather = "65"
            PostCropVignetteRoundness = "20"
        }
    }
)

$culture = [Globalization.CultureInfo]::InvariantCulture
$utf8 = New-Object System.Text.UTF8Encoding($false)
$md5 = [Security.Cryptography.MD5]::Create()

function Get-StableUuid([string]$seed) {
    $hash = $md5.ComputeHash([Text.Encoding]::UTF8.GetBytes($seed))
    return (($hash | ForEach-Object { $_.ToString("X2") }) -join "")
}

function Get-EscapedAttribute([string]$value) {
    return [Security.SecurityElement]::Escape($value)
}

function New-PresetXmp([hashtable]$profile) {
    $fields = [ordered]@{}
    foreach ($key in $baseFields.Keys) {
        $fields[$key] = $baseFields[$key]
    }
    foreach ($key in $profile.Fields.Keys) {
        $fields[$key] = $profile.Fields[$key]
    }
    $fields["UUID"] = Get-StableUuid $profile.UUIDSeed

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add('<?xpacket begin="' + [char]0xFEFF + '" id="W5M0MpCehiHzreSzNTczkc9d"?>')
    $lines.Add('<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0-c000 1.000000, 0000/00/00-00:00:00        ">')
    $lines.Add(' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">')
    $lines.Add('  <rdf:Description rdf:about=""')
    $lines.Add('    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"')

    foreach ($key in $fields.Keys) {
        $escapedValue = Get-EscapedAttribute ([string]$fields[$key])
        $lines.Add('   crs:' + $key + '="' + $escapedValue + '"')
    }

    $lines.Add('   >')
    $lines.Add('   <crs:Name>')
    $lines.Add('    <rdf:Alt>')
    $lines.Add('     <rdf:li xml:lang="x-default">' + (Get-EscapedAttribute $profile.Name) + '</rdf:li>')
    $lines.Add('    </rdf:Alt>')
    $lines.Add('   </crs:Name>')
    $lines.Add('   <crs:ShortName>')
    $lines.Add('    <rdf:Alt>')
    $lines.Add('     <rdf:li xml:lang="x-default">' + (Get-EscapedAttribute $profile.Name) + '</rdf:li>')
    $lines.Add('    </rdf:Alt>')
    $lines.Add('   </crs:ShortName>')
    $lines.Add('   <crs:Group>')
    $lines.Add('    <rdf:Alt>')
    $lines.Add('     <rdf:li xml:lang="x-default">' + (Get-EscapedAttribute $profile.Group) + '</rdf:li>')
    $lines.Add('    </rdf:Alt>')
    $lines.Add('   </crs:Group>')
    $lines.Add('   <crs:Description>')
    $lines.Add('    <rdf:Alt>')
    $lines.Add('     <rdf:li xml:lang="x-default">' + (Get-EscapedAttribute $profile.Description) + '</rdf:li>')
    $lines.Add('    </rdf:Alt>')
    $lines.Add('   </crs:Description>')
    $lines.Add('   <crs:ToneCurvePV2012>')
    $lines.Add('    <rdf:Seq>')
    foreach ($point in $profile.Curve) {
        $lines.Add('     <rdf:li>' + [string]::Format($culture, "{0}, {1}", $point[0], $point[1]) + '</rdf:li>')
    }
    $lines.Add('    </rdf:Seq>')
    $lines.Add('   </crs:ToneCurvePV2012>')
    $lines.Add('  </rdf:Description>')
    $lines.Add(' </rdf:RDF>')
    $lines.Add('</x:xmpmeta>')
    $lines.Add('<?xpacket end="w"?>')
    return ($lines -join [Environment]::NewLine)
}

foreach ($profile in $profiles) {
    $target = Join-Path $outputDirectory $profile.File
    $xml = New-PresetXmp $profile
    [IO.File]::WriteAllText($target, $xml, (New-Object System.Text.UTF8Encoding($true)))
    Write-Output "Generated $($profile.File)"
}

$md5.Dispose()