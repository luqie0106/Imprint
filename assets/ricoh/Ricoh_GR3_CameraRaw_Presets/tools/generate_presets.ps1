# Generates Adobe Camera Raw / Photoshop preset .xmp files for the GR3 look pack.
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
        File = "GR3_Standard.xmp"
        Name = "GR3 标准 Standard"
        Description = "GR III-inspired standard look: balanced contrast, restrained saturation, cool shadows and gently warm highlights."
        Group = "GR3 Camera Raw Looks"
        UUIDSeed = "GR3_Standard_v1"
        Curve = @(@(0,1), @(32,30), @(92,94), @(156,160), @(220,223), @(255,253))
        Fields = [ordered]@{
            Contrast2012 = "8"
            Highlights2012 = "-10"
            Shadows2012 = "8"
            Whites2012 = "3"
            Blacks2012 = "-5"
            Texture = "5"
            Clarity2012 = "4"
            Dehaze = "0"
            Vibrance = "-3"
            Saturation = "-3"
            Sharpness = "38"
            SharpenRadius = "+0.8"
            SharpenDetail = "25"
            SharpenEdgeMasking = "10"
            LuminanceSmoothing = "2"
            ColorNoiseReduction = "20"
            HueAdjustmentRed = "-1"
            HueAdjustmentOrange = "1"
            HueAdjustmentYellow = "-3"
            HueAdjustmentGreen = "3"
            HueAdjustmentAqua = "1"
            HueAdjustmentBlue = "-2"
            HueAdjustmentMagenta = "1"
            SaturationAdjustmentRed = "2"
            SaturationAdjustmentOrange = "-1"
            SaturationAdjustmentYellow = "-8"
            SaturationAdjustmentGreen = "-10"
            SaturationAdjustmentAqua = "-3"
            SaturationAdjustmentBlue = "3"
            SaturationAdjustmentPurple = "-2"
            SaturationAdjustmentMagenta = "-4"
            LuminanceAdjustmentRed = "2"
            LuminanceAdjustmentOrange = "1"
            LuminanceAdjustmentYellow = "-5"
            LuminanceAdjustmentGreen = "-8"
            LuminanceAdjustmentAqua = "-4"
            LuminanceAdjustmentBlue = "-4"
            LuminanceAdjustmentMagenta = "-2"
            SplitToningShadowHue = "195"
            SplitToningShadowSaturation = "5"
            SplitToningHighlightHue = "45"
            SplitToningHighlightSaturation = "5"
            SplitToningBalance = "-4"
            ColorGradeMidtoneHue = "40"
            ColorGradeMidtoneSat = "1"
            GrainAmount = "7"
            GrainSize = "18"
            GrainFrequency = "35"
            PostCropVignetteAmount = "-6"
            PostCropVignetteMidpoint = "50"
            PostCropVignetteFeather = "70"
            PostCropVignetteRoundness = "10"
        }
    },
    @{
        File = "GR3_Positive_Film.xmp"
        Name = "GR3 正片 Positive Film"
        Description = "GR III-inspired positive film: deep blues, restrained greens, cyan shadows and warm highlights."
        Group = "GR3 Camera Raw Looks"
        UUIDSeed = "GR3_Positive_Film_v1"
        Curve = @(@(0,2), @(30,24), @(72,66), @(128,133), @(190,199), @(232,231), @(255,248))
        Fields = [ordered]@{
            Contrast2012 = "14"
            Highlights2012 = "-28"
            Shadows2012 = "16"
            Whites2012 = "8"
            Blacks2012 = "-12"
            Texture = "4"
            Clarity2012 = "7"
            Dehaze = "0"
            Vibrance = "-5"
            Saturation = "-6"
            Sharpness = "46"
            SharpenRadius = "+0.7"
            SharpenDetail = "30"
            SharpenEdgeMasking = "12"
            LuminanceSmoothing = "4"
            ColorNoiseReduction = "22"
            HueAdjustmentRed = "-4"
            HueAdjustmentOrange = "1"
            HueAdjustmentYellow = "-6"
            HueAdjustmentGreen = "7"
            HueAdjustmentAqua = "2"
            HueAdjustmentBlue = "-3"
            HueAdjustmentMagenta = "2"
            SaturationAdjustmentRed = "6"
            SaturationAdjustmentOrange = "-2"
            SaturationAdjustmentYellow = "-14"
            SaturationAdjustmentGreen = "-20"
            SaturationAdjustmentAqua = "-5"
            SaturationAdjustmentBlue = "6"
            SaturationAdjustmentPurple = "-3"
            SaturationAdjustmentMagenta = "-9"
            LuminanceAdjustmentRed = "4"
            LuminanceAdjustmentOrange = "3"
            LuminanceAdjustmentYellow = "-9"
            LuminanceAdjustmentGreen = "-15"
            LuminanceAdjustmentAqua = "-7"
            LuminanceAdjustmentBlue = "-8"
            LuminanceAdjustmentMagenta = "-3"
            SplitToningShadowHue = "190"
            SplitToningShadowSaturation = "10"
            SplitToningHighlightHue = "43"
            SplitToningHighlightSaturation = "8"
            SplitToningBalance = "-12"
            ColorGradeMidtoneHue = "36"
            ColorGradeMidtoneSat = "2"
            GrainAmount = "12"
            GrainSize = "18"
            GrainFrequency = "42"
            PostCropVignetteAmount = "-10"
            PostCropVignetteMidpoint = "50"
            PostCropVignetteFeather = "72"
            PostCropVignetteRoundness = "15"
        }
    },
    @{
        File = "GR3_Negative_Film.xmp"
        Name = "GR3 负片 Negative Film"
        Description = "GR III-inspired negative film: lifted blacks, soft contrast, muted color and creamy highlights."
        Group = "GR3 Camera Raw Looks"
        UUIDSeed = "GR3_Negative_Film_v1"
        Curve = @(@(0,16), @(32,38), @(84,88), @(132,139), @(186,185), @(230,220), @(255,241))
        Fields = [ordered]@{
            Contrast2012 = "-7"
            Highlights2012 = "-20"
            Shadows2012 = "28"
            Whites2012 = "-14"
            Blacks2012 = "15"
            Texture = "-3"
            Clarity2012 = "-8"
            Dehaze = "-3"
            Vibrance = "-7"
            Saturation = "-9"
            Sharpness = "24"
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
            SaturationAdjustmentYellow = "-16"
            SaturationAdjustmentGreen = "-15"
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
            SplitToningShadowHue = "205"
            SplitToningShadowSaturation = "12"
            SplitToningHighlightHue = "48"
            SplitToningHighlightSaturation = "6"
            SplitToningBalance = "-15"
            ColorGradeMidtoneHue = "55"
            ColorGradeMidtoneSat = "2"
            GrainAmount = "16"
            GrainSize = "24"
            GrainFrequency = "32"
            PostCropVignetteAmount = "-6"
            PostCropVignetteMidpoint = "50"
            PostCropVignetteFeather = "80"
            PostCropVignetteRoundness = "0"
        }
    },
    @{
        File = "GR3_Vivid_Street.xmp"
        Name = "GR3 鲜明街头 Vivid Street"
        Description = "GR III-inspired vivid street: hard contrast, strong blues, crisp detail and controlled skin tones."
        Group = "GR3 Camera Raw Looks"
        UUIDSeed = "GR3_Vivid_Street_v1"
        Curve = @(@(0,0), @(24,15), @(72,70), @(128,137), @(192,204), @(236,237), @(255,250))
        Fields = [ordered]@{
            Contrast2012 = "20"
            Highlights2012 = "-34"
            Shadows2012 = "18"
            Whites2012 = "10"
            Blacks2012 = "-15"
            Texture = "10"
            Clarity2012 = "13"
            Dehaze = "4"
            Vibrance = "4"
            Saturation = "3"
            Sharpness = "56"
            SharpenRadius = "+0.6"
            SharpenDetail = "34"
            SharpenEdgeMasking = "8"
            LuminanceSmoothing = "1"
            ColorNoiseReduction = "16"
            HueAdjustmentRed = "-4"
            HueAdjustmentOrange = "1"
            HueAdjustmentYellow = "-6"
            HueAdjustmentGreen = "7"
            HueAdjustmentAqua = "2"
            HueAdjustmentBlue = "-5"
            HueAdjustmentPurple = "-1"
            HueAdjustmentMagenta = "1"
            SaturationAdjustmentRed = "5"
            SaturationAdjustmentOrange = "-3"
            SaturationAdjustmentYellow = "-12"
            SaturationAdjustmentGreen = "-18"
            SaturationAdjustmentAqua = "-4"
            SaturationAdjustmentBlue = "8"
            SaturationAdjustmentPurple = "-4"
            SaturationAdjustmentMagenta = "-10"
            LuminanceAdjustmentRed = "3"
            LuminanceAdjustmentOrange = "2"
            LuminanceAdjustmentYellow = "-9"
            LuminanceAdjustmentGreen = "-15"
            LuminanceAdjustmentAqua = "-7"
            LuminanceAdjustmentBlue = "-9"
            LuminanceAdjustmentMagenta = "-4"
            SplitToningShadowHue = "210"
            SplitToningShadowSaturation = "8"
            SplitToningHighlightHue = "42"
            SplitToningHighlightSaturation = "6"
            SplitToningBalance = "-8"
            ColorGradeMidtoneHue = "38"
            ColorGradeMidtoneSat = "2"
            GrainAmount = "10"
            GrainSize = "16"
            GrainFrequency = "48"
            PostCropVignetteAmount = "-13"
            PostCropVignetteMidpoint = "50"
            PostCropVignetteFeather = "65"
            PostCropVignetteRoundness = "20"
        }
    },
    @{
        File = "GR3_Bleach_Bypass.xmp"
        Name = "GR3 漂白负冲 Bleach Bypass"
        Description = "GR III-inspired bleach bypass: high contrast, low saturation and metallic gray tonality."
        Group = "GR3 Camera Raw Looks"
        UUIDSeed = "GR3_Bleach_Bypass_v1"
        Curve = @(@(0,0), @(20,8), @(60,48), @(124,131), @(188,204), @(232,238), @(255,255))
        Fields = [ordered]@{
            Contrast2012 = "25"
            Highlights2012 = "-35"
            Shadows2012 = "12"
            Whites2012 = "12"
            Blacks2012 = "-18"
            Texture = "8"
            Clarity2012 = "18"
            Dehaze = "5"
            Vibrance = "-18"
            Saturation = "-28"
            Sharpness = "44"
            SharpenRadius = "+0.8"
            SharpenDetail = "30"
            SharpenEdgeMasking = "15"
            LuminanceSmoothing = "2"
            ColorNoiseReduction = "22"
            HueAdjustmentRed = "-2"
            HueAdjustmentYellow = "-3"
            HueAdjustmentGreen = "4"
            HueAdjustmentAqua = "1"
            HueAdjustmentBlue = "-2"
            SaturationAdjustmentRed = "-14"
            SaturationAdjustmentOrange = "-18"
            SaturationAdjustmentYellow = "-24"
            SaturationAdjustmentGreen = "-28"
            SaturationAdjustmentAqua = "-22"
            SaturationAdjustmentBlue = "-24"
            SaturationAdjustmentPurple = "-20"
            SaturationAdjustmentMagenta = "-23"
            LuminanceAdjustmentRed = "3"
            LuminanceAdjustmentOrange = "2"
            LuminanceAdjustmentYellow = "-6"
            LuminanceAdjustmentGreen = "-8"
            LuminanceAdjustmentAqua = "-5"
            LuminanceAdjustmentBlue = "-7"
            LuminanceAdjustmentMagenta = "-4"
            SplitToningShadowHue = "205"
            SplitToningShadowSaturation = "6"
            SplitToningHighlightHue = "45"
            SplitToningHighlightSaturation = "4"
            SplitToningBalance = "0"
            GrainAmount = "14"
            GrainSize = "20"
            GrainFrequency = "40"
            PostCropVignetteAmount = "-11"
            PostCropVignetteMidpoint = "50"
            PostCropVignetteFeather = "68"
            PostCropVignetteRoundness = "10"
        }
    },
    @{
        File = "GR3_High_Contrast_BW.xmp"
        Name = "GR3 高反差黑白 High Contrast B&W"
        Description = "GR III-inspired high-contrast monochrome: deep blacks, bright whites and visible grain."
        Group = "GR3 Camera Raw Looks"
        UUIDSeed = "GR3_High_Contrast_BW_v1"
        Curve = @(@(0,0), @(18,5), @(58,44), @(126,129), @(182,202), @(226,235), @(255,255))
        Fields = [ordered]@{
            ConvertToGrayscale = "True"
            Contrast2012 = "24"
            Highlights2012 = "-26"
            Shadows2012 = "10"
            Whites2012 = "8"
            Blacks2012 = "-14"
            Texture = "12"
            Clarity2012 = "18"
            Dehaze = "4"
            Sharpness = "54"
            SharpenRadius = "+0.8"
            SharpenDetail = "32"
            SharpenEdgeMasking = "15"
            LuminanceSmoothing = "2"
            ColorNoiseReduction = "20"
            GrayMixerRed = "38"
            GrayMixerOrange = "18"
            GrayMixerYellow = "-12"
            GrayMixerGreen = "-24"
            GrayMixerAqua = "-14"
            GrayMixerBlue = "-18"
            GrayMixerPurple = "0"
            GrayMixerMagenta = "6"
            GrainAmount = "20"
            GrainSize = "22"
            GrainFrequency = "45"
            PostCropVignetteAmount = "-15"
            PostCropVignetteMidpoint = "50"
            PostCropVignetteFeather = "70"
            PostCropVignetteRoundness = "0"
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