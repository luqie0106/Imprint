/*
 * Ricoh GR III Film Look for Adobe Photoshop
 * Version 1.0
 *
 * Six practical GR III-inspired Image Control looks:
 * - Standard
 * - Positive Film
 * - Negative Film
 * - Vivid Street
 * - Bleach Bypass
 * - High Contrast B&W
 *
 * The script duplicates the selected pixel layer, grades the duplicate,
 * and leaves the original layer untouched.
 */

#target photoshop
#targetengine "main"

app.bringToFront();

(function () {
    var SCRIPT_VERSION = "1.0";
    var DIALOG_TITLE = "Ricoh GR III Film Look";
    var PROFILE_POSITIVE = 1;

    var PROFILES = [
        {
            name: "GR3 标准 Standard",
            shortName: "Standard",
            curve: [[0, 1], [32, 30], [92, 94], [156, 160], [220, 223], [255, 253]],
            brightness: 0,
            contrast: 7,
            saturation: -4,
            shadows: [-2, 1, 2],
            midtones: [1, 1, -1],
            highlights: [3, 0, -3],
            grainScale: 0.88,
            sharpenScale: 1.00,
            defaults: { strength: 85, grain: 8, sharpen: 34, vignette: 7 },
            note: "标准：保留 GR 的清晰感，轻微压低饱和，阴影冷、高光暖。建议 85/8/34/7。"
        },
        {
            name: "GR3 正片 Positive Film",
            shortName: "PositiveFilm",
            curve: [[0, 2], [30, 24], [72, 66], [128, 133], [190, 199], [232, 231], [255, 248]],
            brightness: -2,
            contrast: 12,
            saturation: -7,
            shadows: [-7, 1, 6],
            midtones: [4, -1, -5],
            highlights: [8, 0, -7],
            grainScale: 1.05,
            sharpenScale: 1.12,
            defaults: { strength: 100, grain: 12, sharpen: 48, vignette: 10 },
            note: "正片：GR 招牌风格。暗部偏青，红暖高光，绿色稍收，蓝更沉。建议 100/12/48/10。"
        },
        {
            name: "GR3 负片 Negative Film",
            shortName: "NegativeFilm",
            curve: [[0, 16], [32, 38], [84, 88], [132, 139], [186, 185], [230, 220], [255, 241]],
            brightness: 0,
            contrast: -5,
            saturation: -12,
            shadows: [-4, 2, 5],
            midtones: [5, 0, -5],
            highlights: [4, 1, -6],
            grainScale: 1.18,
            sharpenScale: 0.74,
            defaults: { strength: 90, grain: 15, sharpen: 28, vignette: 8 },
            note: "负片：黑位抬起、反差柔和、颜色克制，适合逆光和生活片段。建议 90/15/28/8。"
        },
        {
            name: "GR3 鲜明街头 Vivid Street",
            shortName: "VividStreet",
            curve: [[0, 0], [24, 15], [72, 70], [128, 137], [192, 204], [236, 237], [255, 250]],
            brightness: -3,
            contrast: 18,
            saturation: 6,
            shadows: [-6, 1, 4],
            midtones: [4, -2, -2],
            highlights: [5, -1, -8],
            grainScale: 0.92,
            sharpenScale: 1.24,
            defaults: { strength: 100, grain: 11, sharpen: 56, vignette: 14 },
            note: "鲜明街头：更硬的反差、更深的蓝和更有力的锐度。建议 100/11/56/14。"
        },
        {
            name: "GR3 漂白负冲 Bleach Bypass",
            shortName: "BleachBypass",
            curve: [[0, 0], [20, 8], [60, 48], [124, 131], [188, 204], [232, 238], [255, 255]],
            brightness: -2,
            contrast: 23,
            saturation: -40,
            shadows: [-2, 1, 3],
            midtones: [1, 0, -1],
            highlights: [2, -1, -2],
            grainScale: 1.12,
            sharpenScale: 1.10,
            defaults: { strength: 95, grain: 14, sharpen: 45, vignette: 12 },
            note: "漂白负冲：高反差、低饱和、金属灰质感，适合建筑与硬光。建议 95/14/45/12。"
        },
        {
            name: "GR3 高反差黑白 High Contrast B&W",
            shortName: "HighContrastBW",
            curve: [[0, 0], [18, 5], [58, 44], [126, 129], [182, 202], [226, 235], [255, 255]],
            brightness: -1,
            contrast: 22,
            saturation: -100,
            shadows: [0, 0, 0],
            midtones: [0, 0, 0],
            highlights: [0, 0, 0],
            grainScale: 1.35,
            sharpenScale: 1.18,
            defaults: { strength: 100, grain: 20, sharpen: 62, vignette: 15 },
            note: "高反差黑白：暗部更实、亮部更硬，颗粒可见。建议 100/20/62/15。"
        }
    ];

    function main() {
        if (app.documents.length === 0) {
            alert("请先打开一张图片。");
            return;
        }

        var doc = app.activeDocument;

        if (doc.mode !== DocumentMode.RGB) {
            alert("此滤镜针对 RGB 图像调校。请先执行“图像 > 模式 > RGB 颜色”。");
            return;
        }

        if (doc.bitsPerChannel === BitsPerChannelType.THIRTYTWO) {
            alert("暂不支持 32 位/通道。请先转换为 16 位或 8 位/通道。");
            return;
        }

        var source = doc.activeLayer;
        if (source.typename !== "ArtLayer") {
            alert("请选择一个像素图层、智能对象、文字图层或形状图层，不要选择图层组。");
            return;
        }

        var options = showDialog();
        if (!options) {
            return;
        }

        try {
            applyGR3Look(doc, PROFILES[options.profileIndex], options);
        } catch (error) {
            alert("滤镜应用失败：\n" + error.message);
        }
    }

    function showDialog() {
        var dlg = new Window("dialog", DIALOG_TITLE + "  v" + SCRIPT_VERSION);
        dlg.orientation = "column";
        dlg.alignChildren = "fill";
        dlg.margins = 18;
        dlg.spacing = 10;

        var intro = dlg.add("statictext", undefined, "复制当前图层并套用 GR3 风格；原图层保持不变。");
        intro.alignment = "left";

        var styleGroup = dlg.add("group");
        styleGroup.orientation = "row";
        styleGroup.alignChildren = "center";
        styleGroup.add("statictext", undefined, "风格");
        var profileInput = styleGroup.add("dropdownlist", undefined, profileNames());
        profileInput.selection = PROFILE_POSITIVE;
        profileInput.preferredSize.width = 320;

        var sliders = dlg.add("panel", undefined, "参数");
        sliders.orientation = "column";
        sliders.alignChildren = "fill";
        sliders.margins = 14;
        sliders.spacing = 8;

        var strength = addSliderRow(sliders, "强度", 100, 10, 100, "%");
        var grain = addSliderRow(sliders, "颗粒", 12, 0, 40, "");
        var sharpen = addSliderRow(sliders, "锐化", 48, 0, 100, "");
        var vignette = addSliderRow(sliders, "暗角", 10, 0, 100, "");

        var note = dlg.add("statictext", undefined, PROFILES[PROFILE_POSITIVE].note);
        note.alignment = "left";
        note.preferredSize.width = 410;

        profileInput.onChange = function () {
            updateProfileDefaults(
                profileInput.selection.index,
                strength,
                grain,
                sharpen,
                vignette,
                note
            );
        };

        var buttons = dlg.add("group");
        buttons.orientation = "row";
        buttons.alignment = "right";
        buttons.add("button", undefined, "应用", { name: "ok" });
        buttons.add("button", undefined, "取消", { name: "cancel" });

        if (dlg.show() !== 1) {
            return null;
        }

        return {
            profileIndex: profileInput.selection.index,
            strength: Math.round(strength.value),
            grain: Math.round(grain.value),
            sharpen: Math.round(sharpen.value),
            vignette: Math.round(vignette.value)
        };
    }

    function profileNames() {
        var names = [];
        for (var i = 0; i < PROFILES.length; i++) {
            names.push(PROFILES[i].name);
        }
        return names;
    }

    function updateProfileDefaults(profileIndex, strength, grain, sharpen, vignette, note) {
        var profile = PROFILES[profileIndex];
        setSliderValue(strength, profile.defaults.strength);
        setSliderValue(grain, profile.defaults.grain);
        setSliderValue(sharpen, profile.defaults.sharpen);
        setSliderValue(vignette, profile.defaults.vignette);
        note.text = profile.note;
    }

    function addSliderRow(parent, labelText, initialValue, minValue, maxValue, suffix) {
        var row = parent.add("group");
        row.orientation = "row";
        row.alignChildren = "center";
        row.spacing = 8;

        var label = row.add("statictext", undefined, labelText);
        label.preferredSize.width = 40;

        var slider = row.add("slider", undefined, initialValue, minValue, maxValue);
        slider.preferredSize.width = 250;

        var valueText = row.add("statictext", undefined, String(initialValue) + suffix);
        valueText.preferredSize.width = 44;
        valueText.alignment = "right";

        slider.valueText = valueText;
        slider.suffix = suffix;

        slider.onChanging = function () {
            updateSliderText(slider);
        };

        slider.onChange = function () {
            updateSliderText(slider);
        };

        return slider;
    }

    function setSliderValue(slider, value) {
        slider.value = value;
        updateSliderText(slider);
    }

    function updateSliderText(slider) {
        slider.valueText.text = Math.round(slider.value) + slider.suffix;
    }

    function applyGR3Look(doc, profile, options) {
        var source = doc.activeLayer;
        var sourceName = source.name;
        var duplicate = source.duplicate();
        duplicate.name = "GR3 " + profile.shortName + " [S" + options.strength + " G" + options.grain + "]";
        doc.activeLayer = duplicate;

        if (duplicate.kind !== LayerKind.NORMAL) {
            try {
                duplicate.rasterize(RasterizeType.ENTIRELAYER);
            } catch (rasterizeError) {
                duplicate.remove();
                throw new Error("无法栅格化该图层副本。请先将智能对象/文字/形状图层转换为像素图层。");
            }
        }

        try {
            doc.selection.deselect();
        } catch (ignoreDeselectError) {
        }

        doc.adjustCurves(profile.curve);
        doc.adjustBrightnessContrast(profile.brightness, profile.contrast);

        doc.adjustColorBalance(
            profile.shadows,
            profile.midtones,
            profile.highlights,
            true
        );

        if (profile.saturation !== 0) {
            doc.adjustHueSaturation(0, profile.saturation, 0);
        }

        if (options.sharpen > 0) {
            var sharpenAmount = Math.round(options.sharpen * 0.82 * profile.sharpenScale);
            if (sharpenAmount > 0) {
                doc.applyUnSharpMask(sharpenAmount, 0.6, 2);
            }
        }

        if (options.grain > 0) {
            var noiseAmount = options.grain * 0.18 * profile.grainScale;
            if (noiseAmount < 0.1) {
                noiseAmount = 0.1;
            }
            doc.applyAddNoise(noiseAmount, NoiseDistribution.GAUSSIAN, true);
        }

        if (options.vignette > 0) {
            applyVignette(doc, duplicate, options.vignette);
        }

        duplicate.opacity = options.strength;

        try {
            duplicate.name = "GR3 " + profile.shortName + " [S" + options.strength + " G" + options.grain + "]  " + sourceName;
        } catch (ignoreNameError) {
        }
    }

    function applyVignette(doc, layer, amount) {
        var bounds = layer.bounds;
        var left = bounds[0].as("px");
        var top = bounds[1].as("px");
        var right = bounds[2].as("px");
        var bottom = bounds[3].as("px");
        var width = right - left;
        var height = bottom - top;

        if (width <= 1 || height <= 1) {
            return;
        }

        var normalized = amount / 100;
        var spread = 0.025 + normalized * 0.18;
        var feather = Math.max(18, Math.min(width, height) * (0.07 + normalized * 0.14));
        var opacity = 3 + normalized * 15;

        setEllipseSelection(
            left + width * spread,
            top + height * spread,
            right - width * spread,
            bottom - height * spread
        );

        doc.selection.feather(feather);
        doc.selection.invert();

        var black = new SolidColor();
        black.rgb.red = 0;
        black.rgb.green = 0;
        black.rgb.blue = 0;
        doc.selection.fill(black, ColorBlendMode.NORMAL, opacity);
        doc.selection.deselect();
    }

    function setEllipseSelection(left, top, right, bottom) {
        var desc = new ActionDescriptor();
        var selectionRef = new ActionReference();
        selectionRef.putProperty(charIDToTypeID("Chnl"), charIDToTypeID("fsel"));
        desc.putReference(charIDToTypeID("null"), selectionRef);

        var boundsDesc = new ActionDescriptor();
        boundsDesc.putUnitDouble(charIDToTypeID("Top "), charIDToTypeID("#Pxl"), top);
        boundsDesc.putUnitDouble(charIDToTypeID("Left"), charIDToTypeID("#Pxl"), left);
        boundsDesc.putUnitDouble(charIDToTypeID("Btom"), charIDToTypeID("#Pxl"), bottom);
        boundsDesc.putUnitDouble(charIDToTypeID("Rght"), charIDToTypeID("#Pxl"), right);
        desc.putObject(charIDToTypeID("To  "), charIDToTypeID("Elps"), boundsDesc);

        executeAction(charIDToTypeID("setd"), desc, DialogModes.NO);
    }

    main();
})();