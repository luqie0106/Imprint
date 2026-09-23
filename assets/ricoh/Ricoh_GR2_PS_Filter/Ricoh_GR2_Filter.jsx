/*
 * Ricoh GR2 Film Look for Adobe Photoshop
 * Version 1.0
 *
 * A practical, adjustable GR2-inspired look:
 * - Positive Film
 * - Negative Film
 * - High Contrast B&W
 * - Street Positive
 *
 * The script duplicates the selected pixel layer, grades the duplicate,
 * and keeps the original layer untouched.
 */

#target photoshop
#targetengine "main"

app.bringToFront();

(function () {
    var SCRIPT_VERSION = "1.0";
    var DIALOG_TITLE = "Ricoh GR2 Film Look";

    var PROFILE_POSITIVE = 0;
    var PROFILE_NEGATIVE = 1;
    var PROFILE_HI_BW = 2;
    var PROFILE_STREET = 3;

    var PROFILES = [
        {
            name: "GR2 正片 Positive Film",
            shortName: "PositiveFilm",
            curve: [[0, 3], [34, 28], [78, 72], [128, 132], [184, 195], [230, 229], [255, 248]],
            brightness: -1,
            contrast: 9,
            saturation: -6,
            shadows: [-5, 0, 4],
            midtones: [2, -1, -2],
            highlights: [5, -1, -5],
            grainScale: 1.0,
            sharpenScale: 1.0
        },
        {
            name: "GR2 负片 Negative Film",
            shortName: "NegativeFilm",
            curve: [[0, 17], [34, 42], [92, 96], [145, 150], [198, 198], [235, 225], [255, 238]],
            brightness: 1,
            contrast: -4,
            saturation: -11,
            shadows: [-7, 3, 6],
            midtones: [5, 2, -4],
            highlights: [3, 1, -6],
            grainScale: 1.18,
            sharpenScale: 0.72
        },
        {
            name: "GR2 高对比黑白 Hi-BW",
            shortName: "HiBW",
            curve: [[0, 0], [28, 15], [72, 61], [126, 126], [178, 192], [226, 229], [255, 255]],
            brightness: 0,
            contrast: 18,
            saturation: -100,
            shadows: [0, 0, 0],
            midtones: [0, 0, 0],
            highlights: [0, 0, 0],
            grainScale: 1.25,
            sharpenScale: 1.2
        },
        {
            name: "GR2 街头正片 Street Positive",
            shortName: "StreetPositive",
            curve: [[0, 0], [30, 20], [76, 69], [128, 134], [183, 199], [229, 230], [255, 251]],
            brightness: -2,
            contrast: 15,
            saturation: -3,
            shadows: [-6, 1, 5],
            midtones: [3, -2, -3],
            highlights: [6, -1, -6],
            grainScale: 1.1,
            sharpenScale: 1.12
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
            applyGR2Look(doc, PROFILES[options.profileIndex], options);
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

        var intro = dlg.add("statictext", undefined, "复制当前图层并套用 GR2 风格；原图层保持不变。");
        intro.alignment = "left";

        var styleGroup = dlg.add("group");
        styleGroup.orientation = "row";
        styleGroup.alignChildren = "center";
        styleGroup.add("statictext", undefined, "风格");
        var profileInput = styleGroup.add("dropdownlist", undefined, profileNames());
        profileInput.selection = PROFILE_POSITIVE;
        profileInput.preferredSize.width = 300;

        var sliders = dlg.add("panel", undefined, "参数");
        sliders.orientation = "column";
        sliders.alignChildren = "fill";
        sliders.margins = 14;
        sliders.spacing = 8;

        var strength = addSliderRow(sliders, "强度", 100, 10, 100, "%");
        var grain = addSliderRow(sliders, "颗粒", 14, 0, 40, "");
        var sharpen = addSliderRow(sliders, "锐化", 42, 0, 100, "");
        var vignette = addSliderRow(sliders, "暗角", 10, 0, 100, "");

        var note = dlg.add("statictext", undefined, "建议：正片 100/14/42/10；负片 85/16/25/8；黑白 100/18/55/12");
        note.alignment = "left";

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

    function addSliderRow(parent, labelText, initialValue, minValue, maxValue, suffix) {
        var row = parent.add("group");
        row.orientation = "row";
        row.alignChildren = "center";
        row.spacing = 8;

        var label = row.add("statictext", undefined, labelText);
        label.preferredSize.width = 40;

        var slider = row.add("slider", undefined, initialValue, minValue, maxValue);
        slider.preferredSize.width = 240;

        var valueText = row.add("statictext", undefined, String(initialValue) + suffix);
        valueText.preferredSize.width = 42;
        valueText.alignment = "right";

        slider.onChanging = function () {
            valueText.text = Math.round(slider.value) + suffix;
        };

        slider.onChange = function () {
            valueText.text = Math.round(slider.value) + suffix;
        };

        return slider;
    }

    function applyGR2Look(doc, profile, options) {
        var source = doc.activeLayer;
        var sourceName = source.name;
        var duplicate = source.duplicate();
        duplicate.name = "GR2 " + profile.shortName + " [S" + options.strength + " G" + options.grain + "]";
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

        if (profile.saturation !== 0) {
            doc.adjustHueSaturation(0, profile.saturation, 0);
        }

        doc.adjustColorBalance(
            profile.shadows,
            profile.midtones,
            profile.highlights,
            true
        );

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
            duplicate.name = "GR2 " + profile.shortName + " [S" + options.strength + " G" + options.grain + "]  " + sourceName;
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