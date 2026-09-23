using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;

public static class RicohGr3Preview
{
    private sealed class Lut
    {
        public int Size;
        public float[, , ,] Values;

        public static Lut Load(string path)
        {
            string[] lines = File.ReadAllLines(path);
            Lut lut = new Lut();
            int parsed = 0;
            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i].Trim();
                if (line.StartsWith("LUT_3D_SIZE"))
                {
                    lut.Size = int.Parse(line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries)[1], CultureInfo.InvariantCulture);
                    lut.Values = new float[lut.Size, lut.Size, lut.Size, 3];
                }
                else if (!line.StartsWith("TITLE") && !line.StartsWith("DOMAIN_") && line.Length > 0)
                {
                    string[] parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length == 3 && lut.Values != null)
                    {
                        int b = parsed / (lut.Size * lut.Size);
                        int remainder = parsed % (lut.Size * lut.Size);
                        int g = remainder / lut.Size;
                        int r = remainder % lut.Size;
                        lut.Values[r, g, b, 0] = float.Parse(parts[0], CultureInfo.InvariantCulture);
                        lut.Values[r, g, b, 1] = float.Parse(parts[1], CultureInfo.InvariantCulture);
                        lut.Values[r, g, b, 2] = float.Parse(parts[2], CultureInfo.InvariantCulture);
                        parsed++;
                    }
                }
            }
            return lut;
        }

        public Color Apply(Color input)
        {
            float r = input.R / 255f;
            float g = input.G / 255f;
            float b = input.B / 255f;
            float rr = r * (Size - 1);
            float gg = g * (Size - 1);
            float bb = b * (Size - 1);
            int r0 = Math.Min((int)Math.Floor(rr), Size - 2);
            int g0 = Math.Min((int)Math.Floor(gg), Size - 2);
            int b0 = Math.Min((int)Math.Floor(bb), Size - 2);
            int r1 = r0 + 1;
            int g1 = g0 + 1;
            int b1 = b0 + 1;
            float tr = rr - r0;
            float tg = gg - g0;
            float tb = bb - b0;
            float[] values = new float[3];
            for (int channel = 0; channel < 3; channel++)
            {
                float c000 = Values[r0, g0, b0, channel];
                float c100 = Values[r1, g0, b0, channel];
                float c010 = Values[r0, g1, b0, channel];
                float c110 = Values[r1, g1, b0, channel];
                float c001 = Values[r0, g0, b1, channel];
                float c101 = Values[r1, g0, b1, channel];
                float c011 = Values[r0, g1, b1, channel];
                float c111 = Values[r1, g1, b1, channel];
                float c00 = c000 + (c100 - c000) * tr;
                float c10 = c010 + (c110 - c010) * tr;
                float c01 = c001 + (c101 - c001) * tr;
                float c11 = c011 + (c111 - c011) * tr;
                float c0 = c00 + (c10 - c00) * tg;
                float c1 = c01 + (c11 - c01) * tg;
                values[channel] = c0 + (c1 - c0) * tb;
            }
            return Color.FromArgb(255,
                Math.Max(0, Math.Min(255, (int)Math.Round(values[0] * 255f))),
                Math.Max(0, Math.Min(255, (int)Math.Round(values[1] * 255f))),
                Math.Max(0, Math.Min(255, (int)Math.Round(values[2] * 255f))));
        }
    }

    private sealed class Item
    {
        public string Label;
        public string File;

        public Item(string label, string file)
        {
            Label = label;
            File = file;
        }
    }

    public static void Generate(string lutDirectory, string outputPath)
    {
        Item[] items = new[]
        {
            new Item("ORIGINAL", null),
            new Item("STANDARD", "GR3_Standard.cube"),
            new Item("POSITIVE FILM", "GR3_Positive_Film.cube"),
            new Item("NEGATIVE FILM", "GR3_Negative_Film.cube"),
            new Item("VIVID STREET", "GR3_Vivid_Street.cube"),
            new Item("BLEACH BYPASS", "GR3_Bleach_Bypass.cube"),
            new Item("HIGH CONTRAST B&W", "GR3_High_Contrast_BW.cube")
        };

        const int sceneWidth = 240;
        const int sceneHeight = 640;
        const int panelWidth = 252;
        const int header = 80;
        const int footer = 55;
        int outputWidth = 20 + items.Length * panelWidth + (items.Length - 1) * 8;
        int outputHeight = header + sceneHeight + footer;

        using (Bitmap scene = new Bitmap(sceneWidth, sceneHeight, PixelFormat.Format24bppRgb))
        {
            using (Graphics graphics = Graphics.FromImage(scene))
            {
                graphics.SmoothingMode = SmoothingMode.AntiAlias;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;
                DrawScene(graphics, sceneWidth, sceneHeight);
            }

            using (Bitmap sheet = new Bitmap(outputWidth, outputHeight, PixelFormat.Format24bppRgb))
            using (Graphics sheetGraphics = Graphics.FromImage(sheet))
            {
                sheetGraphics.SmoothingMode = SmoothingMode.AntiAlias;
                sheetGraphics.Clear(Color.FromArgb(18, 19, 20));

                using (Font titleFont = new Font("Segoe UI Semibold", 22, FontStyle.Bold, GraphicsUnit.Pixel))
                using (Font subtitleFont = new Font("Segoe UI", 11, FontStyle.Regular, GraphicsUnit.Pixel))
                using (Font labelFont = new Font("Segoe UI Semibold", 13, FontStyle.Bold, GraphicsUnit.Pixel))
                using (Font smallFont = new Font("Segoe UI", 10, FontStyle.Regular, GraphicsUnit.Pixel))
                using (SolidBrush white = new SolidBrush(Color.FromArgb(242, 242, 238)))
                using (SolidBrush muted = new SolidBrush(Color.FromArgb(165, 170, 172)))
                using (Pen border = new Pen(Color.FromArgb(78, 82, 84), 1))
                {
                    sheetGraphics.DrawString("RICOH GR III / IMAGE CONTROL LOOKS", titleFont, white, 20, 15);
                    sheetGraphics.DrawString("Synthetic reference: color, skin, sky, foliage, street light and monochrome behavior", subtitleFont, muted, 22, 48);

                    for (int i = 0; i < items.Length; i++)
                    {
                        Item item = items[i];
                        Bitmap panel = item.File == null ? (Bitmap)scene.Clone() : ApplyLut(scene, Path.Combine(lutDirectory, item.File));
                        int x = 20 + i * (panelWidth + 8);
                        int y = header;
                        sheetGraphics.DrawImage(panel, x, y, panelWidth, sceneHeight);
                        sheetGraphics.DrawRectangle(border, x, y, panelWidth - 1, sceneHeight - 1);
                        sheetGraphics.DrawString(item.Label, labelFont, white, x + 1, y + sceneHeight + 12);
                        string detail = item.File == null ? "camera baseline" : "33 cubed LUT";
                        sheetGraphics.DrawString(detail, smallFont, muted, x + 1, y + sceneHeight + 31);
                        panel.Dispose();
                    }
                }

                sheet.Save(outputPath, ImageFormat.Png);
            }
        }
    }

    private static Bitmap ApplyLut(Bitmap source, string lutPath)
    {
        Lut lut = Lut.Load(lutPath);
        Bitmap target = new Bitmap(source.Width, source.Height, PixelFormat.Format24bppRgb);
        for (int y = 0; y < source.Height; y++)
        {
            for (int x = 0; x < source.Width; x++)
            {
                target.SetPixel(x, y, lut.Apply(source.GetPixel(x, y)));
            }
        }
        return target;
    }

    private static void DrawScene(Graphics graphics, int width, int height)
    {
        Rectangle skyRect = new Rectangle(0, 0, width, (int)(height * 0.62));
        using (LinearGradientBrush sky = new LinearGradientBrush(skyRect, Color.FromArgb(55, 103, 166), Color.FromArgb(202, 194, 171), 90f))
        {
            graphics.FillRectangle(sky, skyRect);
        }

        using (SolidBrush sun = new SolidBrush(Color.FromArgb(245, 218, 153)))
        using (SolidBrush sunGlow = new SolidBrush(Color.FromArgb(65, 246, 211, 145)))
        {
            graphics.FillEllipse(sunGlow, width * 0.68f, height * 0.09f, 82, 82);
            graphics.FillEllipse(sun, width * 0.72f, height * 0.13f, 38, 38);
        }

        using (SolidBrush distant = new SolidBrush(Color.FromArgb(126, 132, 132)))
        using (SolidBrush building = new SolidBrush(Color.FromArgb(82, 78, 73)))
        using (SolidBrush buildingWarm = new SolidBrush(Color.FromArgb(119, 91, 72)))
        using (SolidBrush windowOn = new SolidBrush(Color.FromArgb(236, 177, 93)))
        using (SolidBrush windowOff = new SolidBrush(Color.FromArgb(62, 75, 83)))
        {
            graphics.FillRectangle(distant, 0, height * 0.47f, width, height * 0.16f);
            graphics.FillRectangle(building, width * 0.03f, height * 0.31f, width * 0.24f, height * 0.31f);
            graphics.FillRectangle(buildingWarm, width * 0.25f, height * 0.37f, width * 0.25f, height * 0.25f);
            graphics.FillRectangle(building, width * 0.49f, height * 0.28f, width * 0.20f, height * 0.34f);
            graphics.FillRectangle(buildingWarm, width * 0.68f, height * 0.34f, width * 0.29f, height * 0.28f);

            for (int row = 0; row < 5; row++)
            {
                for (int col = 0; col < 4; col++)
                {
                    float x = width * 0.06f + col * 13f;
                    float y = height * 0.35f + row * 26f;
                    graphics.FillRectangle((row + col) % 3 == 0 ? windowOn : windowOff, x, y, 7, 11);
                }
                for (int col = 0; col < 5; col++)
                {
                    float x = width * 0.29f + col * 12f;
                    float y = height * 0.40f + row * 25f;
                    graphics.FillRectangle((row + col) % 4 == 1 ? windowOn : windowOff, x, y, 6, 10);
                }
                for (int col = 0; col < 4; col++)
                {
                    float x = width * 0.52f + col * 11f;
                    float y = height * 0.32f + row * 25f;
                    graphics.FillRectangle((row + col) % 3 == 2 ? windowOn : windowOff, x, y, 6, 10);
                }
            }
        }

        using (SolidBrush sidewalk = new SolidBrush(Color.FromArgb(150, 143, 129)))
        using (SolidBrush road = new SolidBrush(Color.FromArgb(47, 51, 53)))
        using (Pen curb = new Pen(Color.FromArgb(210, 201, 178), 2))
        {
            graphics.FillRectangle(sidewalk, 0, height * 0.60f, width, height * 0.10f);
            graphics.FillRectangle(road, 0, height * 0.70f, width, height * 0.30f);
            graphics.DrawLine(curb, 0, height * 0.70f, width, height * 0.70f);
        }

        using (Pen lane = new Pen(Color.FromArgb(220, 217, 196), 4))
        {
            lane.DashStyle = DashStyle.Dash;
            lane.DashPattern = new float[] { 5f, 4f };
            graphics.DrawLine(lane, width * 0.50f, height * 0.78f, width * 0.36f, height);
            graphics.DrawLine(lane, width * 0.50f, height * 0.78f, width * 0.72f, height);
        }

        using (SolidBrush crosswalk = new SolidBrush(Color.FromArgb(221, 217, 199)))
        {
            for (int i = 0; i < 5; i++)
            {
                float x = width * 0.08f + i * 30f;
                float y = height * 0.83f + i * 8f;
                graphics.FillPolygon(crosswalk, new[]
                {
                    new PointF(x, y),
                    new PointF(x + 16, y),
                    new PointF(x + 26, y + 46),
                    new PointF(x + 8, y + 46)
                });
            }
        }

        using (SolidBrush leafDark = new SolidBrush(Color.FromArgb(25, 72, 48)))
        using (SolidBrush leaf = new SolidBrush(Color.FromArgb(48, 119, 66)))
        using (SolidBrush leafLight = new SolidBrush(Color.FromArgb(89, 142, 75)))
        {
            graphics.FillEllipse(leafDark, -30, height * 0.42f, 125, 205);
            graphics.FillEllipse(leaf, -12, height * 0.39f, 104, 180);
            graphics.FillEllipse(leafLight, 20, height * 0.45f, 65, 95);
            graphics.FillEllipse(leafDark, width - 65, height * 0.46f, 110, 180);
            graphics.FillEllipse(leaf, width - 45, height * 0.44f, 83, 158);
        }

        using (SolidBrush sign = new SolidBrush(Color.FromArgb(186, 49, 43)))
        using (SolidBrush signFace = new SolidBrush(Color.FromArgb(229, 91, 68)))
        using (Pen pole = new Pen(Color.FromArgb(48, 49, 49), 5))
        {
            graphics.DrawLine(pole, width * 0.83f, height * 0.42f, width * 0.83f, height * 0.77f);
            graphics.FillEllipse(sign, width * 0.76f, height * 0.34f, 47, 47);
            graphics.FillEllipse(signFace, width * 0.78f, height * 0.36f, 35, 35);
        }

        using (SolidBrush coat = new SolidBrush(Color.FromArgb(39, 54, 68)))
        using (SolidBrush skin = new SolidBrush(Color.FromArgb(213, 160, 119)))
        using (SolidBrush hair = new SolidBrush(Color.FromArgb(35, 29, 25)))
        using (SolidBrush bag = new SolidBrush(Color.FromArgb(143, 73, 49)))
        using (SolidBrush trousers = new SolidBrush(Color.FromArgb(27, 31, 33)))
        {
            graphics.FillEllipse(hair, width * 0.40f, height * 0.58f, 24, 25);
            graphics.FillEllipse(skin, width * 0.41f, height * 0.588f, 20, 22);
            graphics.FillRectangle(coat, width * 0.385f, height * 0.62f, 29, 88);
            graphics.FillRectangle(coat, width * 0.36f, height * 0.64f, 11, 65);
            graphics.FillRectangle(coat, width * 0.458f, height * 0.64f, 11, 65);
            graphics.FillRectangle(bag, width * 0.45f, height * 0.67f, 18, 36);
            graphics.FillRectangle(trousers, width * 0.39f, height * 0.755f, 9, 72);
            graphics.FillRectangle(trousers, width * 0.435f, height * 0.755f, 9, 72);
        }
    }
}