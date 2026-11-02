"""Creates the equal-aspect copies of the screenshots that are shown side by side or stacked as pairs.
The original screenshots are not modified. Needs Pillow.   python report/scripts/make_pair_images.py"""
from pathlib import Path
from PIL import Image

S = Path(__file__).resolve().parents[1] / "screenshots"

# --- Figure 5.2: side-by-side view vs overlay view (both 564 px wide) -> common box 564 x 580
a = Image.open(S / "06_heatmap_side_by_side.png").convert("RGB")      # 564 x 516
b = Image.open(S / "06d_heatmap_overlay.png").convert("RGB")          # 564 x 664
H = 580
pad = a.getpixel((2, a.height - 2))                                    # page background colour of the dashboard
canvas = Image.new("RGB", (a.width, H), pad)
canvas.paste(a, (0, 0))
canvas.save(S / "pair_06_side_by_side.png")
b.crop((0, 0, b.width, H)).save(S / "pair_06d_overlay.png")           # keeps controls + slider; the grey note below is the same text as in the other image

# --- Figure 5.4: 12 is 1190 x 207; take the matching strip of the Face-Tracks page (same aspect ratio)
t = Image.open(S / "13_video_face_tracks.png").convert("RGB")          # 1440 x 900
w = 1230
h = round(w * 207 / 1190)                                              # 214
x0, y0 = 104, 440                                                      # card "Face tracks" (x 104..1336)
t.crop((x0, y0, x0 + w, y0 + h)).save(S / "pair_13_face_tracks.png")
Image.open(S / "12_video_frame_not_stored.png").convert("RGB").save(S / "pair_12_not_stored.png")
for n in ("pair_06_side_by_side", "pair_06d_overlay", "pair_13_face_tracks", "pair_12_not_stored"):
    print(n, Image.open(S / (n + ".png")).size)
