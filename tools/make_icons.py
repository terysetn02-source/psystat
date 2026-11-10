"""Create the Windows (.ico) and macOS (.icns) icons from one square PNG.

Usage:  python tools/make_icons.py [assets/icon.png]
Needs:  pip install pillow
The PNG should be square and at least 1024 x 1024 pixels (transparent background recommended).
Writes assets/icon.ico and assets/icon.icns next to the source image.
"""
import os
import sys

from PIL import Image

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "icon.png")
if not os.path.exists(src):
    sys.exit(f"Source image not found: {src}\nPut a square PNG (1024x1024 recommended) at assets/icon.png")

out_dir = os.path.dirname(os.path.abspath(src))
img = Image.open(src).convert("RGBA")
side = max(img.size)
square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
big = square.resize((1024, 1024), Image.LANCZOS)

big.save(os.path.join(out_dir, "icon.ico"), format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
big.save(os.path.join(out_dir, "icon.icns"), format="ICNS")
print("wrote", os.path.join(out_dir, "icon.ico"), "and", os.path.join(out_dir, "icon.icns"))
