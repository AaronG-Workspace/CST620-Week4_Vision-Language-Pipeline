import numpy as np, math, random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONTS = {
 'black':   ('/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc', 0),
 'poppins': ('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf', 0),
 'poppins_i': ('/usr/share/fonts/truetype/google-fonts/Poppins-BoldItalic.ttf', 0),
 'dejavu':  ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 0),
 'caladea': ('/usr/share/fonts/truetype/crosextra/Caladea-Bold.ttf', 0),
 'carlito': ('/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf', 0),
 'regular': ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 0),
 'cond':    ('/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf', 0),
}
def font(name, size):
    p, i = FONTS[name]; return ImageFont.truetype(p, int(size), index=i)

def fit_text(d, text, box, fname, fill, stroke=0, stroke_fill=None, max_size=400, anchor='mm'):
    x0, y0, x1, y1 = box; bw, bh = x1-x0, y1-y0
    s = max_size
    while s > 8:
        f = font(fname, s)
        l, t, r, b = d.textbbox((0, 0), text, font=f, stroke_width=int(stroke*s))
        if r-l <= bw and b-t <= bh: break
        s = int(s*0.93)
    sw = int(stroke*s)
    d.text(((x0+x1)/2, (y0+y1)/2), text, font=f, fill=fill, anchor=anchor, stroke_width=sw, stroke_fill=stroke_fill)
    return s

def grad(W, H, c1, c2, vertical=True):
    t = np.linspace(0, 1, H if vertical else W)
    g = (np.array(c1)[None]*(1-t[:, None]) + np.array(c2)[None]*t[:, None])
    if vertical: a = np.repeat(g[:, None, :], W, 1)
    else: a = np.repeat(g[None, :, :], H, 0)
    return Image.fromarray(a.astype(np.uint8))

def sunburst(img, cx, cy, color, n=18, alpha=70):
    ov = Image.new('RGBA', img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    R = max(img.size)*1.5
    for k in range(n):
        a0 = 2*math.pi*k/n; a1 = a0 + math.pi/n
        d.polygon([(cx, cy), (cx+R*math.cos(a0), cy+R*math.sin(a0)), (cx+R*math.cos(a1), cy+R*math.sin(a1))], fill=color+(alpha,))
    img.alpha_composite(ov)

def stripes(img, color, width=60, alpha=60, angle=30):
    W, H = img.size; ov = Image.new('RGBA', (W*3, H*3), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    for x in range(0, W*3, width*2): d.rectangle([x, 0, x+width, H*3], fill=color+(alpha,))
    ov = ov.rotate(angle, center=(W*1.5, H*1.5)).crop((W, H, 2*W, 2*H)); img.alpha_composite(ov)

def piece(d, kind, x, y, r, col, rng):
    dark = tuple(max(0, int(c*0.7)) for c in col); lite = tuple(min(255, int(c*1.25+20)) for c in col)
    if kind == 'ring':
        d.ellipse([x-r, y-r*0.8, x+r, y+r*0.8], fill=col, outline=dark, width=max(1, int(r*0.12)))
        d.ellipse([x-r*0.38, y-r*0.3, x+r*0.38, y+r*0.3], fill=(250, 248, 240))
    elif kind == 'puff':
        d.ellipse([x-r, y-r*0.85, x+r, y+r*0.85], fill=col, outline=dark, width=max(1, int(r*0.1)))
        d.ellipse([x-r*0.5, y-r*0.6, x-r*0.05, y-r*0.2], fill=lite)
    elif kind == 'flake':
        pts = [(x+r*math.cos(a)*rng.uniform(0.6, 1.2), y+r*0.7*math.sin(a)*rng.uniform(0.6, 1.2)) for a in np.linspace(0, 2*math.pi, 9)[:-1]]
        d.polygon(pts, fill=col, outline=dark)
        d.line([(x-r*0.4, y), (x+r*0.3, y-r*0.15)], fill=lite, width=max(1, int(r*0.12)))
    elif kind == 'square':
        d.rounded_rectangle([x-r, y-r*0.8, x+r, y+r*0.8], radius=int(r*0.3), fill=col, outline=dark, width=max(1, int(r*0.1)))
        for k in (-0.4, 0, 0.4):
            d.line([(x-r*0.8, y+k*r), (x+r*0.8, y+k*r)], fill=dark, width=max(1, int(r*0.08)))
            d.line([(x+k*r, y-r*0.65), (x+k*r, y+r*0.65)], fill=dark, width=max(1, int(r*0.08)))
    elif kind == 'cluster':
        for _ in range(5):
            ox, oy = rng.uniform(-r*0.5, r*0.5), rng.uniform(-r*0.4, r*0.4); rr = r*rng.uniform(0.45, 0.7)
            d.ellipse([x+ox-rr, y+oy-rr*0.8, x+ox+rr, y+oy+rr*0.8], fill=col, outline=dark)
    elif kind == 'berry':
        d.ellipse([x-r*0.7, y-r*0.7, x+r*0.7, y+r*0.7], fill=col)
        d.ellipse([x-r*0.35, y-r*0.45, x-r*0.05, y-r*0.15], fill=lite)

def bowl(img, cx, cy, R, bowl_col, kind, piece_cols, rng, berries=None):
    ov = Image.new('RGBA', img.size, (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
    od.ellipse([cx-R*1.0, cy+R*0.62, cx+R*1.0, cy+R*0.95], fill=(0, 0, 0, 70))
    img.alpha_composite(ov.filter(ImageFilter.GaussianBlur(R*0.06)))
    d = ImageDraw.Draw(img)
    d.chord([cx-R, cy-R*0.9, cx+R, cy+R*0.9], 0, 180, fill=bowl_col, outline=tuple(int(c*0.75) for c in bowl_col), width=4)
    d.ellipse([cx-R, cy-R*0.32, cx+R, cy+R*0.32], fill=tuple(int(c*0.85) for c in bowl_col))
    d.ellipse([cx-R*0.93, cy-R*0.27, cx+R*0.93, cy+R*0.27], fill=(248, 246, 238))
    # spoon handle behind pieces
    d.line([(cx+R*0.25, cy-R*0.05), (cx+R*1.05, cy-R*0.75)], fill=(185, 188, 195), width=int(R*0.09))
    pts = []
    for _ in range(int(55*R/150)):
        a = rng.uniform(0, 2*math.pi); rr = math.sqrt(rng.uniform(0, 1))
        x = cx + R*0.85*rr*math.cos(a); y = cy + R*0.22*rr*math.sin(a) - R*0.25*(1-rr)*rng.uniform(0.6, 1.0)
        pts.append((y, x))
    pts.sort()
    for y, x in pts:
        piece(d, kind, x, y, R*rng.uniform(0.12, 0.17), piece_cols[rng.randrange(len(piece_cols))], rng)
    if berries:
        for _ in range(6):
            x = cx + rng.uniform(-R*0.6, R*0.6); y = cy - R*rng.uniform(0.05, 0.3)
            piece(d, 'berry', x, y, R*0.16, berries[rng.randrange(len(berries))], rng)
    d.ellipse([cx+R*0.12, cy-R*0.18, cx+R*0.42, cy+R*0.02], fill=(200, 203, 210), outline=(150, 153, 160), width=3)

def paper(img, rng, amt=6):
    a = np.asarray(img.convert('RGB')).astype(np.float32)
    n = rng_np(rng).normal(0, amt, a.shape[:2])[..., None]
    return Image.fromarray(np.clip(a+n, 0, 255).astype(np.uint8))

def rng_np(rng): return np.random.default_rng(rng.randrange(1 << 30))

def front_cereal(ds, W, H, rng):
    img = grad(W, H, ds['bg1'], ds['bg2']).convert('RGBA')
    if ds.get('pattern') == 'sunburst': sunburst(img, W*0.5, H*0.68, ds['accent'], alpha=55)
    elif ds.get('pattern') == 'stripes': stripes(img, ds['accent'], width=W//12, alpha=50, angle=ds.get('angle', 30))
    elif ds.get('pattern') == 'dots':
        d = ImageDraw.Draw(img)
        for _ in range(60):
            x, y, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(W*0.01, W*0.03)
            d.ellipse([x-r, y-r, x+r, y+r], fill=ds['accent']+(70,))
    d = ImageDraw.Draw(img)
    lum = lambda c: 0.299*c[0]+0.587*c[1]+0.114*c[2]
    dark = ds['stroke'] if lum(ds['stroke']) < 140 else ds['text']
    # maker badge
    d.rounded_rectangle([W*0.06, H*0.025, W*0.42, H*0.085], radius=int(W*0.03), fill=(255, 255, 255), outline=dark, width=4)
    fit_text(d, ds.get('maker', 'Prairie Mill'), (W*0.08, H*0.03, W*0.40, H*0.08), 'caladea', dark)
    # brand name banner
    by0 = H*0.10
    if ds.get('banner'):
        d.rounded_rectangle([W*0.04, by0, W*0.96, H*0.44], radius=int(W*0.06), fill=ds['banner'])
    lines = ds['name']; n = len(lines); lh = (H*0.32)/n
    for k, ln in enumerate(lines):
        fit_text(d, ln, (W*0.07, by0+k*lh+H*0.01, W*0.93, by0+(k+1)*lh), ds['font'], ds['text'], stroke=0.06, stroke_fill=ds['stroke'])
    # subtitle ribbon
    d.rounded_rectangle([W*0.12, H*0.455, W*0.88, H*0.515], radius=int(W*0.03), fill=dark)
    fit_text(d, ds.get('sub', 'Whole Grain Cereal'), (W*0.15, H*0.46, W*0.85, H*0.51), 'poppins', (255, 255, 255))
    bowl(img, W*0.5, H*0.74, W*0.36, ds.get('bowl', (240, 240, 245)), ds['piece'], ds['pcols'], rng, ds.get('berries'))
    d = ImageDraw.Draw(img)
    # badge burst
    bx, by, br = W*0.82, H*0.585, W*0.12
    pts = [(bx+br*(1 if k % 2 == 0 else 0.78)*math.cos(math.pi*k/12), by+br*(1 if k % 2 == 0 else 0.78)*math.sin(math.pi*k/12)) for k in range(24)]
    d.polygon(pts, fill=ds.get('badge_col', (230, 40, 40)), outline=(255, 255, 255))
    fit_text(d, ds.get('badge', 'NEW!'), (bx-br*0.65, by-br*0.4, bx+br*0.65, by+br*0.4), 'black', (255, 255, 255))
    # net weight
    fit_text(d, ds.get('netwt', 'NET WT 12 OZ (340g)'), (W*0.05, H*0.935, W*0.55, H*0.975), 'cond', ds['text'], stroke=0.08, stroke_fill=ds['stroke'])
    fit_text(d, 'SERVING SUGGESTION', (W*0.60, H*0.94, W*0.95, H*0.97), 'regular', ds['text'])
    return paper(img, rng)

def side_panel(ds, W, H, rng, title='Nutrition Facts'):
    img = grad(W, H, ds['bg2'], ds['bg1']).convert('RGBA'); d = ImageDraw.Draw(img)
    # vertical brand name
    nm = ' '.join(ds['name'])
    t = Image.new('RGBA', (H, W), (0, 0, 0, 0)); td = ImageDraw.Draw(t)
    fit_text(td, nm, (H*0.04, W*0.05, H*0.96, W*0.30), ds['font'], ds['text'], stroke=0.06, stroke_fill=ds['stroke'])
    img.alpha_composite(t.rotate(90, expand=True))
    # nutrition panel
    x0, y0, x1, y1 = W*0.1, H*0.18, W*0.9, H*0.72
    d.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    fit_text(d, title, (x0+6, y0+6, x1-6, y0+(y1-y0)*0.09), 'black', (0, 0, 0))
    yy = y0+(y1-y0)*0.11; f = font('regular', max(10, int(W*0.055)))
    rows = ['Serving size 1 cup (40g)', 'Calories 150', 'Total Fat 2g', 'Sodium 190mg', 'Total Carb 33g', 'Dietary Fiber 4g', 'Total Sugars 9g', 'Protein 3g', 'Vitamin D 2mcg', 'Iron 8mg', 'Potassium 160mg']
    for r in rows:
        if yy > y1-(y1-y0)*0.06: break
        d.line([x0+6, yy, x1-6, yy], fill=(0, 0, 0), width=2)
        d.text((x0+10, yy+3), r, fill=(0, 0, 0), font=f); yy += (y1-y0)*0.075
    d.rectangle([W*0.15, H*0.78, W*0.85, H*0.86], fill=(255, 255, 255))
    for k in range(28):
        xx = W*0.18 + k*(W*0.64/28)
        d.line([xx, H*0.785, xx, H*0.84], fill=(0, 0, 0), width=1 + (k*7 % 3))
    return paper(img, rng)

def top_panel(ds, W, H, rng):
    img = Image.new('RGBA', (W, H), ds['bg1']+(255,)); d = ImageDraw.Draw(img)
    d.line([0, H*0.5, W, H*0.5], fill=tuple(int(c*0.75) for c in ds['bg1']), width=3)
    fit_text(d, ' '.join(ds['name']), (W*0.1, H*0.1, W*0.9, H*0.45), ds['font'], ds['text'], stroke=0.06, stroke_fill=ds['stroke'])
    fit_text(d, 'OPEN HERE', (W*0.3, H*0.58, W*0.7, H*0.85), 'poppins', ds['text'])
    return paper(img, rng)

def plain(color, W, H, rng):
    return paper(Image.new('RGB', (W, H), color), rng)
