import math, numpy as np
from PIL import Image, ImageDraw, ImageFilter
from textures import fit_text, grad, sunburst, stripes, paper, font, plain

def flower(img, cx, cy, R, petal=(255, 255, 255), center=(250, 200, 40)):
    d = ImageDraw.Draw(img)
    for k in range(12):
        a = 2*math.pi*k/12; x, y = cx+R*0.6*math.cos(a), cy+R*0.6*math.sin(a)
        d.ellipse([x-R*0.42, y-R*0.2, x+R*0.42, y+R*0.2], fill=petal, outline=(220, 220, 210))
    d.ellipse([cx-R*0.35, cy-R*0.35, cx+R*0.35, cy+R*0.35], fill=center, outline=(200, 150, 20), width=3)

def tea_front(W, H, rng):
    img = grad(W, H, (225, 240, 200), (140, 190, 110)).convert('RGBA'); d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, H*0.18], fill=(60, 110, 50))
    fit_text(d, 'MEADOW LEAF', (W*0.05, H*0.03, W*0.6, H*0.15), 'poppins', (255, 255, 255))
    fit_text(d, '20 TEA BAGS', (W*0.62, H*0.04, W*0.95, H*0.14), 'cond', (255, 240, 180))
    fit_text(d, 'Chamomile', (W*0.05, H*0.22, W*0.62, H*0.5), 'caladea', (50, 90, 40))
    fit_text(d, 'Herbal Tea', (W*0.05, H*0.5, W*0.62, H*0.68), 'poppins', (90, 120, 50))
    fit_text(d, 'Caffeine free  |  Net wt 1.0 oz (28g)', (W*0.05, H*0.85, W*0.7, H*0.95), 'regular', (40, 70, 30))
    flower(img, W*0.8, H*0.52, H*0.28); flower(img, W*0.68, H*0.75, H*0.16)
    return paper(img, rng)

def tea_side(W, H, rng):
    img = grad(W, H, (200, 225, 175), (150, 195, 120)).convert('RGBA'); d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, H*0.18], fill=(60, 110, 50))
    fit_text(d, 'Chamomile', (W*0.08, H*0.3, W*0.92, H*0.55), 'caladea', (50, 90, 40))
    fit_text(d, 'Steep 5 min', (W*0.1, H*0.6, W*0.9, H*0.75), 'regular', (40, 70, 30))
    return paper(img, rng)

def tissue_front(W, H, rng):
    img = grad(W, H, (215, 225, 245), (170, 190, 230)).convert('RGBA'); d = ImageDraw.Draw(img)
    for k in range(7):  # soft wave pattern
        y = H*(0.15+0.12*k)
        d.arc([-W*0.2, y-H*0.3, W*1.2, y+H*0.3], 200, 340, fill=(240, 245, 255), width=int(H*0.03))
    fit_text(d, 'Softwell', (W*0.06, H*0.12, W*0.55, H*0.5), 'poppins_i', (40, 60, 140), stroke=0.05, stroke_fill=(255, 255, 255))
    fit_text(d, 'FACIAL TISSUES', (W*0.06, H*0.52, W*0.55, H*0.72), 'cond', (40, 60, 140))
    d.rounded_rectangle([W*0.65, H*0.2, W*0.93, H*0.75], radius=int(H*0.08), fill=(255, 255, 255), outline=(40, 60, 140), width=4)
    fit_text(d, '160', (W*0.68, H*0.25, W*0.9, H*0.55), 'black', (40, 60, 140))
    fit_text(d, '2-PLY TISSUES', (W*0.68, H*0.57, W*0.9, H*0.7), 'cond', (40, 60, 140))
    return paper(img, rng)

def tissue_top(W, H, rng):
    img = grad(W, H, (215, 225, 245), (190, 205, 238)).convert('RGBA'); d = ImageDraw.Draw(img)
    d.ellipse([W*0.25, H*0.3, W*0.75, H*0.7], fill=(70, 75, 90))
    ov = Image.new('RGBA', img.size, (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
    od.polygon([(W*0.35, H*0.5), (W*0.45, H*0.18), (W*0.55, H*0.15), (W*0.62, H*0.42), (W*0.58, H*0.62), (W*0.42, H*0.6)], fill=(250, 250, 252))
    img.alpha_composite(ov.filter(ImageFilter.GaussianBlur(2)))
    return paper(img, rng)

def foil_front(W, H, rng):
    img = grad(W, H, (200, 30, 40), (150, 15, 25)).convert('RGBA'); d = ImageDraw.Draw(img)
    # metallic roll illustration on the right
    g = grad(int(W*0.32), int(H*0.5), (235, 238, 242), (150, 155, 165)).convert('RGBA')
    img.alpha_composite(g, (int(W*0.62), int(H*0.25)))
    d.ellipse([W*0.6, H*0.25, W*0.64, H*0.75], fill=(120, 90, 60))
    fit_text(d, 'KITCHEN PRO', (W*0.04, H*0.1, W*0.55, H*0.5), 'black', (255, 255, 255))
    fit_text(d, 'Aluminum Foil  75 sq ft', (W*0.04, H*0.55, W*0.55, H*0.85), 'poppins', (255, 230, 120))
    return paper(img, rng)

def foil_side(W, H, rng):
    img = Image.new('RGBA', (W, H), (175, 25, 35, 255)); d = ImageDraw.Draw(img)
    fit_text(d, 'KITCHEN PRO', (W*0.1, H*0.3, W*0.9, H*0.7), 'cond', (255, 255, 255))
    return paper(img, rng)

def lookalike_tex(kind, rng):
    if kind == 'tea':
        dims = (1.55, 1.0, 0.75)
        return dims, {'front': tea_front(620, 400, rng), 'right': tea_side(300, 400, rng), 'left': tea_side(300, 400, rng),
                      'top': tea_side(620, 300, rng), 'back': plain((160, 200, 130), 300, 200, rng), 'bottom': plain((180, 170, 150), 300, 100, rng)}
    if kind == 'tissue':
        dims = (2.4, 1.0, 1.15)
        return dims, {'front': tissue_front(960, 400, rng), 'right': tissue_front(460, 400, rng), 'left': tissue_front(460, 400, rng),
                      'top': tissue_top(960, 460, rng), 'back': tissue_front(960, 400, rng), 'bottom': plain((180, 170, 150), 300, 100, rng)}
    if kind == 'foil':
        dims = (6.5, 1.0, 1.0)
        return dims, {'front': foil_front(1300, 200, rng), 'right': foil_side(200, 200, rng), 'left': foil_side(200, 200, rng),
                      'top': foil_side(1300, 200, rng), 'back': plain((160, 20, 30), 300, 100, rng), 'bottom': plain((180, 170, 150), 300, 100, rng)}
