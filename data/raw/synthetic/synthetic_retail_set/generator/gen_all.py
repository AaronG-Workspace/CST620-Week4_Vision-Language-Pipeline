import json, math, random, csv, os, sys
import numpy as np, cv2, pandas as pd
from PIL import Image
from textures import front_cereal, side_panel, top_panel, plain
from render3d import render_box
from damage import crease, corner_crush, edge_dent, top_edge_crush, combine, crack_line, scuffs
from designs import DESIGNS
from lookalikes import lookalike_tex

OI = '/home/claude/oi'; OUT = '/home/claude/out/synthetic_retail_set'
for sub in ('images', 'masks', 'previews'): os.makedirs(f'{OUT}/{sub}', exist_ok=True)
pick = json.load(open(f'{OI}/pick2.json'))
B = pick['bases']
GRID = (48, 64, 12)

def cereal_tex(ds, rng, W=600):
    H = int(W/ds['dims'][0]); D = max(int(W*ds['dims'][2]/ds['dims'][0]), 120)
    return {'front': front_cereal(ds, W, H, rng), 'right': side_panel(ds, D, H, rng), 'left': side_panel(ds, D, H, rng),
            'top': top_panel(ds, W, D, rng), 'back': plain(ds['bg2'], 300, 400, rng), 'bottom': plain((180, 170, 150), 300, 100, rng)}

# placement: base index, footprint centre (x, y) as fraction of image, projected height fraction, yaw, pitch
PLACE = {
 'sunny_oat_crunch':       (0, 0.42, 0.40, 0.33, 25, 32),
 'honey_halos':            (1, 0.50, 0.62, 0.40, -22, 35),
 'choco_comets':           (2, 0.27, 0.66, 0.40, 28, 30),
 'berry_bran_flakes':      (3, 0.50, 0.58, 0.42, 15, 30),
 'maple_morning_clusters': (4, 0.62, 0.55, 0.45, -32, 25),
 'cinnamon_swirl_squares': (6, 0.48, 0.50, 0.38, -18, 30),
 'puffed_rice_rockets':    (7, 0.50, 0.68, 0.34, 22, 35),
 'frosty_wheat_pillows':   (8, 0.72, 0.48, 0.38, -28, 30),
 'rainbow_rings':          (10, 0.38, 0.47, 0.30, 30, 22),
 'golden_corn_crisps':     (12, 0.55, 0.275, 0.20, -20, 15),
 'tea':                    (11, 0.35, 0.48, 0.14, -40, 15),
 'tissue':                 (9, 0.55, 0.445, 0.12, 22, 10),
 'foil':                   (5, 0.32, 0.65, 0.16, 30, 28),
}

# damage plans: (type, severity, params) ; coordinates in box units (w, h=1, d)
def damage_plan(key, dims, yaw, rng):
    w, h, d = dims; vis = 'right' if yaw > 0 else 'left'; vx = w if yaw > 0 else 0.0; far = 0.0 if yaw > 0 else w
    plans = {
     'sunny_oat_crunch':       [('front_crease', 'moderate', dict(c=(0.30, 0.58), ang=35, L=0.42, wd=0.07, dp=0.05)),
                                ('corner_crush', 'moderate', dict(K=(vx, h, 0), amt=0.10, rad=0.24))],
     'honey_halos':            [('edge_dent', 'severe', dict(side=vis, y0=0.55, amt=0.10, rad=0.15))],
     'choco_comets':           [('top_edge_crush', 'moderate', dict(x0=0.12*w, x1=0.75*w, amt=0.11, rad=0.17))],
     'berry_bran_flakes':      [('front_crease', 'subtle', dict(c=(0.45, 0.35), ang=-20, L=0.32, wd=0.07, dp=0.042))],
     'maple_morning_clusters': [('front_crease', 'severe', dict(c=(0.36, 0.62), ang=40, L=0.5, wd=0.08, dp=0.065)),
                                ('front_crease', 'severe', dict(c=(0.40, 0.58), ang=-45, L=0.42, wd=0.08, dp=0.06))],
     'cinnamon_swirl_squares': [('corner_crush', 'severe', dict(K=(vx, h, 0), amt=0.17, rad=0.32))],
     'puffed_rice_rockets':    [('front_crease', 'moderate', dict(c=(0.32, 0.30), ang=10, L=0.36, wd=0.08, dp=0.05, dimple=0.9))],
     'frosty_wheat_pillows':   [('edge_dent', 'moderate', dict(side=vis, y0=0.25, amt=0.07, rad=0.14))],
     'rainbow_rings':          [('top_edge_crush', 'severe', dict(x0=0.35*w, x1=1.05*w, amt=0.10, rad=0.15)),
                                ('corner_crush', 'severe', dict(K=(vx, h, 0), amt=0.12, rad=0.25))],
     'golden_corn_crisps':     [('front_crease', 'moderate', dict(c=(0.36, 0.60), ang=-30, L=0.42, wd=0.08, dp=0.06, dimple=0.7))],
    }
    return plans[key]

def build_damage(plan, dims, tex, rng):
    w, h, d = dims; TW, TH = tex['front'].size
    def ft(x, y): return (x/w*TW, (1-y/h)*TH)
    fs = []
    for typ, sev, p in plan:
        if typ == 'front_crease':
            fs.append(crease(dims, p['c'], p['ang'], p['L'], p['wd'], p['dp'], dimple=p.get('dimple', 0.5)))
            u = (math.cos(math.radians(p['ang'])), math.sin(math.radians(p['ang'])))
            a = ft(p['c'][0]-u[0]*p['L']*0.4, p['c'][1]-u[1]*p['L']*0.4); b = ft(p['c'][0]+u[0]*p['L']*0.4, p['c'][1]+u[1]*p['L']*0.4)
            crack_line(tex['front'], a, b, rng, width=3 if sev != 'subtle' else 2, alpha=180 if sev != 'subtle' else 140)
        elif typ == 'corner_crush':
            K = p['K']; fs.append(corner_crush(dims, K, p['amt'], p['rad']))
            cx, cy = ft(K[0], K[1]); scuffs(tex['front'], cx, cy, p['rad']/h*TH*0.6, rng, n=55)
            side = 'right' if K[0] > w/2 else 'left'; SW, SH = tex[side].size
            scuffs(tex[side], (0 if side == 'right' else SW), 0, SW*0.5, rng, n=30)
            TW2, TH2 = tex['top'].size; scuffs(tex['top'], (TW2 if side == 'right' else 0), TH2, TH2*0.6, rng, n=30)
        elif typ == 'edge_dent':
            fs.append(edge_dent(dims, p['side'], p['y0'], p['amt'], p['rad']))
            ex = w if p['side'] == 'right' else 0.0; inset = -1 if p['side'] == 'right' else 1
            a = ft(ex + inset*0.02, p['y0']+p['rad']*1.2); b = ft(ex + inset*0.02, p['y0']-p['rad']*1.2)
            crack_line(tex['front'], a, b, rng, width=2, alpha=150 if sev != 'subtle' else 90, jag=5)
            cx, cy = ft(ex, p['y0']); scuffs(tex['front'], cx, cy, p['rad']/h*TH*0.5, rng, n=25 if sev != 'subtle' else 10)
        elif typ == 'top_edge_crush':
            fs.append(top_edge_crush(dims, p['x0'], p['x1'], p['amt'], p['rad']))
            a = ft(p['x0'], h-0.03); b = ft(min(p['x1'], w), h-0.03)
            crack_line(tex['front'], a, b, rng, width=3, alpha=170, jag=6)
            for xx in np.linspace(p['x0'], min(p['x1'], w), 4): 
                cx, cy = ft(xx, h); scuffs(tex['front'], cx, cy, TH*0.05, rng, n=12)
    return combine(*fs)

def composite(bg, rgb, alpha, foot, fx, fy, seed):
    rs = np.random.default_rng(seed)
    H, W = bg.shape[:2]; bh, bw = alpha.shape
    cen = foot.mean(0); ox = int(round(fx*W - cen[0])); oy = int(round(fy*H - cen[1]))
    # local lighting match from the area around the box
    ys, xs = np.nonzero(alpha > 0.5)
    y0, y1 = max(oy+ys.min()-bh//2, 0), min(oy+ys.max()+bh//2, H); x0, x1 = max(ox+xs.min()-bw//2, 0), min(ox+xs.max()+bw//2, W)
    region = bg[y0:y1, x0:x1].reshape(-1, 3).mean(0)
    lum = float(region @ [0.299, 0.587, 0.114])
    gain = float(np.clip((lum/0.5)**0.3, 0.72, 1.08))
    tint = (region/max(region.mean(), 1e-3))**0.18
    box = np.clip(rgb*gain*tint[None, None], 0, 1)
    box = cv2.GaussianBlur(box, (0, 0), 0.7)
    box = np.clip(box + rs.normal(0, 0.012, box.shape), 0, 1)
    al = cv2.GaussianBlur(alpha, (0, 0), 0.5)
    # soft attached shadow (no hard separate shadow)
    out = bg.copy()
    sh = np.zeros((H, W), np.float32); bhpx = bh
    fp = (foot + [ox, oy]); c = fp.mean(0)
    big = ((fp - c)*1.18 + c + [0.04*bw, -0.01*bh]).astype(np.int32)
    cv2.fillConvexPoly(sh, big, 1.0); sh = cv2.GaussianBlur(sh, (0, 0), max(2, 0.045*bhpx))
    contact = np.zeros((H, W), np.float32); cv2.fillConvexPoly(contact, fp.astype(np.int32), 1.0)
    contact = cv2.GaussianBlur(contact, (0, 0), max(1.5, 0.012*bhpx))
    out *= (1 - 0.38*sh - 0.30*contact)[..., None]
    # paste box
    ya, yb = max(oy, 0), min(oy+bh, H); xa, xb = max(ox, 0), min(ox+bw, W)
    sa = al[ya-oy:yb-oy, xa-ox:xb-ox][..., None]
    out[ya:yb, xa:xb] = box[ya-oy:yb-oy, xa-ox:xb-ox]*sa + out[ya:yb, xa:xb]*(1-sa)
    mask = np.zeros((H, W), np.uint8); mask[ya:yb, xa:xb] = (al[ya-oy:yb-oy, xa-ox:xb-ox] >= 0.5)*255
    return np.clip(out, 0, 1), mask

def bbox_of(mask):
    ys, xs = np.nonzero(mask); return int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1

def margin(b, W, H): return min(b[0]/W, b[1]/H, (W-b[2])/W, (H-b[3])/H)

def save_jpg(arr, path): Image.fromarray((arr*255).round().astype(np.uint8)).save(path, quality=92, subsampling=0)

def load_bg(i): return np.asarray(Image.open(f'{OI}/img/{i}.jpg').convert('RGB')).astype(np.float32)/255

rows = []
only = sys.argv[1:]
for k, ds in enumerate(DESIGNS, 1):
    if only and ds['key'] not in only: continue
    bi, fx, fy, hr, yaw, pitch = PLACE[ds['key']]
    bg = load_bg(B[bi]); H, W = bg.shape[:2]
    rng = random.Random(1000+k)
    tex = cereal_tex(ds, rng)
    texA = {a: np.asarray(b.convert('RGB')) for a, b in tex.items()}
    rgb, al, foot, frame = render_box(ds['dims'], texA, yaw, pitch, hr*H, grid=GRID)
    imgA, mA = composite(bg, rgb, al, foot, fx, fy, seed=k)
    plan = damage_plan(ds['key'], ds['dims'], yaw, rng)
    dtex = {a: b.convert('RGBA') for a, b in tex.items()}
    disp = build_damage(plan, ds['dims'], dtex, random.Random(2000+k))
    texD = {a: np.asarray(b.convert('RGB')) for a, b in dtex.items()}
    rgbD, alD, footD, _ = render_box(ds['dims'], texD, yaw, pitch, hr*H, grid=GRID, disp=disp, frame=frame)
    imgD, mD = composite(bg, rgbD, alD, footD, fx, fy, seed=k)
    for state, img, m in (('intact', imgA, mA), ('dented', imgD, mD)):
        name = f'cereal_{k:02d}_{ds["key"]}_{state}'
        save_jpg(img, f'{OUT}/images/{name}.jpg'); Image.fromarray(m).save(f'{OUT}/masks/{name}_mask.png')
        b = bbox_of(m)
        rows.append(dict(file=f'images/{name}.jpg', category=f'cereal_{state}', pair_id=f'cereal_{k:02d}', product=ds['key'],
                         source='synthetic_composite', background_openimages_id=B[bi], width=W, height=H,
                         gt_label='cereal box', gt_x1=b[0], gt_y1=b[1], gt_x2=b[2], gt_y2=b[3], mask_file=f'masks/{name}_mask.png',
                         damage_types='' if state == 'intact' else '+'.join(p[0] for p in plan),
                         damage_severity='' if state == 'intact' else max((p[1] for p in plan), key=['subtle', 'moderate', 'severe'].index),
                         yaw_deg=yaw, pitch_deg=pitch, frame_margin=round(margin(b, W, H), 3)))
        print(name, b, 'margin', round(margin(b, W, H), 3), flush=True)

for j, kind in enumerate(['tea', 'tissue', 'foil'], 1):
    if only and kind not in only: continue
    bi, fx, fy, hr, yaw, pitch = PLACE[kind]
    bg = load_bg(B[bi]); H, W = bg.shape[:2]
    rng = random.Random(3000+j); dims, t = lookalike_tex(kind, rng)
    tex = {a: np.asarray(b.convert('RGB')) for a, b in t.items()}
    rgb, al, foot, _ = render_box(dims, tex, yaw, pitch, hr*H, grid=(40, 20, 16) if kind != 'foil' else (80, 10, 10))
    img, m = composite(bg, rgb, al, foot, fx, fy, seed=100+j)
    labels = {'tea': 'tea box', 'tissue': 'tissue box', 'foil': 'aluminum foil box'}
    name = f'distractor_syn_{j:02d}_{kind}_box'
    save_jpg(img, f'{OUT}/images/{name}.jpg'); Image.fromarray(m).save(f'{OUT}/masks/{name}_mask.png')
    b = bbox_of(m)
    rows.append(dict(file=f'images/{name}.jpg', category='distractor', pair_id='', product=f'{kind}_box', source='synthetic_composite',
                     background_openimages_id=B[bi], width=W, height=H, gt_label=labels[kind], gt_x1=b[0], gt_y1=b[1], gt_x2=b[2], gt_y2=b[3],
                     mask_file=f'masks/{name}_mask.png', damage_types='', damage_severity='', yaw_deg=yaw, pitch_deg=pitch, frame_margin=round(margin(b, W, H), 3)))
    print(name, b, 'margin', round(margin(b, W, H), 3), flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/_synthetic_rows{"_partial" if only else ""}.csv', index=False)
