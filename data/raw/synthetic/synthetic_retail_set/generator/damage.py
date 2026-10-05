import numpy as np, math
from PIL import Image, ImageDraw, ImageFilter

def _n(v): v = np.asarray(v, float); return v/np.linalg.norm(v)

def crease(dims, c, ang_deg, length, width, depth, dimple=0.5, zfall=0.5):
    w, h, d = dims; c = np.array(c, float)
    u = np.array([math.cos(math.radians(ang_deg)), math.sin(math.radians(ang_deg))]); nperp = np.array([-u[1], u[0]])
    def f(V):
        q = V[:, :2] - c
        a = (q @ u)/(length/2); r = np.abs(q @ nperp)
        taper = np.clip(1 - a**2, 0, 1)**0.7
        prof = depth*taper*np.clip(1 - r/width, 0, 1)
        prof += depth*dimple*np.exp(-(q**2).sum(1)/(2*(width*1.1)**2))
        wz = np.clip(1 - V[:, 2]/(zfall*d), 0, 1)
        out = np.zeros_like(V); out[:, 2] = prof*wz
        return out
    return f

def corner_crush(dims, corner, amount, radius, power=1.2):
    w, h, d = dims; K = np.array(corner, float)
    dirv = _n([(-1 if K[0] > w/2 else 1)*0.35, (-1.0 if K[1] > h/2 else 1.0), 0.5])
    def f(V):
        r = np.linalg.norm(V - K, axis=1)
        m = amount*np.clip(1 - r/radius, 0, 1)**power
        return m[:, None]*dirv[None]
    return f

def edge_dent(dims, side, y0, amount, radius, elong=1.6):
    w, h, d = dims; x = w if side == 'right' else 0.0
    K = np.array([x, y0, 0.0]); dirv = _n([-1 if side == 'right' else 1, 0, 1])
    def f(V):
        q = V - K
        r = np.sqrt((q[:, 0]/radius)**2 + (q[:, 1]/(radius*elong))**2 + (q[:, 2]/radius)**2)
        m = amount*np.clip(1 - r, 0, 1)**1.1
        return m[:, None]*dirv[None]
    return f

def top_edge_crush(dims, x0, x1, amount, radius):
    w, h, d = dims
    def f(V):
        xm = (x0+x1)/2; hw = (x1-x0)/2
        a = np.clip(1 - ((V[:, 0]-xm)/hw)**2, 0, 1)
        r = np.sqrt((h - V[:, 1])**2 + V[:, 2]**2)
        m = amount*a*np.clip(1 - r/radius, 0, 1)**1.2
        out = np.zeros_like(V); out[:, 1] = -m; out[:, 2] = m*0.45
        return out
    return f

def combine(*fs):
    def f(V):
        out = np.zeros_like(V)
        for g in fs: out = out + g(V)
        return out
    return f

# --- texture wear marks (cracked ink along folds, scuffs) ---
def crack_line(img, p0, p1, rng, width=3, col=(236, 230, 218), alpha=170, jag=4):
    ov = Image.new('RGBA', img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    n = 24; pts = []
    for k in range(n+1):
        t = k/n; x = p0[0]+(p1[0]-p0[0])*t + rng.uniform(-jag, jag); y = p0[1]+(p1[1]-p0[1])*t + rng.uniform(-jag, jag); pts.append((x, y))
    d.line(pts, fill=col+(alpha,), width=width)
    for k in range(0, n, 3):
        if rng.random() < 0.6:
            x, y = pts[k]; L = rng.uniform(4, 12); a = rng.uniform(0, math.pi*2)
            d.line([(x, y), (x+L*math.cos(a), y+L*math.sin(a))], fill=col+(int(alpha*0.7),), width=max(1, width-1))
    img.alpha_composite(ov.filter(ImageFilter.GaussianBlur(0.6)))

def scuffs(img, cx, cy, rad, rng, n=40, col=(225, 215, 195)):
    ov = Image.new('RGBA', img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    for _ in range(n):
        a = rng.uniform(0, 2*math.pi); r = rad*math.sqrt(rng.random())
        x, y = cx + r*math.cos(a), cy + r*math.sin(a); L = rng.uniform(3, rad*0.25); b = a + rng.uniform(-0.6, 0.6)
        d.line([(x, y), (x+L*math.cos(b), y+L*math.sin(b))], fill=col+(rng.randint(90, 200),), width=rng.randint(1, 3))
    img.alpha_composite(ov.filter(ImageFilter.GaussianBlur(0.5)))
