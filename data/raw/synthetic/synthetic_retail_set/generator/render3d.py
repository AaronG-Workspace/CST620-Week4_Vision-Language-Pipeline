import numpy as np, cv2, math

def rot_y(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rot_x(a):
    c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def face_defs(w, h, d):
    # name, P(s,t) -> xyz, outward normal
    return {
     'front':  (lambda s, t: np.stack([s*w, (1-t)*h, 0*s], -1), (0, 0, -1)),
     'right':  (lambda s, t: np.stack([w+0*s, (1-t)*h, s*d], -1), (1, 0, 0)),
     'left':   (lambda s, t: np.stack([0*s, (1-t)*h, (1-s)*d], -1), (-1, 0, 0)),
     'back':   (lambda s, t: np.stack([(1-s)*w, (1-t)*h, d+0*s], -1), (0, 0, 1)),
     'top':    (lambda s, t: np.stack([s*w, h+0*s, (1-t)*d], -1), (0, 1, 0)),
     'bottom': (lambda s, t: np.stack([s*w, 0*s, t*d], -1), (0, -1, 0)),
    }

def render_box(dims, tex, yaw_deg, pitch_deg, out_h, disp=None, grid=(34, 44, 10), dist=4.0, ss=2,
               light=(-0.45, 0.75, -0.6), ambient=0.6, diffuse=0.45, ao=0.28, frame=None):
    """dims=(w,h,d) in box units. tex: dict face->RGB uint8 array. disp(P)->delta (N,3).
    Returns rgba (float 0..1), plus the projected footprint polygon (bottom face corners) in output pixels."""
    w, h, d = dims
    nx, ny, nz = grid
    res = {'front': (nx, ny), 'back': (nx, ny), 'right': (nz, ny), 'left': (nz, ny), 'top': (nx, nz), 'bottom': (nx, nz)}
    R = rot_x(math.radians(-pitch_deg)) @ rot_y(math.radians(yaw_deg))
    center = np.array([w/2, h/2, d/2])
    L = np.array(light, float); L /= np.linalg.norm(L)
    tris = []  # (depth, dst(3x2), src(3x2), face, shade)
    texf = {}
    allpts = []
    for name, (P, n_out) in face_defs(w, h, d).items():
        if name not in tex: continue
        gs, gt = res[name]
        s, t = np.meshgrid(np.linspace(0, 1, gs), np.linspace(0, 1, gt))
        V0 = P(s, t).reshape(-1, 3)
        D = disp(V0) if disp is not None else np.zeros_like(V0)
        V = V0 + D
        C = (V - center) @ R.T + np.array([0, 0, dist*h])
        uv = np.stack([C[:, 0]/C[:, 2], -C[:, 1]/C[:, 2]], -1)
        TH, TW = tex[name].shape[:2]
        T = np.stack([s.ravel()*(TW-1), t.ravel()*(TH-1)], -1)
        # winding check on undeformed param derivative
        P0 = P(np.array([0., 1., 0.]), np.array([0., 0., 1.]))
        ng = np.cross(P0[1]-P0[0], P0[2]-P0[0])
        flip = np.dot(ng, n_out) < 0
        # smooth per-vertex shading baked into the texture (bilinear between grid vertices)
        Cg = C.reshape(gt, gs, 3)
        nv = np.cross(np.gradient(Cg, axis=1), np.gradient(Cg, axis=0))
        if flip: nv = -nv
        nv /= np.linalg.norm(nv, axis=-1, keepdims=True) + 1e-12
        sh = ambient + diffuse*np.clip(nv @ L, 0, None)
        dm = np.linalg.norm(D, axis=1).reshape(gt, gs)
        if dm.max() > 1e-9: sh = sh*(1 - ao*dm/max(dm.max(), 0.04*h))
        shimg = cv2.resize(sh.astype(np.float32), (TW, TH), interpolation=cv2.INTER_LINEAR)
        texf[name] = tex[name].astype(np.float32)*shimg[..., None]
        idx = np.arange(gs*gt).reshape(gt, gs)
        for j in range(gt-1):
            for i in range(gs-1):
                a, b, c, e = idx[j, i], idx[j, i+1], idx[j+1, i+1], idx[j+1, i]
                for tri in ((a, b, c), (a, c, e)):
                    if flip: tri = (tri[0], tri[2], tri[1])
                    p = C[list(tri)]
                    n = np.cross(p[1]-p[0], p[2]-p[0]); nn = np.linalg.norm(n)
                    if nn < 1e-12: continue
                    n = n/nn
                    if np.dot(n, p.mean(0)) >= 0: continue  # back-facing
                    tris.append((p[:, 2].mean(), uv[list(tri)], T[list(tri)], name, 1.0))
                    allpts.append(uv[list(tri)])
    allpts = np.concatenate(allpts)
    if frame is None:
        mn, mx = allpts.min(0), allpts.max(0)
        scale = (out_h*ss) / (mx[1]-mn[1])
        frame = (mn, mx, scale)
    mn, mx, scale = frame
    pad = 6*ss
    Wc = int((mx[0]-mn[0])*scale) + 2*pad; Hc = int((mx[1]-mn[1])*scale) + 2*pad
    img = np.zeros((Hc, Wc, 3), np.float32); alpha = np.zeros((Hc, Wc), np.float32)
    tris.sort(key=lambda z: -z[0])
    for depth, dst, src, name, shade in tris:
        dst = (dst - mn)*scale + pad
        cen = dst.mean(0); dst = cen + (dst-cen)*1.0 + np.sign(dst-cen)*0.6  # slight grow to hide seams
        x, y, ww, hh = cv2.boundingRect(dst.astype(np.float32))
        sx, sy, sw, sh = cv2.boundingRect(src.astype(np.float32))
        x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x+ww, Wc), min(y+hh, Hc)
        if x1 <= x0 or y1 <= y0: continue
        T = texf[name]
        sx0, sy0 = max(sx-1, 0), max(sy-1, 0); sx1, sy1 = min(sx+sw+1, T.shape[1]), min(sy+sh+1, T.shape[0])
        crop = T[sy0:sy1, sx0:sx1]
        M = cv2.getAffineTransform((src - [sx0, sy0]).astype(np.float32), (dst - [x0, y0]).astype(np.float32))
        warped = cv2.warpAffine(crop, M, (x1-x0, y1-y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        m = np.zeros((y1-y0, x1-x0), np.uint8)
        cv2.fillConvexPoly(m, np.round(dst - [x0, y0]).astype(np.int32), 1)
        m = m.astype(bool)
        img[y0:y1, x0:x1][m] = warped[m]*shade
        alpha[y0:y1, x0:x1][m] = 1.0
    out_w = max(1, round(Wc/ss)); out_hh = max(1, round(Hc/ss))
    img = cv2.resize(img, (out_w, out_hh), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (out_w, out_hh), interpolation=cv2.INTER_AREA)
    # footprint: bottom face corners projected
    foot = np.array([[0, 0, 0], [w, 0, 0], [w, 0, d], [0, 0, d]], float)
    C = (foot - center) @ R.T + np.array([0, 0, dist*h])
    fuv = np.stack([C[:, 0]/C[:, 2], -C[:, 1]/C[:, 2]], -1)
    fuv = ((fuv - mn)*scale + pad)/ss
    return np.clip(img/255.0, 0, 1), alpha, fuv, frame
