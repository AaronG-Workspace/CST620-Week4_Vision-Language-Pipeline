import json, shutil, os, numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
OI = '/home/claude/oi'; OUT = '/home/claude/out/synthetic_retail_set'
meta = pd.concat([pd.read_csv(f'{OI}/val_meta.csv'), pd.read_csv(f'{OI}/test_meta.csv')]).set_index('ImageID')
pick = json.load(open(f'{OI}/pick2.json'))
syn = pd.read_csv(f'{OUT}/_synthetic_rows.csv')

# measured damage severity
def pct_changed(n):
    a = np.asarray(Image.open(f'{OUT}/images/{n}_intact.jpg')).astype(int); b = np.asarray(Image.open(f'{OUT}/images/{n}_dented.jpg')).astype(int)
    m = np.asarray(Image.open(f'{OUT}/masks/{n}_intact_mask.png')) > 0
    return round(100*(np.abs(a-b).sum(2)[m] > 30).mean(), 2)
syn['pct_box_changed'] = np.nan; syn['damage_severity'] = pd.Series([None]*len(syn), dtype=object)
for i, r in syn[syn.category == 'cereal_dented'].iterrows():
    p = pct_changed(r.file.replace('images/', '').replace('_dented.jpg', ''))
    syn.at[i, 'pct_box_changed'] = p
    syn.at[i, 'damage_severity'] = 'subtle' if p < 6 else ('moderate' if p < 10 else 'severe')
syn['source'] = 'synthetic_render_on_openimages_photo'
syn = syn.rename(columns={'background_openimages_id': 'openimages_id'})

rows = []; attr = []
def add_attr(i, used):
    m = meta.loc[i]
    attr.append(dict(openimages_id=i, used_as=used, author=m.Author, author_profile=m.AuthorProfileURL, title=m.Title,
                     license=m.License, original_landing_url=m.OriginalLandingURL, original_url=m.OriginalURL))
for r in syn.itertuples():
    pass
for i in syn.openimages_id.unique():
    add_attr(i, 'background for ' + ', '.join(syn[syn.openimages_id == i].file.str.replace('images/', '')))

# real distractors
names = ['medicine_box', 'cardboard_box', 'sports_drink_bottle', 'spray_bottle', 'tin_can', 'tin_can_peas']
labels = ['box', 'box', 'bottle', 'bottle', 'tin can', 'tin can']
sel = pd.read_csv(f'{OI}/sel_dist.csv').iloc[:6]
for k, (r, nm, lb) in enumerate(zip(sel.itertuples(), names, labels), 1):
    fn = f'distractor_oi_{k:02d}_{nm}.jpg'; shutil.copy(f'{OI}/img/{r.ImageID}.jpg', f'{OUT}/images/{fn}')
    W, H = Image.open(f'{OUT}/images/{fn}').size
    rows.append(dict(file=f'images/{fn}', category='distractor', pair_id='', product=nm, source='openimages_photo', openimages_id=r.ImageID,
                     width=W, height=H, gt_label=lb, gt_x1=round(r.XMin*W), gt_y1=round(r.YMin*H), gt_x2=round(r.XMax*W), gt_y2=round(r.YMax*H),
                     mask_file='', frame_margin=round(r.margin, 3)))
    add_attr(r.ImageID, fn)
# clean backgrounds
for k, i in enumerate(pick['negs'], 1):
    fn = f'background_oi_{k:02d}.jpg'; shutil.copy(f'{OI}/img/{i}.jpg', f'{OUT}/images/{fn}')
    W, H = Image.open(f'{OUT}/images/{fn}').size
    rows.append(dict(file=f'images/{fn}', category='background', pair_id='', product='', source='openimages_photo', openimages_id=i,
                     width=W, height=H, gt_label='none'))
    add_attr(i, fn)

man = pd.concat([syn, pd.DataFrame(rows)], ignore_index=True)
cols = ['file', 'category', 'pair_id', 'product', 'gt_label', 'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2', 'mask_file', 'damage_types', 'damage_severity',
        'pct_box_changed', 'source', 'openimages_id', 'width', 'height', 'frame_margin', 'yaw_deg', 'pitch_deg']
man = man[cols]
for c in ['gt_x1', 'gt_y1', 'gt_x2', 'gt_y2', 'yaw_deg', 'pitch_deg']: man[c] = man[c].astype('Int64')
order = {'cereal_intact': 0, 'cereal_dented': 1, 'distractor': 2, 'background': 3}
man = man.sort_values(['category', 'file'], key=lambda s: s.map(order) if s.name == 'category' else s).reset_index(drop=True)
man.to_csv(f'{OUT}/manifest.csv', index=False)
pd.DataFrame(attr).to_csv(f'{OUT}/ATTRIBUTION.csv', index=False)
os.remove(f'{OUT}/_synthetic_rows.csv')
print(man.category.value_counts().to_string()); print('attribution rows', len(attr))
print(man[man.category == 'cereal_dented'][['product', 'damage_types', 'damage_severity', 'pct_box_changed']].to_string(index=False))

# contact sheet with GT boxes
f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
tw, th = 300, 240; cols_n = 6; n = len(man); rows_n = (n+cols_n-1)//cols_n
S = Image.new('RGB', (cols_n*tw, rows_n*th), 'white'); d = ImageDraw.Draw(S)
for k, r in man.iterrows():
    im = Image.open(f'{OUT}/{r.file}').convert('RGB'); W, H = im.size; dd = ImageDraw.Draw(im)
    if pd.notna(r.gt_x1): dd.rectangle([r.gt_x1, r.gt_y1, r.gt_x2, r.gt_y2], outline=(255, 0, 0), width=4)
    im.thumbnail((tw-6, th-22)); x = (k % cols_n)*tw; y = (k//cols_n)*th
    S.paste(im, (x+3, y+20)); d.text((x+3, y+3), os.path.basename(r.file)[:40], fill='black', font=f)
S.save(f'{OUT}/contact_sheet.jpg', quality=85)
