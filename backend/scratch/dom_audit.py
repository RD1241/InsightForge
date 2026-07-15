import re

with open('frontend/app.js', 'r', encoding='utf-8') as f:
    js = f.read()
with open('frontend/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

js_ids = set(re.findall(r'getElementById\("([^"]+)"\)', js))
html_ids = set(re.findall(r'id="([^"]+)"', html))

missing = sorted(js_ids - html_ids)
print(f'IDs in app.js NOT found in index.html ({len(missing)}):')
for m in missing:
    print(f'  MISSING: {m}')
print(f'\nTotal JS refs: {len(js_ids)}, HTML ids: {len(html_ids)}')
