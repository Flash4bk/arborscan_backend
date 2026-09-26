"""Checks actual PDFs with pypdf/pdfplumber; visual page review remains a separate step."""
import hashlib, json, pathlib, re
import pypdf
import pdfplumber

root=pathlib.Path('output/pdf/as12')
checks=[]
for p in sorted(root.glob('**/*.pdf')):
    reader=pypdf.PdfReader(p)
    text='\n'.join(page.extract_text() for page in reader.pages)
    assert 'ArborScan' in text and 'DBH' in text and 'β' in text, p
    assert not re.search(r'\b(?:NaN|Infinity|Bearer)\b|https?://',text),p
    for page in reader.pages:
        assert abs(float(page.mediabox.width)-595.276)<1 and abs(float(page.mediabox.height)-841.89)<1,p
        for f in page['/Resources']['/Font'].get_object().values():
            f=f.get_object()
            if '/DescendantFonts' in f: f=f['/DescendantFonts'][0].get_object()
            desc=f['/FontDescriptor'].get_object()
            assert any(k in desc for k in ['/FontFile','/FontFile2','/FontFile3']),p
    if 'full-v1' in p.name: assert '4.8 м' in text and '5.4 м' not in text,p
    if 'full-v2' in p.name: assert '5.4 м' in text and '4.8 м' not in text,p
    if 'missing' in p.name: assert 'НЕПОЛНЫЙ' in text,p
    words=0
    with pdfplumber.open(p) as doc:
        for page in doc.pages:
            for word in page.extract_words():
                words+=1
                assert 0<=word['x0']<=word['x1']<=page.width,p
                assert 0<=word['top']<=word['bottom']<=page.height,p
    checks.append({'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'pages':len(reader.pages),'words':words,
        'cyrillic_embedded_font':True,'page_bounds':True,'text_checks':True})
(root/'pdf-audit.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'{len(checks)} PDFs passed text, version, fonts and page bounds checks')
