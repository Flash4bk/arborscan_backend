"""Verify synthetic PDFs saved through Samsung's system document picker."""
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader
from PIL import ImageChops, ImageStat

root = Path('output/as14')
versions = ['41821299-2bac-41fc-aa84-c64a9bebc061',
            '84bb70d2-aef8-49b8-bf03-3207e06a21de']
results = []
for index, version in enumerate(versions, 1):
    path = root / f's24-v{index}.pdf'
    pdf = PdfReader(path)
    reference = PdfReader(root / f'server-v{index}.pdf')
    text = '\n'.join(page.extract_text() for page in pdf.pages)
    assert len(pdf.pages) == 4
    assert version in text
    assert 'a190464d-85c3-48bf-bcb4-9b5940e5ace5' in text
    assert ('4.8 м' if index == 1 else '9.6 м') in text
    assert ('2.1 м' if index == 1 else '4.2 м') in text
    assert 'normalized_oriented_image' in text
    # Compare both photo and rendered annotation with the independently checked
    # same synthetic snapshot from the emulator, excluding IDs/timestamps.
    for page_index in (0, 2):
        actual = pdf.pages[page_index].images[0].image.convert('RGB')
        expected = reference.pages[page_index].images[0].image.convert('RGB')
        assert actual.size == expected.size
        assert max(ImageStat.Stat(ImageChops.difference(actual, expected)).rms) < 1
    results.append({'file': path.name, 'pages': 4, 'version': version,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'photo_and_annotation_match': True})
old_path = root / 's24-old.pdf'
old_pdf = PdfReader(old_path)
old_text = '\n'.join(page.extract_text() for page in old_pdf.pages)
assert len(old_pdf.pages) == 3
assert 'Локальный снимок' in old_text and 'Версия: old' in old_text
assert 'Разметка отсутствует' in old_text and 'нет данных' in old_text
results.append({'file': old_path.name, 'pages': 3,
                'sha256': hashlib.sha256(old_path.read_bytes()).hexdigest(),
                'legacy_missing_fields_explicit': True})
(root / 's24-pdf-checks.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print('PASS: both S24 server PDFs and old local PDF; 11 pages')
