"""Assign native clone confidence to exact target AST units."""

def score_nicad(xml_path, units):
    import xml.etree.ElementTree as ET
    scores = {u['id']: None for u in units}
    for clone in ET.parse(xml_path).getroot().iter('clone'):
        spans = clone.findall('source')
        if not any(s.attrib['file'].startswith('source/') for s in spans):
            continue
        value = float(clone.attrib['similarity']) / 100
        for span in spans:
            name = span.attrib['file']
            if not name.startswith('target/'):
                continue
            start, end = int(span.attrib['startline']), int(span.attrib['endline'])
            for u in units:
                if u['path'] == name[7:] and any(start <= line <= end for line in u['owned_lines']):
                    scores[u['id']] = max(scores[u['id']] or 0, value)
    return scores
