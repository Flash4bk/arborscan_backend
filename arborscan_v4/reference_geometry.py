"""Versioned photo-plane measurements, not field validation or DBH certification."""
import math


def reference_geometry(ref):
    w, h = ref['width'], ref['height']
    def points(key):
        line = ref.get(key, [])
        if not isinstance(line, list) or len(line) not in (0, 2):
            raise ValueError('invalid segment')
        for p in line:
            if not isinstance(p, dict) or set(p) != {'x', 'y'} or any(
                type(p[k]) not in (int, float) or not math.isfinite(p[k]) or not 0 <= p[k] <= 1 for k in ('x', 'y')):
                raise ValueError('invalid point')
        return line
    def vector(line):
        return ((line[1]['x']-line[0]['x'])*w, (line[1]['y']-line[0]['y'])*h)
    def dot(a,b): return a[0]*b[0]+a[1]*b[1]
    def cross(a,b): return a[0]*b[1]-a[1]*b[0]
    rv=vector(ref['reference']); length=math.hypot(*rv)
    scale=ref['length_m']/length
    vertical=(rv[0]/length,rv[1]/length)
    def projection(line): return abs(dot(vector(line),vertical))*scale
    lines={key: points(key) for key in ('crown_height','trunk','trunk_axis')}
    for line in lines.values():
        if line and math.hypot(*vector(line)) <= 0: raise ValueError('zero segment')
    axis=vector(ref['tree']); signed_height=dot(axis,vertical)
    sign=1 if signed_height>0 else -1
    def level(point): return dot(vector([ref['tree'][0],point]),vertical)*sign*scale
    for key in ('crown_height','trunk'):
        for point in lines[key]:
            if not -1e-8 <= level(point) <= abs(signed_height)*scale+1e-8:
                raise ValueError('section or crown outside tree height')
    ch=projection(lines['crown_height']) if lines['crown_height'] else None
    diameter=lean=section_level=None
    if lines['trunk_axis']:
        axis=vector(lines['trunk_axis'])
        lean=math.degrees(math.atan2(abs(cross(axis,vertical)),abs(dot(axis,vertical))))
        if lines['trunk']: diameter=abs(cross(vector(lines['trunk']),axis))/math.hypot(*axis)*scale
    if lines['trunk']:
        a,b=lines['trunk']; mid={k:(a[k]+b[k])/2 for k in ('x','y')}
        section_level=level(mid)
    if ch is not None and ch<=0 or diameter is not None and diameter<=0:
        raise ValueError('zero projection')
    def metric(value,unit,definition,reason=None):
        if value is not None and not math.isfinite(value): raise ValueError('nonfinite result')
        return dict(value=value,unit=unit,method='known_object_segment_v2',source='reference',definition=definition,
            reason=reason if value is None else None,
            limitations=['2D projection; weak perspective; same depth','field accuracy unvalidated'])
    return {
      'tree_height':metric(projection(ref['tree']),'m','vertical projection along reference'),
      'tree_segment_length':metric(math.hypot(*vector(ref['tree']))*scale,'m','base-top straight segment in photo plane; not trunk path'),
      'crown_width':metric(abs(cross(vector(ref['crown']),vertical))*scale,'m','marked crown span perpendicular to reference'),
      'crown_height':metric(ch,'m','marked live crown base to top, along reference','Mark live crown base and top'),
      'trunk_diameter':metric(diameter,'m','marked stem width perpendicular to local axis; circular-section assumption','Mark stem edges and local stem axis'),
      'trunk_measurement_height':metric(section_level,'m','marked section midpoint above marked tree base in projection; not forestry DBH rule','Mark stem edges'),
      'trunk_lean':metric(lean,'deg','acute 2D angle of marked local axis to reference; curvature and out-of-plane lean unresolved','Mark local stem axis'),
      'dbh':metric(None,'m','DBH requires field measurement position protocol','1.3 m alone does not establish DBH: base, slope, forks and lean require field verification'),
      'crown_porosity':metric(None,'1','crown gap fraction','No validated crown region and gap-preserving mask; filled polygon is insufficient'),
    }
