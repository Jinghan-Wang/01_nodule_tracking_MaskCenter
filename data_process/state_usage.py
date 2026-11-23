"""离线标注状态用途；只划分清单，不移动或删除原始数据。"""
import csv
import json
from pathlib import Path


def build_state_usage(preparation, radius_mm):
    preparation = Path(preparation)
    manifest = json.loads((preparation/'dataset.json').read_text(encoding='utf-8'))
    rows = []
    for split, patients in manifest['patient_split'].items():
        for patient in patients:
            labels = json.loads((preparation/'patients'/patient/'projected_centers.json').read_text(encoding='utf-8'))
            plan = labels['states']['original']['center_lps_mm']
            for phase in manifest['patients'][patient]['phases']:
                delta = [x-y for x, y in zip(labels['states'][phase]['center_lps_mm'], plan)]
                inside = max(map(abs, delta)) <= radius_mm+1e-6
                rows.append(dict(patient=patient, split=split, phase=phase,
                                 usage='main' if inside else 'out_of_range_test',
                                 dx_mm=delta[0], dy_mm=delta[1], dz_mm=delta[2]))
    return rows


def main_phases(rows, patients):
    """同一phase的kv1/kv2一起入选；患者集合由已有患者划分决定。"""
    return {p:[r['phase'] for r in rows if r['patient']==p and r['usage']=='main'] for p in patients}


def write_state_usage(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
