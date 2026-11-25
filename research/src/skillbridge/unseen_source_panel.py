"""Conservative image-ID exclusion and prespecified score-blind sampling."""
from pathlib import Path
import random
import re


def appledouble_metadata(name,header):
    return name.startswith('._') and header==b'\x00\x05\x16\x07\x00\x02\x00\x00'


def single_spatial_relation(row):
    relations=[step for step in row.get('semantic',[]) if 'rel' in step.get('operation','')]
    return (row.get('answer') in ('yes','no') and len(relations)==1
            and any(t in relations[0].get('argument','')
                    for t in ('to the left of','to the right of','above','below')))


def image_identity(value):
    name=Path(value).name
    match=re.fullmatch(r'(?:COCO_[^_]+_)?(\d{12})\.(?:jpg|png|jpeg)',name,re.I)
    if match: return 'coco:'+str(int(match[1]))
    return 'other:'+name


def collect_images(value):
    if isinstance(value,str):
        return {image_identity(value)} if value.lower().endswith(('.jpg','.png','.jpeg')) else set()
    if isinstance(value,dict): values=value.values()
    elif isinstance(value,list): values=value
    else: return set()
    result=set()
    for item in values: result.update(collect_images(item))
    return result


def select_panel(rows,excluded,*,labels,per_label):
    if per_label<1 or len(set(labels))!=len(labels): raise ValueError('invalid fixed panel budget')
    ordered=sorted(rows,key=lambda r:r['split_key'])
    random.Random(20260913).shuffle(ordered)
    used=set(excluded); counts={label:0 for label in labels}; selected=[]
    for row in ordered:
        label=row['answer_text']; image=image_identity(row['user_image'])
        if label not in counts or counts[label]>=per_label or image in used: continue
        selected.append(row); used.add(image); counts[label]+=1
    if any(n!=per_label for n in counts.values()):
        raise ValueError(f'insufficient truly unallocated images: {counts}; requested {per_label} each')
    return selected
