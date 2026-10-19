"""Clean a frozen collection into traceable review tiers and conversation candidates."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from .common import ROOT, read_jsonl, write_json, write_jsonl

NUMBER = r'(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:[.,]\d+)?|\.\d+)'
STAGES = {'mash_or_steep':r'\bmash|\bsteep', 'boil':r'\bboil', 'cooling':r'\bcool|\bchill',
          'fermentation':r'\bferment|\bpitch', 'packaging':r'\bbottl|\bkeg|\bpackag|\bcarbonat'}
HEADINGS = {'malt','malts','fermentables','hops','yeast','water','misc.','misc','additional items','other ingredients'}
SYSTEM = ('You are a friendly brewing teacher. Give clear ingredients and ordered instructions. '
          'Preserve source quantities, temperatures and timings. Explain terms plainly and acknowledge '
          'missing information. Never invent a recipe parameter.')


def text(value):
    value = str(value or '').replace('º','°')
    for char, fraction in [('½',' 1/2'),('¼',' 1/4'),('¾',' 3/4'),('⅓',' 1/3'),('⅔',' 2/3')]:
        value = value.replace(char,fraction)
    value = unicodedata.normalize('NFKC',value).replace('⁄','/')
    return re.sub(r'\s+',' ',value).strip()


def number(value):
    return float(sum((Fraction(x.replace(',','.')) for x in value.split()),Fraction(0)))


def batch_liters(value):
    if isinstance(value,dict):
        unit=str(value.get('unit','')).lower()
        amount=value.get('value')
        if isinstance(amount,(int,float)) and not isinstance(amount,bool) and amount>0 and unit in ['l','litres','liters']:
            return {'value':amount,'unit':'L','basis':'explicit_metric'}
        return None
    value=text(value)
    metric=re.search(rf'({NUMBER})\s*(?:liters?|litres?|l)\b',value,re.I)
    if metric:
        amount=number(metric[1])
        return {'value':amount,'unit':'L','basis':'explicit_metric'} if amount>0 else None
    us=re.search(rf'({NUMBER})\s*US\s*gal',value,re.I)
    if us:
        return {'value':number(us[1])*3.785411784,'unit':'L','basis':'explicit_US_gallon_conversion'}
    # An unqualified gallon can mean different volumes. Preserve the source and require review.
    return None


def mass(line):
    match=re.match(rf'^({NUMBER})\s*(kilograms?|kg|grams?|g|lbs?\.?|pounds?|oz\.?|ounces?)\b',line,re.I)
    if not match: return None
    unit=match[2].lower().rstrip('.')
    factors={'kg':1000,'kilogram':1000,'kilograms':1000,'g':1,'gram':1,'grams':1,
             'lb':453.59237,'lbs':453.59237,'pound':453.59237,'pounds':453.59237,
             'oz':28.349523125,'ounce':28.349523125,'ounces':28.349523125}
    return {'value':number(match[1])*factors[unit],'unit':'g','source_quantity':match[0],
            'basis':'leading_source_mass; oz interpreted as weight, verify liquid ingredients'}


def sections(value):
    result={}
    current=None
    for raw in (value or '').splitlines():
        line=raw.strip()
        if re.fullmatch(r'[A-Z][A-Z _-]+:',line):
            current=line[:-1]
            result.setdefault(current,[])
        elif current and line:
            result[current].append(text(line))
    return result


def repair_ingredients(lines):
    cleaned=[]; repairs=[]; group=None
    for index,raw in enumerate(lines):
        line=text(raw)
        if not line: continue
        if line.casefold().rstrip(':') in HEADINGS:
            group=line.casefold().rstrip(':')
            continue
        # HTML sometimes breaks the alpha-acid/timing suffix into a separate list item.
        if cleaned and re.match(r'^\d+(?:\.\d+)?%\s*a\.?a\.?',line,re.I):
            cleaned[-1]['source_text'] += ' '+line
            cleaned[-1]['source_line_indices'].append(index)
            repairs.append('joined_wrapped_alpha_acid_line')
            continue
        cleaned.append({'source_text':line,'source_line_indices':[index],'group':group,
                        'normalized_mass':mass(line)})
    return cleaned,repairs


def temperature_conflicts(value):
    findings=[]
    num=r'\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?'
    pattern=rf'({num})\s*°?\s*F\s*\(\s*({num})\s*°?\s*C\s*\)'
    for m in re.finditer(pattern,value,re.I):
        f=[float(x) for x in re.split(r'\s*[-–]\s*',m[1])]
        c=[float(x) for x in re.split(r'\s*[-–]\s*',m[2])]
        if len(f)==len(c) and any(abs((a-32)*5/9-b)>1.5 for a,b in zip(f,c)):
            findings.append({'text':m[0],'reason':'Fahrenheit/Celsius mismatch exceeds 1.5 C'})
    return findings


def nonbeer(record,ingredients):
    title=text(record['title']).lower()
    style=text(record.get('style')).lower()
    name=title+' '+style
    # Keep barleywine and wheatwine, which are beer styles.
    stripped=re.sub(r'\b(?:barley\s*wine|wheat\s*wine)\b','',name)
    if re.search(r'\b(?:mead|cider|perry|kombucha|seltzer|wine)\b',stripped): return True
    if re.search(r'\btea\b',title) and not re.search(r'\b(?:ale|beer|stout|porter|ipa|lager)\b',name): return True
    return False


def clean(record):
    source=record['source']
    sec=sections(record.get('process_text')) if source=='brewersfriend' else {}
    if sec:
        original=[]
        for key in ['FERMENTABLES','HOPS','OTHER INGREDIENTS']:
            original.extend(sec.get(key,[]))
        if sec.get('YEAST'): original.append(sec['YEAST'][0])
        lines=original or record['ingredients_text']
    else:
        lines=record['ingredients_text']
    ingredients,repairs=repair_ingredients(lines)
    directions=[text(x) for x in record.get('directions_text',[]) if text(x)]
    notes=text(record.get('notes'))
    if sec.get('NOTES'): notes=' '.join(sec['NOTES'])
    process=text(record.get('process_text'))
    all_text=' '.join([*directions,notes,process,*[x['source_text'] for x in ingredients]])
    stages={key:bool(re.search(pattern,' '.join(directions)+' '+notes,re.I)) for key,pattern in STAGES.items()}
    flags=[]
    if not directions: flags.append('no_written_directions')
    if not ingredients: flags.append('no_ingredients')
    batch=batch_liters(record.get('batch_yield'))
    if batch is None: flags.append('batch_volume_unresolved')
    warnings=re.findall(r'WARNING:[^\r\n]*',record.get('process_text') or '')
    if warnings: flags.append('source_equipment_or_calculation_warning')
    conflicts=temperature_conflicts(all_text)
    if conflicts: flags.append('temperature_conversion_conflict')
    # Known contradictory source instructions observed during this audit.
    if record['source_url'].rstrip('/').endswith('/a-kveiking-we-will-go-ipa'):
        flags.append('contradictory_pitching_temperatures')
    if any(re.search(r'\bYeast\s+(?:White Labs|Wyeast)',x['source_text']) and re.match(NUMBER,x['source_text']) for x in ingredients):
        flags.append('embedded_yeast_text_in_ingredient_line')
    if re.search(r'(?im)^Brew Method:\s*Extract',record.get('process_text') or '') and re.search(r'\b(?:Mash|Infusion)\b',' '.join(directions)):
        flags.append('brew_method_process_requires_review')
    excluded=nonbeer(record,ingredients)
    numeric_temp=bool(re.search(r'\d\s*°?\s*[CF]\b',' '.join(directions),re.I))
    if not numeric_temp: flags.append('no_explicit_temperature_in_directions')
    missing=[key for key,present in stages.items() if not present]
    if excluded: tier='excluded_nonbeer'
    elif conflicts or 'contradictory_pitching_temperatures' in flags: tier='quarantine_conflict'
    elif not missing and not flags: tier='instruction_candidate'
    else: tier='needs_enrichment'
    result={'recipe_id':record['content_id'],'title':text(record['title']),'source':source,
            'source_url':record['source_url'],'sources':record.get('sources',[]),
            'provenance':{key:record.get(key) for key in
                          ['html_sha256','raw_html','fetched_at','rights_status']},
            'batch_yield_original':record.get('batch_yield'),'batch_volume':batch,
            'ingredients':ingredients,'directions':directions,'notes':notes,
            'source_sections':sec,'original_process_text':record.get('process_text'),
            'instruction_origin':record.get('instruction_origin'),
            'audit':{'tier':tier,'missing_stages':missing,'flags':flags,'repairs':repairs,
                     'temperature_conflicts':conflicts,'source_warnings':warnings,
                     'expert_reviewed':False},'family_id':None}
    identity={key:result[key] for key in ['batch_volume','ingredients','directions','notes','original_process_text']}
    result['clean_content_hash']=hashlib.sha256(json.dumps(identity,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    return result


def assign_families(records):
    parent={r['recipe_id']:r['recipe_id'] for r in records}
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for r in records:
        name=re.sub(r'\([^)]*\)|\b(?:v\d+|20\d\d|clone|all.grain|extract)\b',' ',r['title'].casefold())
        name=re.sub(r'[^a-z0-9]+',' ',name).strip()
        keys=[('content',r['clean_content_hash'])]
        if name: keys.append(('name',name))
        # Same quantity-bearing ingredients join a family; distinct processes remain separate records.
        signature=tuple(sorted(x['source_text'].casefold() for x in r['ingredients']))
        if signature: keys.append(('ingredients',signature))
        for key in keys:
            if key in seen:
                a,b=root(r['recipe_id']),root(seen[key]);parent[max(a,b)]=min(a,b)
            else: seen[key]=r['recipe_id']
    for r in records:
        r['family_id']=root(r['recipe_id'])
        bucket=int(hashlib.sha256(('curation-v1:'+r['family_id']).encode()).hexdigest()[:8],16)%100
        r['split']='train' if bucket<80 else 'validation' if bucket<90 else 'test'


def conversation(r):
    yield_text=r['batch_yield_original']
    if isinstance(yield_text,dict): yield_text=f"{yield_text['value']} {yield_text['unit']}"
    answer=f"Let's brew {r['title']}. The source recipe makes {yield_text}.\n\nIngredients:\n"
    answer+='\n'.join('- '+x['source_text'] for x in r['ingredients'])
    answer+='\n\nBrewing instructions:\n'+'\n\n'.join(f'{i}. {step}' for i,step in enumerate(r['directions'],1))
    if r['notes']: answer+='\n\nRecipe notes: '+r['notes']
    return {'recipe_id':r['recipe_id'],'family_id':r['family_id'],'split':r['split'],
            'source_url':r['source_url'],'review_status':'candidate_not_approved',
            'messages':[{'role':'system','content':SYSTEM},
                        {'role':'user','content':f"How do I brew {r['title']}? Give me the ingredients and walk me through the process."},
                        {'role':'assistant','content':answer}]}


def build_release(input_path,output):
    payload=input_path.read_bytes()
    records=[clean(r) for r in read_jsonl(input_path)]
    assign_families(records)
    output.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(payload).hexdigest()
    release_path=output/'manifest.json'
    if release_path.exists() and json.loads(release_path.read_text())['input_sha256']!=digest:
        raise ValueError('This release has a different source snapshot; choose a new --output')
    snapshot=output/'source_snapshot.jsonl'
    temporary=snapshot.with_suffix('.jsonl.tmp')
    temporary.write_bytes(payload)
    temporary.replace(snapshot)
    write_jsonl(output/'recipes.jsonl',records)
    tiers=Counter(r['audit']['tier'] for r in records)
    for tier in ['instruction_candidate','needs_enrichment','excluded_nonbeer','quarantine_conflict']:
        write_jsonl(output/f'{tier}.jsonl',[r for r in records if r['audit']['tier']==tier])
    for split in ['train','validation','test']:
        write_jsonl(output/'conversation_candidates'/f'{split}.jsonl',
                    [conversation(r) for r in records if r['audit']['tier']=='instruction_candidate' and r['split']==split])
    flags=Counter(flag for r in records for flag in r['audit']['flags'])
    stages=Counter(stage for r in records for stage in r['audit']['missing_stages'])
    manifest={'input':str(input_path.relative_to(ROOT)) if input_path.is_relative_to(ROOT) else str(input_path),
              'input_sha256':digest,'source_records':len(records),'tiers':dict(tiers),
              'curation_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_snapshot':'source_snapshot.jsonl',
              'source_counts':dict(Counter(r['source'] for r in records)),
              'recipe_families':len({r['family_id'] for r in records}),
              'candidate_splits':dict(Counter(r['split'] for r in records if r['audit']['tier']=='instruction_candidate')),
              'flags':dict(flags),'missing_stages':dict(stages),
              'conversation_candidates':sum(r['audit']['tier']=='instruction_candidate' for r in records),
              'approved_training_examples':0,
              'note':'Instruction candidates passed a structural audit, not an expert review. No missing instructions are invented. Original quantities remain in targets; normalized mass fields are supplementary.'}
    write_json(release_path,manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/collection/unique_recipes.jsonl')
    parser.add_argument('--output',type=Path,default=ROOT/'data/curated_v1')
    args=parser.parse_args()
    print(json.dumps(build_release(args.input,args.output),indent=2))


if __name__=='__main__': main()
