"""Discover, collect and deduplicate public brewing recipes with a resumable local queue."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
import gzip
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import shutil
import os
import time
import unicodedata
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup
from .aha import parse_recipe
from .common import ROOT, write_json, write_jsonl

OUT = ROOT/'data/collection'
DB = OUT/'queue.sqlite3'
AGENT = 'BeerLLM-Learning/0.1'
SITES = {
    'aha': ('https://homebrewersassociation.org', '/sitemap_index.xml', 10),
    'brewersfriend': ('https://www.brewersfriend.com', '/homebrew/sitemap.xml', 5),
}


def connection():
    OUT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB, timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('''CREATE TABLE IF NOT EXISTS queue (
        url TEXT PRIMARY KEY, source TEXT NOT NULL, source_id TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending', error TEXT, updated_at TEXT,
        UNIQUE(source,source_id))''')
    db.execute('''CREATE TABLE IF NOT EXISTS recipes (
        url TEXT PRIMARY KEY, source TEXT NOT NULL, data TEXT NOT NULL)''')
    db.execute('CREATE TABLE IF NOT EXISTS source_state (source TEXT PRIMARY KEY, paused_until REAL, reason TEXT, failures INTEGER DEFAULT 0)')
    db.commit()
    return db


def now():
    return datetime.now(timezone.utc).isoformat()


def normal(text):
    text = unicodedata.normalize('NFKC', str(text)).casefold()
    return re.sub(r'\s+', ' ', text).strip()


def fingerprint(record):
    # Deliberately conservative: a process/quantity difference must not be discarded.
    body = {key:record.get(key) for key in ['batch_yield','ingredients_text','directions_text','process_text','notes']}
    return hashlib.sha256(normal(json.dumps(body, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()


def family(record):
    title = re.sub(r'\b(?:clone|v\d+|20\d\d|all.grain|extract)\b', ' ', normal(record['title']))
    return re.sub(r'[^\w]+', ' ', title).strip()


def source_id(source, url):
    path = urlparse(url).path
    pattern = r'/homebrew-recipe/([^/]+)/?' if source=='aha' else r'/homebrew/recipe/view/(\d+)(?:/[^/]+)?/?'
    match = re.fullmatch(pattern,path)
    return match.group(1) if match else None


class Client:
    def __init__(self, source):
        self.source = source
        self.origin, _, self.delay = SITES[source]
        self.session = requests.Session()
        self.session.headers['User-Agent'] = AGENT
        self.last = 0
        self.blocked = False
        self.retry_seconds = 3600
        response = self.session.get(self.origin+'/robots.txt',timeout=40)
        response.raise_for_status()
        self.robots = RobotFileParser()
        self.robots.parse(response.text.splitlines())
        self.delay = max(self.delay,self.robots.crawl_delay(AGENT) or self.robots.crawl_delay('*') or 0)
        self.last = time.monotonic()

    def get(self, url):
        if urlparse(url).netloc != urlparse(self.origin).netloc or not self.robots.can_fetch(AGENT,url):
            raise ValueError('URL outside the permitted source or excluded by robots.txt')
        if self.blocked:
            raise ValueError('Source paused after access denial or throttling')
        time.sleep(max(0,self.delay-(time.monotonic()-self.last)))
        try:
            response = self.session.get(url,timeout=45,allow_redirects=False)
        finally:
            self.last = time.monotonic()
        if response.status_code in [401,403,429]:
            self.blocked = True
            if response.status_code==429:
                value=response.headers.get('Retry-After','')
                try:
                    self.retry_seconds=max(3600,float(value))
                except ValueError:
                    from email.utils import parsedate_to_datetime
                    try: self.retry_seconds=max(3600,parsedate_to_datetime(value).timestamp()-time.time())
                    except (ValueError,TypeError): pass
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError(f'Unexpected HTTP {response.status_code}; redirects/login are not followed')
        return response


def locations(xml):
    root = ET.fromstring(xml)
    return root.tag.rsplit('}',1)[-1], [e.text.strip() for e in root.iter()
        if e.tag.rsplit('}',1)[-1]=='loc' and e.text]


def discover(source):
    client = Client(source)
    origin,index,_ = SITES[source]
    kind,children = locations(client.get(origin+index).content)
    pattern = 'recipes-sitemap' if source=='aha' else 'sitemap-recipes-'
    maps = [u for u in children if pattern in u] if kind=='sitemapindex' else [origin+index]
    db = connection()
    for sitemap in maps:
        response = client.get(sitemap)
        path = OUT/'sitemaps'/f'{source}-{hashlib.sha256(sitemap.encode()).hexdigest()[:12]}.xml.gz'
        path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(path,'wb') as stream: stream.write(response.content)
        _,urls = locations(response.content)
        rows = []
        for url in urls:
            # AHA sitemap currently uses canonical non-www URLs.
            if source=='aha': url=url.replace('https://www.homebrewersassociation.org/','https://homebrewersassociation.org/')
            ident = source_id(source,url)
            if ident and urlparse(url).netloc==urlparse(origin).netloc:
                rows.append((url,source,ident))
        db.executemany('INSERT OR IGNORE INTO queue(url,source,source_id) VALUES(?,?,?)', rows)
        db.commit()
        print(f'{source}: indexed {len(rows)} recipe URLs from {sitemap}',flush=True)
    db.close()


def instructions(value):
    if isinstance(value,str): return [BeautifulSoup(value,'html.parser').get_text(' ',strip=True)]
    if isinstance(value,list): return [step for x in value for step in instructions(x)]
    if isinstance(value,dict):
        return instructions(value.get('itemListElement') or value.get('text') or '')
    return []


def parse_brewersfriend(html,url):
    soup=BeautifulSoup(html,'html.parser')
    recipe=None
    def find(value):
        if isinstance(value,list):
            for v in value:
                result=find(v)
                if result: return result
        if isinstance(value,dict):
            types=value.get('@type',[])
            if types=='Recipe' or isinstance(types,list) and 'Recipe' in types: return value
            return find(value.get('@graph',[]))
        return None
    for tag in soup.select('script[type="application/ld+json"]'):
        try: recipe=find(json.loads(tag.get_text()))
        except json.JSONDecodeError: continue
        if recipe: break
    if not recipe or not recipe.get('recipeIngredient'):
        raise ValueError('No public recipe ingredients found; unavailable/restricted or unsupported page')
    detailed=next((t.get_text() for t in soup.find_all('textarea') if 'HOME BREW RECIPE:' in t.get_text()),'')
    # Full public recipe text retains hop timings, yeast and notes absent from JSON-LD ingredients.
    detailed=re.split(r'This recipe has been published online at:',detailed)[0]
    detailed=re.sub(r'^Title:.*$', '', detailed, flags=re.M).strip()
    return {'title':recipe['name'],'source_url':url,'batch_yield':recipe.get('recipeYield'),
            'ingredients_text':recipe['recipeIngredient'],
            'directions_text':instructions(recipe.get('recipeInstructions',[])),
            'process_text':detailed,'notes':None,
            'author':recipe.get('author'),'style':recipe.get('recipeCategory'),
            'instruction_origin':'site_structured_data_and_public_recipe_text',
            'status':'needs_review','rights_status':'unverified','expert_reviewed':False}


def parse_aha(html,url):
    record=parse_recipe(html,url)
    return record | {'process_text':None,'notes':None,'instruction_origin':'original_written_directions',
                     'status':'needs_review'}


def save_record(db,source,url,record):
    record['source']=source
    record.setdefault('fetched_at',now())
    record['imported_at']=now()
    db.execute('INSERT OR REPLACE INTO recipes(url,source,data) VALUES(?,?,?)',(url,source,json.dumps(record,ensure_ascii=False)))
    db.execute("UPDATE queue SET status='done',error=NULL,updated_at=? WHERE url=?",(now(),url))
    db.commit()


def import_brewdog():
    path=ROOT/'data/punkapi/raw.json'
    if not path.exists():
        from scripts.fetch_recipes import download
        write_json(path,download())
    recipes=json.loads(path.read_text())
    snapshot=ROOT/'data/punkapi/report.json'
    snapshot_time=json.loads(snapshot.read_text()).get('fetched_at') if snapshot.exists() else None
    db=connection()
    for r in recipes:
        url=f"https://punkapi.dev/v3/beers/{r['id']}"
        ingredients=[]
        for kind in ['malt','hops']:
            for x in r['ingredients'].get(kind,[]):
                q=x['amount']
                suffix=f"; addition: {x['add']}" if 'add' in x else ''
                ingredients.append(f"{q['value']} {q['unit']} {x['name']}{suffix}")
        ingredients.append('Yeast: '+str(r['ingredients'].get('yeast') or 'not specified'))
        record={'title':r['name'],'source_url':url,'batch_yield':r.get('volume'),
                'ingredients_text':ingredients,'directions_text':[],
                'process_text':json.dumps(r.get('method'),ensure_ascii=False,sort_keys=True),
                'notes':r.get('brewers_tips'),'source_recipe':r,
                'instruction_origin':'structured_fields_and_brewer_tips',
                'status':'needs_review','rights_status':'unverified','expert_reviewed':False,
                'fetched_at':snapshot_time}
        db.execute('INSERT OR IGNORE INTO queue(url,source,source_id) VALUES(?,?,?)',(url,'brewdog',str(r['id'])))
        save_record(db,'brewdog',url,record)
    db.close()
    print(f'brewdog: imported {len(recipes)} recipes from the local API snapshot',flush=True)


def import_cached_aha():
    path=ROOT/'data/aha/recipes.jsonl'
    if not path.exists(): return
    db=connection()
    for line in path.read_text().splitlines():
        r=json.loads(line)
        url=r['source_url']
        if db.execute('SELECT 1 FROM recipes WHERE url=?',(url,)).fetchone(): continue
        ident=source_id('aha',url)
        if not ident: continue
        db.execute('INSERT OR IGNORE INTO queue(url,source,source_id) VALUES(?,?,?)',(url,'aha',ident))
        r.update(process_text=None,notes=None,instruction_origin='original_written_directions',status='needs_review')
        save_record(db,'aha',url,r)
    db.close()


def collect(source,limit):
    db=connection()
    pause=db.execute('SELECT paused_until,reason,failures FROM source_state WHERE source=?',(source,)).fetchone()
    db.close()
    if pause:
        until,reason,failures=pause
        if until is None or failures>=3:
            print(f'{source}: paused; review access errors before resuming: {reason}',flush=True)
            return
        if until>time.time():
            print(f'{source}: cooling down until {datetime.fromtimestamp(until,timezone.utc).isoformat()}',flush=True)
            while time.time()<until:
                if (OUT/'STOP').exists(): return
                time.sleep(max(0,min(30,until-time.time())))
    client=Client(source)
    db=connection()
    sql="SELECT url FROM queue WHERE source=? AND status='pending' ORDER BY source_id"
    rows=db.execute(sql+(" LIMIT ?" if limit else ''),(source,limit) if limit else (source,)).fetchall()
    for offset,(url,) in enumerate(rows):
        if (OUT/'STOP').exists():
            print(f'{source}: stopped by data/collection/STOP',flush=True)
            break
        if shutil.disk_usage(ROOT).free < 5*1024**3:
            print(f'{source}: stopped with less than 5 GiB free disk space',flush=True)
            break
        key=hashlib.sha256(url.encode()).hexdigest()
        path=OUT/'raw'/source/f'{key}.html.gz'
        try:
            if path.exists():
                with gzip.open(path,'rt',encoding='utf-8') as stream: html=stream.read()
            else:
                html=client.get(url).text
                path.parent.mkdir(parents=True,exist_ok=True)
                with gzip.open(path,'wt',encoding='utf-8') as stream: stream.write(html)
            parser=parse_aha if source=='aha' else parse_brewersfriend
            record=parser(html,url)
            record['html_sha256']=hashlib.sha256(html.encode()).hexdigest()
            record['raw_html']=str(path.relative_to(ROOT))
            save_record(db,source,url,record)
            db.execute('DELETE FROM source_state WHERE source=?',(source,))
            db.commit()
            print(f"{source}: {record['title']}",flush=True)
        except (requests.RequestException,ValueError) as exc:
            db.execute("UPDATE queue SET status='error',error=?,updated_at=? WHERE url=?",(str(exc),now(),url))
            db.commit()
            print(f'{source}: skipped {url}: {exc}',flush=True)
            if client.blocked:
                code=exc.response.status_code if isinstance(exc,requests.HTTPError) and exc.response is not None else None
                until=time.time()+client.retry_seconds if code==429 else None
                previous=db.execute('SELECT failures FROM source_state WHERE source=?',(source,)).fetchone()
                failures=(previous[0] if previous else 0)+1
                db.execute('INSERT OR REPLACE INTO source_state VALUES(?,?,?,?)',(source,until,str(exc),failures))
                if code==429:
                    db.execute("UPDATE queue SET status='pending' WHERE url=?",(url,))
                db.commit()
                db.close()
                if code==429 and failures<3:
                    return collect(source,max(1,limit-offset) if limit else 0)
                return
    db.close()


def export_snapshot():
    db=connection()
    unique={}
    groups=defaultdict(list)
    source_counts=Counter()
    for source,url,payload in db.execute('SELECT source,url,data FROM recipes ORDER BY source,url'):
        r=json.loads(payload)
        source_counts[source]+=1
        key=fingerprint(r)
        provenance={'source':source,'url':url,'fetched_at':r['fetched_at']}
        if key in unique:
            unique[key]['sources'].append(provenance)
        else:
            unique[key]=r | {'content_id':key,'sources':[provenance]}
            groups[family(r)].append(key)
    write_jsonl(OUT/'unique_recipes.jsonl',unique.values())
    related=[{'name_key':key,'content_ids':ids} for key,ids in groups.items() if len(ids)>1]
    write_jsonl(OUT/'similar_recipe_groups.jsonl',related)
    paused=[dict(source=s,paused_until=until,reason=reason,consecutive_failures=n) for s,until,reason,n in db.execute('SELECT * FROM source_state')]
    queues=[dict(source=s,status=status,count=n) for s,status,n in db.execute('SELECT source,status,count(*) FROM queue GROUP BY source,status')]
    report={'updated_at':now(),'collected_source_records':sum(source_counts.values()),'source_counts':dict(source_counts),
            'unique_exact_content':len(unique),'exact_duplicates_merged':sum(source_counts.values())-len(unique),
            'possible_related_groups':len(related),'queue':queues,'source_pauses':paused,
            'with_written_or_site_structured_directions':sum(bool(r['directions_text']) for r in unique.values()),
            'with_original_written_directions':sum(r.get('instruction_origin')=='original_written_directions' for r in unique.values()),
            'note':'Unique means conservative exact content after whitespace/case normalization. Similar recipes are flagged, not discarded; cross-source paraphrases and unit variants need review. Collected records are not automatically training-ready.'}
    write_json(OUT/'report.json',report)
    db.close()
    print(json.dumps(report,indent=2),flush=True)


def export():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        export_snapshot()


def status():
    db=connection()
    rows=[{'source':s,'status':state,'count':n} for s,state,n in
          db.execute('SELECT source,status,count(*) FROM queue GROUP BY source,status')]
    pauses=[dict(source=s,paused_until=until,reason=reason,consecutive_failures=n) for s,until,reason,n in db.execute('SELECT * FROM source_state')]
    count=db.execute('SELECT count(*) FROM recipes').fetchone()[0]
    db.close()
    print(json.dumps({'updated_at':now(),'collected_source_records':count,'queue':rows,'source_pauses':pauses,
                      'snapshot_report':str(OUT/'report.json')},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['discover','run','export','status'])
    parser.add_argument('--sources',nargs='+',choices=['aha','brewersfriend','brewdog'],default=['aha','brewersfriend','brewdog'])
    parser.add_argument('--limit',type=int,default=50,help='Maximum pending pages per website; 0 processes the entire queue')
    args=parser.parse_args()
    if args.limit<0: parser.error('--limit must be >= 0')
    if args.action=='status':
        status()
        return
    if args.action=='discover':
        for source in args.sources:
            if source=='brewdog': import_brewdog()
            else: discover(source)
    elif args.action=='run':
        if 'brewdog' in args.sources: import_brewdog()
        if 'aha' in args.sources: import_cached_aha()
        selected=[s for s in args.sources if s in SITES]
        # One sequential worker per website; independent hosts can run together.
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'collector.lock').open('a') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit('Another collector is running; use status/export to inspect it')
            lock.seek(0)
            lock.truncate()
            lock.write(str(os.getpid())+'\n')
            lock.flush()
            with ThreadPoolExecutor(max_workers=max(1,len(selected))) as pool:
                pending={pool.submit(collect,source,args.limit) for source in selected}
                while pending:
                    done,pending=wait(pending,timeout=60)
                    export()
                    for future in done: future.result()
    export()


if __name__=='__main__': main()
