"""Summarize the released empirical records without reassigning categories."""
import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from common import ROOT,save

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'reproduced/empirical.json')
    args=p.parse_args()
    with (ROOT/'data/empirical_records.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    # The CSV retains the source fields used to draw the empirical figures.
    result={'pairs':len(rows),'columns':list(rows[0]),'distributions':{}}
    for key in rows[0]:
        values=Counter(r[key] for r in rows)
        if len(values)<=20:result['distributions'][key]=dict(values)
    values=sorted(float(r['creation_delay_days']) for r in rows if r['creation_delay_days'])
    def quantile(p):
        x=(len(values)-1)*p;i=int(x);j=min(i+1,len(values)-1)
        return values[i]+(values[j]-values[i])*(x-i)
    result['record_time_days']={'count':len(values),'median':quantile(.5),'p75':quantile(.75),'p90':quantile(.9),
                                'negative_count':sum(v<0 for v in values)}
    save(args.output,result)
    print('Empirical records:',len(rows))

if __name__=='__main__':main()
