"""Summarize archived practical observations; this does not run new searches."""
import argparse
from pathlib import Path
from common import ROOT, read, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'reproduced/practical_summary.json')
    args=p.parse_args()
    cases=read(ROOT/'data/practical_propagation_cases_14.json')
    outputs=read(ROOT/'results/practical_localization_results_14.json')
    assert {x['case_id'] for x in cases}=={x['case_id'] for x in outputs}
    pilot=read(ROOT/'results/practical_pilot_observations.json')
    feedback=read(ROOT/'results/maintainer_feedback.json')
    summary={'archived_confirmation_cases':len(cases),
             'cases_with_at_least_one_file_hit':sum(x['file_hits']>0 for x in outputs),
             'cases_with_at_least_one_statement_hit':sum((x.get('statement_hits') or 0)>0 for x in outputs),
             'pilot_sources':len(pilot['results']),
             'pilot_target_inspections':sum(len(x['targets']) for x in pilot['results']),
             'maintainer_reports':len(feedback),
             'planned_upstream_merge':sum(x['outcome']=='planned_upstream_merge' for x in feedback),
             'not_applicable':sum(x['outcome']=='not_applicable' for x in feedback),
             'note':'Archived observations, not fresh inference or a prospective success rate.'}
    save(args.output,summary);print(summary)


if __name__=='__main__':main()
