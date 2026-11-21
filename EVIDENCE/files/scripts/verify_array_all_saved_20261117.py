"""One entry point for all saved-result audits. No estimator loading or inference."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
OUT=ROOT/'reports/ARRAY_ENHANCEMENT_20261117'
def main():
    import audit_array_revision_20261117 as audit
    import verify_array_revision_20261117 as parent
    import verify_array_enhancement_20261117 as enhanced
    destination=OUT/'parent_saved_checks';destination.mkdir(parents=True,exist_ok=True)
    audit.REPORT=destination
    for phase in ['core','integration','concise','external','guard','payload']:audit.run(phase)
    for name in ['denominator_and_comparison_audit.json','figure_revision.json']:
        shutil.copy2(ROOT/'reports/ARRAY_REVISION_20261117'/name,destination/name)
    parent.OUT=destination;parent.main();enhanced.main()
    result=dict(status='PASS_ALL_EIGHT_SAVED_RESULT_AUDITS',parent_checks=6,parent_revision=True,enhanced=True,models_deserialized=False,new_fits=0,new_inference=0)
    (OUT/'all_saved_verification.json').write_text(json.dumps(result,indent=2)+'\n','utf8');print(result['status'])
if __name__=='__main__':main()
