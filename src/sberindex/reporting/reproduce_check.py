"""Relocate a minimal refresh bundle and compare regenerated numerical evidence.

Uses the current Python environment: checks path independence and refresh
completeness, not clean dependency installation or full model retraining.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
import pandas as pd

from sberindex.paths import ROOT
from sberindex.input_integrity import CACHED


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--python',type=Path,help='Interpreter in a separately installed environment')
    args=parser.parse_args()
    # Keep the venv symlink path: resolving it would launch the base interpreter.
    interpreter=str(args.python.absolute()) if args.python else sys.executable
    environment=json.loads(subprocess.check_output([interpreter,'-c',
        'import json,sys;print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix)))'],text=True))
    if args.python and (environment['prefix']==environment['base_prefix'] or Path(environment['prefix'])==Path(sys.prefix)):
        raise ValueError('--python must identify a separate virtual environment')
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='sberindex-reproduce-') as temp:
        target=Path(temp)/'project';target.mkdir()
        for name in ('run_pipeline.py', 'config.json', 'data_sources.json', 'refresh_inputs.json'):
            shutil.copy2(ROOT/name, target/name)
        shutil.copytree(ROOT/'src', target/'src', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(ROOT/'data',target/'data')
        shutil.copytree(ROOT/'configs',target/'configs')
        shutil.copytree(ROOT/'docs',target/'docs')
        shutil.copytree(ROOT/'tools',target/'tools',ignore=shutil.ignore_patterns('__pycache__'))
        (target/'results').mkdir()
        (target/'reports').mkdir()
        for name in CACHED:shutil.copy2(ROOT/'results'/name,target/'results'/name)
        cached_hashes={name:sha(target/'results'/name) for name in CACHED}
        (ROOT/'run_logs').mkdir(exist_ok=True)
        log_path=ROOT/'run_logs/reproduce_check.log'
        with log_path.open('w') as log:
            completed=subprocess.run([interpreter,str(target/'run_pipeline.py'),'--mode','refresh'],cwd=temp,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f'Relocated refresh failed; see {log_path}')
        compared=[];byte_equal=[]
        for path in sorted((target/'results').iterdir()):
            if path.name in CACHED:
                assert sha(path)==cached_hashes[path.name], f'Cached input changed: {path.name}'
                continue
            reference=ROOT/'results'/path.name
            assert reference.exists(), f'Missing reference: {path.name}'
            if path.suffix=='.csv':
                pd.testing.assert_frame_equal(pd.read_csv(reference),pd.read_csv(path),check_exact=False,rtol=1e-10,atol=1e-10)
            elif path.suffix=='.parquet':
                pd.testing.assert_frame_equal(pd.read_parquet(reference),pd.read_parquet(path),check_exact=False,rtol=1e-10,atol=1e-10)
            elif path.suffix=='.json':
                assert json.loads(reference.read_text())==json.loads(path.read_text()),path.name
            else:raise ValueError(path)
            compared.append(path.name)
            if sha(path)==sha(reference):byte_equal.append(path.name)
        figures=sorted(p.name for p in (target/'figures').glob('*.png'))
        assert len(figures)==29
        for name in figures:assert sha(target/'figures'/name)==sha(ROOT/'figures'/name),name
        vectors=sorted(p.name for p in (target/'figures').glob('*.svg'))
        assert len(vectors)==len(figures)
        for name in vectors:assert sha(target/'figures'/name)==sha(ROOT/'figures'/name),name
        appendix_equal=[]
        for directory in ['detector_tradeoffs','residual_features','operational_workflow']:
            produced=target/'reports'/directory
            expected=ROOT/'reports'/directory
            assert {p.name for p in produced.iterdir()}=={p.name for p in expected.iterdir()}, directory
            for path in sorted(produced.iterdir()):
                assert sha(path)==sha(expected/path.name),str(path)
                appendix_equal.append(str(path.relative_to(target)))
        source_code=[ROOT/'run_pipeline.py', *sorted((ROOT/'src').rglob('*.py')), *sorted((ROOT/'tools').glob('*.py'))]
        code_hashes={str(path.relative_to(ROOT)):sha(path) for path in source_code}
        for relative,digest in code_hashes.items():
            assert sha(target/relative)==digest, f'Code changed during reproduction: {relative}'
        report={'status':'passed','seconds':round(time.perf_counter()-started,2),
            'cached_inputs_sha256':cached_hashes,'regenerated_results_compared':compared,
            'regenerated_count':len(compared),'byte_identical_results':len(byte_equal),
            'png_figures_byte_identical':figures,
            'svg_figures_byte_identical':vectors,
            'code_sha256':code_hashes,
            'appendix_files_byte_identical':appendix_equal,
            'interpreter':interpreter,
            'separate_environment':bool(args.python),
            'environment':environment,
            'pipeline_steps':len(json.loads((target/'reports/run_metadata.json').read_text())['steps']),
            'checks':'Different absolute directory and working directory; generated results initially absent; cache unchanged; numerical/JSON agreement; PNG agreement',
            'limits':('Separate installed Python environment; installation provenance is recorded in workflow_environment.json. ' if args.python else 'Same installed Python environment. ')+ 'No network isolation or full Prophet/Chronos retraining. Saved core predictions remain required inputs. Residual HGB and delayed Prophet are refitted. No independent time holdout.'}
    filename='workflow_clean_reproduction.json' if args.python else 'reproducibility_report.json'
    (ROOT/'reports'/filename).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k in ['status','seconds','regenerated_count','byte_identical_results','limits']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
