"""Run configured experiments and save evaluation results."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import statistics
import tempfile
import sys

from .config import Experiment

RECIPE_ROOT = Path(__file__).with_name('recipes')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError(f'Unsafe identifier: {value!r}')
    return value


def matrix_jobs(matrix, profiles, output, device='cpu', variants=None):
    if matrix.get('schema_version') != 1:
        raise ValueError('Unsupported matrix schema')
    seeds = matrix['seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('Seeds must be a nonempty list of unique integers')
    jobs = []
    for name in profiles:
        identifier(name)
        profile = matrix['profiles'][name]
        if profile.get('unavailable_reason'):
            raise ValueError(profile['unavailable_reason'])
        selected = variants or [k for k, v in profile['variants'].items() if v['enabled']]
        if not selected:
            raise ValueError(f'{name}: no executable variants')
        for variant in selected:
            identifier(variant)
            spec = profile['variants'][variant]
            if spec.get('unavailable_reason'):
                raise ValueError(f'{name}/{variant}: {spec["unavailable_reason"]}')
            if set(spec['overrides']) - {'curve','ir','consistency'}:
                raise ValueError('Variant overrides support curve, ir, and consistency. Other settings belong in config.')
            stages = spec['stages']
            if not stages or len(set(stages)) != len(stages) or set(stages) - {'base','post'}:
                raise ValueError('stages must select base and/or post (no-post means base)')
            owner = identifier(spec.get('reuse_profile', name))
            if owner != name:
                source = matrix['profiles'][owner]
                if source['config'] != profile['config'] or spec['overrides'] != source['variants'][variant]['overrides']:
                    raise ValueError('Reused configuration does not match the source profile')
            for level in profile['levels']:
                identifier(level)
                for seed in seeds:
                    directory = Path(output).resolve()/owner/variant/level/f's{seed}'
                    cfg = Experiment(**dict(profile['config'], **spec['overrides'], level=level,
                                            seed=seed, reporting_seed=seed, device=device,
                                            output=str(directory/'base'))).validate()
                    jobs.append(dict(profile_id=name,owner_profile=owner,variant=variant,level=level,
                                     seed=seed,stages=stages,directory=str(directory),
                                     config=asdict(cfg),unresolved=profile.get('unresolved')))
    return jobs


def state(path, status, **extra):
    write_json(path/'status.json',dict(status=status,**extra))
    with (path/'status.tsv').open('a') as stream:
        stream.write(f'{datetime.now(timezone.utc).isoformat()}\t{status}\n')


def stage_manifest(job, stage, data_root):
    return dict(schema_version=1,profile_id=job['owner_profile'],variant=job['variant'],
                level=job['level'],model_seed=job['seed'],reporting_seed=job['seed'],
                stage=stage,resolved_config=job['config'],data_root=data_root,
                parent_checkpoint=str(Path(job['directory'])/'base'/'base.pt') if stage=='post' else None,
                selection_protocol=dict(metric='MAE' if job['config']['dataset']=='japan' else 'reportMAE',
                                        direction='min',post_start_epoch=job['config']['post_start_epoch']),
                metric_profile='report_rate_peak_macro_mae_v2')


def check_stage(job, stage, data_root, skip_existing):
    path=Path(job['directory'])/stage
    manifest=stage_manifest(job,stage,data_root)
    if path.exists():
        valid=(path/'run_manifest.json').is_file() and read_json(path/'run_manifest.json')==manifest
        valid=valid and (path/'status.json').is_file() and read_json(path/'status.json')['status']=='trained'
        valid=valid and (path/f'{stage}.pt').is_file() and (path/'selection_freeze.json').is_file()
        if skip_existing and valid:
            import torch
            checkpoint=torch.load(path/f'{stage}.pt',map_location='cpu',weights_only=True)
            freeze=read_json(path/'selection_freeze.json')
            if checkpoint['stage']==stage and checkpoint['epoch']==freeze['epoch'] and checkpoint['config']==freeze['config']:
                return False
        raise FileExistsError(f'{path}: conflict, incomplete run, or config mismatch; use a new output root. ')
    return True


def train_stage(job, stage, data_root):
    import torch
    from .cli import prepare
    path=Path(job['directory'])/stage
    parent=Path(job['directory'])/'base'/'base.pt'
    if stage=='post' and not parent.is_file():
        raise FileNotFoundError(f'Missing corresponding Base: {parent}')
    path.mkdir(parents=True,exist_ok=False)
    manifest=stage_manifest(job,stage,data_root)
    write_json(path/'run_manifest.json',manifest)
    write_json(path/'config.json',job['config'])
    import numpy as np
    write_json(path/'invocation.json',dict(command=list(sys.argv),python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda))
    (path/'command.txt').write_text(shlex.join(sys.argv)+'\n')
    state(path,'running')
    try:
        cfg=Experiment(**dict(job['config'],output=str(path)))
        trainer,loaders=prepare(cfg,data_root)
        restricted={k:v for k,v in loaders.items() if k!='test'}
        if stage=='base':
            trainer.train(restricted,['train','validate'],early_stop_patience=cfg.patience,checkpoint_path=str(path/'base.pt'))
        else:
            trainer.run_post_training(restricted,str(parent))
        checkpoint=torch.load(path/f'{stage}.pt',map_location='cpu',weights_only=True)
        write_json(path/'selection_freeze.json',dict(epoch=checkpoint['epoch'],config=checkpoint['config'],
                   stage=checkpoint['stage'],epoch_mode_losses=checkpoint['epoch_mode_losses']))
        state(path,'trained')
    except BaseException as exc:
        state(path,'cancelled' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
        raise


def evaluate_stage(job,stage,data_root,skip_existing):
    from .cli import run
    checkpoint=Path(job['directory'])/stage/f'{stage}.pt'
    out=Path(job['directory'])/(stage+'_evaluation')
    if out.exists():
        if skip_existing and (out/'evaluation_binding.json').is_file() and read_json(out/'evaluation_binding.json')==stage_manifest(job,stage,data_root):
            if (out/f'{stage}_predictions.npz').is_file() and (out/'status.json').is_file() and read_json(out/'status.json').get('state')=='COMPLETE':
                return
        raise FileExistsError(out)
    run(Experiment(**dict(job['config'],output=str(out))), 'evaluate', checkpoint,data_root)
    write_json(out/'evaluation_binding.json',stage_manifest(job,stage,data_root))


def summarize(jobs,allow_partial=False):
    from .evaluation import rescore
    import numpy as np
    groups={}
    supports={}
    for job in jobs:
        for stage in job['stages']:
            key=(job['profile_id'],job['variant'],job['level'],stage)
            group=groups.setdefault(key,dict(expected=[],rows=[],missing=[]))
            group['expected'].append(job['seed'])
            prediction=Path(job['directory'])/(stage+'_evaluation')/f'{stage}_predictions.npz'
            if not prediction.is_file():
                group['missing'].append(job['seed']);continue
            binding_path=prediction.parent/'evaluation_binding.json'
            if not binding_path.is_file():raise ValueError(f'Missing evaluation binding: {binding_path}')
            binding=read_json(binding_path)
            if binding!=stage_manifest(job,stage,binding.get('data_root')):
                raise ValueError(f'Evaluation configuration differs from selected matrix: {binding_path}')
            with np.load(prediction,allow_pickle=False) as arrays:
                identity=(job['config']['dataset'],job['level'],job['seed'])
                support={k:arrays[k].copy() for k in arrays.files if k in {'target','origin_index','node_ids','future_report_mask','future_report_interval'}}
                if identity in supports:
                    if support.keys()!=supports[identity].keys():raise ValueError('Scoring support fields differ')
                    for k in support:np.testing.assert_array_equal(support[k],supports[identity][k],err_msg=f'Scoring support mismatch: {identity}/{k}')
                else:supports[identity]=support
            metrics=rescore(prediction)
            group['rows'].append(dict(seed=job['seed'],metrics=metrics))
    result=[]
    for key,g in groups.items():
        if len(set(g['expected']))!=len(g['expected']):raise ValueError(f'Duplicate runs: {key}')
        if g['missing'] and not allow_partial:raise ValueError(f'Missing seeds for {key}: {g["missing"]}')
        stats={}
        for metric in ['mae','report_mae','peak_mae']:
            values=[row['metrics'][metric] for row in g['rows'] if isinstance(row['metrics'].get(metric),(int,float))]
            stats[metric]=dict(n=len(values),mean=statistics.mean(values) if values else None,
                              sample_sd=statistics.stdev(values) if len(values)>1 else None)
        result.append(dict(profile=key[0],variant=key[1],level=key[2],stage=key[3],
                           status='partial' if g['missing'] else 'scored',**g,statistics=stats))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['list','plan','validate','train','evaluate','score','summary'])
    parser.add_argument('--matrix',type=Path,default=RECIPE_ROOT/'formal_10seeds.json')
    parser.add_argument('--profiles',nargs='+')
    parser.add_argument('--variants',nargs='+')
    parser.add_argument('--output-root',type=Path,default=Path('outputs/reproduction'))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--stage',choices=['base','post','both'],default='both')
    parser.add_argument('--skip-existing',action='store_true')
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--save',type=Path)
    args=parser.parse_args(argv)
    matrix=read_json(args.matrix)
    if args.action=='list':
        print(json.dumps(matrix,indent=2));return
    if not args.profiles:parser.error('--profiles is required')
    jobs=matrix_jobs(matrix,args.profiles,args.output_root,args.device,args.variants)
    data_root=str(args.data_root.resolve()) if args.data_root else None
    if args.action=='plan':
        result=jobs
    elif args.action=='validate':
        from .cli import prepare
        result=[]
        seen=set()
        for job in jobs:
            key=(job['config']['dataset'],job['level'],job['config']['ir'],job['config']['consistency'])
            if key in seen:continue
            seen.add(key)
            with tempfile.TemporaryDirectory(prefix='costep-validate-') as scratch:
                trainer,loaders=prepare(Experiment(**dict(job['config'],output=scratch)),data_root)
            lengths={k:len(loaders[k].dataset) for k in ['train','validate','test']}
            expected={'japan':[372,63,64],'us':[566,71,73],'japan_irregular':[380,38,40]}[key[0]]
            if list(lengths.values())!=expected:raise ValueError(f'Window protocol mismatch: {lengths}')
            result.append(dict(dataset=key[0],level=key[1],windows=lengths))
    elif args.action in {'score','summary'}:
        result=summarize(jobs,args.allow_partial)
    else:
        for job in jobs:
            if job['unresolved']:raise ValueError(job['unresolved'])
        work=[(job,stage) for stage in ['base','post'] for job in jobs
              if stage in job['stages'] and args.stage in {'both',stage}]
        # Check output paths before training.
        if args.action=='train':
            pending=[];seen=set()
            for job,stage in work:
                key=(job['directory'],stage)
                if key in seen:continue
                seen.add(key)
                if check_stage(job,stage,data_root,args.skip_existing):pending.append((job,stage))
            for job,stage in pending:train_stage(job,stage,data_root)
        else:
            seen=set()
            for job,stage in work:
                key=(job['directory'],stage)
                if key in seen:continue
                seen.add(key)
                evaluate_stage(job,stage,data_root,args.skip_existing)
        result=dict(status='trained' if args.action=='train' else 'evaluated',jobs=len(work))
    if args.save:
        write_json(args.save,result)
    else:print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
