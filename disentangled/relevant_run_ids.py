
import shutil
import os
import re

fd_runs = (
    '24550149',
    '27093745',
    '24550149',
    '27093745',
    '24476757',
    '24476756',
    '24476755',
    '24476757, 24476733, 24476754',
    '24476757, 24476758, 24476759',
    '24550149, 24550148, 24550147',
    '24550149, 24550151, 24550153',
    '24550149, 24550154, 24620473',
    '24550148, 24550151, 24620473',
    '25006249',
    '24836214, 24836234',
    '24833826, 24833841',
    '25626328',
    '26761700,26995361',
    '26999944',
    '27016070, 27016048',
    '27016025, 27016003',
    '27015959, 27015981',
    '27015848, 27015826',
    '27027656, 27027661',
    '27027612, 27027634',
    '27016025, 27016003',
    '27015959, 27015981',
    '27027656, 27027661',
    '27027612, 27027634',
    '3589890, 3589889, 3589888, 3589883, 3589891',
    '3733771, 3733770, 3733769, 3733768, 3733767',
    '24581048, 24581070, 24581092, 24550149',
    '3460221, 3460222, 3460223',
    '3462741, 3462738, 3462742',
    '24550149',
    '3150613',
    '3278158',
    '3278159',
    '24550149',
    '3150467, 3150466, 3150465',
    '3102039',
    '3102036, 3102037, 3102038, 3133088',
    '3278014, 3278015, 3278066, 3278083',
)
full_paths = (
    'conv-rep-ov.2-td5-ld50-bd5-5-1-n_50_196711/genout_models001.tfmod',
    'bconv-rep-ov.2-td4-ld50-bd5-5-1-n_10_197058/genout_models001.tfmod',
)
gp_runs = (
    'mp-gp-comb',
)
bv_runs = (
    '24575724',
    '25626351',
    '24581116, 24581117, 24581118, 24575724',    
)
mv_patterns = (
    '24837[0-9]+',
    '2483[0-9]+',
)

run_pattern = '.*-n_([0-9]*)_{run_inds}'
gp_pattern = '{run_inds}.*'
mv_pattern = '.*-mv_([0-9]*)_{run_inds}'
def _copy_all_run_inds(
        source_folder,
        target_folder,
        run_list,
        run_pattern=run_pattern,
        recopy=False,
        dryrun=False,
):
    full_run_list = []
    for ri in run_list:
        full_run_list.extend(ri.split(', '))
    move_paths = []
    full_str = '({})'.format('|'.join(full_run_list))
    to_match = run_pattern.format(run_inds=full_str)
    to_match = re.compile(to_match)
    fls_all = os.listdir(source_folder)
    for fl in fls_all:
        m = re.match(to_match, fl)
        if m is not None:
            move_paths.append(fl)
    if dryrun:
        print(move_paths)
    else:
        for mp in move_paths:
            sf = os.path.join(source_folder, mp)
            tf = os.path.join(target_folder, mp)
            if not os.path.isdir(tf) or recopy:
                shutil.copytree(sf, tf)
    return move_paths

def _copy_paths(source_folder, target_folder, paths, dryrun=False):
    for p in paths:
        sp = os.path.join(source_folder, p)
        tp = os.path.join(target_folder, p)
        directory, _ = os.path.split(tp)
        if not os.path.isdir(directory):
            os.mkdir(directory)
        if not dryrun and not os.path.isfile(tp):
            shutil.copy2(sp, tp)
        else:
            print(os.path.join(target_folder, p))
    

fd_source_folder = 'disentangled/simulation_data_archive/partition/'
fd_target_folder = 'disentangled/simulation_data/partition/'

bv_source_folder = 'disentangled/simulation_data_archive/bvae/'
bv_target_folder = 'disentangled/simulation_data/bvae/'

mv_source_folder = 'disentangled/disent_multiverse_archive/'
mv_target_folder = 'disentangled/disent_multiverse/'

if __name__ == '__main__':
    dryrun = False
    _copy_all_run_inds(fd_source_folder, fd_target_folder, fd_runs,
                       dryrun=dryrun)

    
    _copy_all_run_inds(bv_source_folder, bv_target_folder, bv_runs,
                       dryrun=dryrun)
    
    _copy_all_run_inds(mv_source_folder, mv_target_folder, mv_patterns,
                       run_pattern=mv_pattern, dryrun=dryrun)
    
    _copy_all_run_inds(fd_source_folder, fd_target_folder, gp_runs,
                       run_pattern=gp_pattern, dryrun=dryrun)

    _copy_paths(fd_source_folder, fd_target_folder, full_paths,
                dryrun=dryrun)
    # write way to transfer full model single files
    
