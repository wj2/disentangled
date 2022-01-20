
import os
import itertools as it	

if __name__ == '__main__':
    dg_lengths = (.2, .4, .6, .8, 1, 2, 4, 6)
    task_lengths = (1, 2, 4, 6, 10, 20)
    templ_path = 'multi_partition_gp-comb_template.sbatch'
    save_folder = 'gp_jobs'
    save_path =	'multi_gp-comb_dg{}-task{}.sbatch'
    template = ''.join(open(templ_path, 'r').readlines())
    for (dg_len, task_len) in it.product(dg_lengths, task_lengths):
        new_str = template.format(dg_length=dg_len, task_length=task_len)
        print(new_str)
        save_file = os.path.join(save_folder,
                                 save_path.format(dg_len, task_len))
        open(save_file, 'w').writelines(new_str)
