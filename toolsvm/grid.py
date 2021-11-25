__all__ = ['find_parameters']

import os
import sys
import re
import click
import numpy as np
import logging
import ray
import concurrent.futures
from itertools import product
from libsvm import svmutil
from pygnuplot import gnuplot as pygnuplot
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')


class GridOption:

    def __init__(self, options):
        self.fold = 5
        self.svm_type = options['svm_type']
        self.c_begin, self.c_end, self.c_step = options['log2c']
        self.g_begin, self.g_end, self.g_step = options['log2g']
        self.p_begin, self.p_end, self.p_step = options['log2p']
        self.grid_with_c = options['c']
        self.grid_with_g = options['g']
        self.grid_with_p = options['p'] and self.svm_type in [3, 4]
        self.dataset = svmutil.svm_read_problem(options['dataset'])
        self.dataset_title = os.path.split(options['dataset'])[1]
        self.out = options['out']
        self.out_pathname = '{0}.out'.format(self.dataset_title if options['out_pathname'] is None else options['out_pathname'])
        self.png_pathname = '{0}.png'.format(self.dataset_title if options['out_pathname'] is None else options['out_pathname'])
        self.resume_pathname = options['resume']
        self.nb_process = options['nb_process']
        self.svm_options = " ".join(options['svm_options'])
        self.with_gnuplot = self.grid_with_c and self.grid_with_g
        self.validate_options()

    def validate_options(self):
        if not any([self.grid_with_c, self.grid_with_g, self.grid_with_p]):
            raise ValueError('-c , -g and -p should not be disabled simultaneously')

    def evaluate_rate(self, rate, best_rate):
        if self.rate_kind == 'rate':
            return rate > best_rate
        return rate < best_rate

    @property
    def rate_kind(self):
        return 'mse' if self.svm_type in [3, 4] else 'rate'


def redraw(db,best_param,gnuplot,options,to_file=False):
    if len(db) == 0: return
    begin_level = round(max(x[2] for x in db)) - 3
    step_size = 0.5
    best_log2c,best_log2g,best_rate = best_param
    # if newly obtained c, g, or cv values are the same,
    # then stop redrawing the contour.
    if all(x[0] == db[0][0]  for x in db): return
    if all(x[1] == db[0][1]  for x in db): return
    if all(x[2] == db[0][2]  for x in db): return

    if to_file:
        gnuplot.write(b"set term png transparent small linewidth 2 medium enhanced\n")
        gnuplot.write("set output \"{0}\"\n".format(options.png_pathname.replace('\\','\\\\')).encode())
        #gnuplot.write(b"set term postscript color solid\n")
        #gnuplot.write("set output \"{0}.ps\"\n".format(options.dataset_title).encode().encode())
    elif sys.platform == 'win32':
        gnuplot.write(b"set term windows\n")
    else:
        gnuplot.write( b"set term x11\n")
    gnuplot.write(b"set xlabel \"log2(C)\"\n")
    gnuplot.write(b"set ylabel \"log2(gamma)\"\n")
    gnuplot.write("set xrange [{0}:{1}]\n".format(options.c_begin,options.c_end).encode())
    gnuplot.write("set yrange [{0}:{1}]\n".format(options.g_begin,options.g_end).encode())
    gnuplot.write(b"set contour\n")
    gnuplot.write("set cntrparam levels incremental {0},{1},100\n".format(begin_level,step_size).encode())
    gnuplot.write(b"unset surface\n")
    gnuplot.write(b"unset ztics\n")
    gnuplot.write(b"set view 0,0\n")
    gnuplot.write("set title \"{0}\"\n".format(options.dataset_title).encode())
    gnuplot.write(b"unset label\n")
    gnuplot.write("set label \"Best log2(C) = {0}  log2(gamma) = {1}  accuracy = {2}%\" \
                  at screen 0.5,0.85 center\n". \
                  format(best_log2c, best_log2g, best_rate).encode())
    gnuplot.write("set label \"C = {0}  gamma = {1}\""
                  " at screen 0.5,0.8 center\n".format(2.0**best_log2c, 2.0**best_log2g).encode())
    gnuplot.write(b"set key at screen 0.9,0.9\n")
    gnuplot.write(b"splot \"-\" with lines\n")

    db.sort(key = lambda x:(x[0], -x[1]))

    prevc = db[0][0]
    for line in db:
        if prevc != line[0]:
            gnuplot.write(b"\n")
            prevc = line[0]
        gnuplot.write("{0[0]} {0[1]} {0[2]}\n".format(line).encode())
    gnuplot.write(b"e\n")
    gnuplot.write(b"\n") # force g back to prompt when term set failure
    g.flush()


def calculate_jobs(options):

    c_seq = np.random.permutation(np.arange(options.c_begin,options.c_end,options.c_step))
    g_seq = np.random.permutation(np.arange(options.g_begin,options.g_end,options.g_step))
    p_seq = np.random.permutation(np.arange(options.p_begin,options.p_end,options.p_step))

    if not options.grid_with_c:
        c_seq = [None]
    if not options.grid_with_g:
        g_seq = [None]
    if not options.grid_with_p:
        p_seq = [None]

    jobs = product(c_seq, g_seq, p_seq)
    resumed_jobs = {}

    if options.resume_pathname is None:
        return jobs, resumed_jobs

    with open(options.resume_pathname, 'r') as resume:
        resumed_jobs = {
            (c, g, p): rate
            for (c, g, p, rate) in
            map(
                lambda x: map(lambda y: float(y) if y else None, x),
                re.findall(
                    (
                        r'(?:log2c=([0-9.-]+) )?'
                        r'(?:log2g=([0-9.-]+) )?'
                        r'(?:log2p=([0-9,-]+) )?'
                        rf'{options.rate_kind}=([0-9.]+)'
                    ),
                    resume.read()
                )
            )
        }
    return jobs, resumed_jobs


@ray.remote
def evaluate(name, c, g, p, options):
    cmdline = ['-q', f'-s {options.svm_type}']

    if options.grid_with_c:
        cmdline.append(f'-c {2.0**c}')

    if options.grid_with_g:
        cmdline.append(f'-g {2.0**g}')

    if options.grid_with_p:
        cmdline.append(f'-p {2.0**p}')

    cmdline.append(f'-v {options.fold} {options.svm_options}')
    return (name, c, g, p, svmutil.svm_train(*options.dataset, " ".join(cmdline)))


def find_parameters(params={}):

    def update_param(c, g, p, rate, best_c, best_g, best_p, best_rate, worker, resumed, options):
        if options.evaluate_rate( rate, best_rate ) or (rate == best_rate and g==best_g  and p == best_p and c < best_c):
            best_rate, best_c, best_g, best_p = rate, c, g, p

        stdout_str = [f'[{worker}] {" ".join(str(x) for x in [c, g, p] if x is not None)} {rate} (best']
        output_str = []
        if options.grid_with_c:
            stdout_str.append(f'c={2.0**best_c},')
            output_str.append(f'log2c={c}')

        if options.grid_with_g:
            stdout_str.append(f'g={2.0**best_g},')
            output_str.append(f'log2g={g}')

        if options.grid_with_p:
            stdout_str.append(f'p={2.0**best_p},')
            output_str.append(f'log2p={p}')

        stdout_str.append(f'{options.rate_kind}={best_rate})')
        logging.info(" ".join(stdout_str))
        if options.out_pathname and not resumed:
            output_str.append(f'{options.rate_kind}={rate}\n')
            result_file.write(" ".join(output_str))
            result_file.flush()

        return best_c, best_g, best_p, best_rate

    options = GridOption(params)
    jobs, resumed_jobs = calculate_jobs(options)
    gnuplot = pygnuplot.Gnuplot()


    workers = [evaluate.remote('local', c, g, p, options) for (c, g , p) in jobs if (c, g, p) not in resumed_jobs]

    if options.out:
        if options.resume_pathname:
            result_file = open(options.out_pathname, 'a')
        else:
            result_file = open(options.out_pathname, 'w')
    db = []
    best_rate = float('+inf') if options.svm_type in [3, 4] else -1
    best_c, best_g, best_p = None, None, None
    for (c, g, p) in resumed_jobs:
        rate = resumed_jobs[(c, g, p)]
        best_c,best_g,best_p,best_rate = update_param(c, g, p, rate, best_c, best_g, best_p, best_rate, 'resumed', True,options)
        db.append((c, g, rate))
        if options.with_gnuplot:
            redraw(db, [best_c, best_g, best_rate], gnuplot, options)
            redraw(db, [best_c, best_g, best_rate], gnuplot, options, to_file=True)

    futs = [worker.future() for worker in workers]
    for fut in concurrent.futures.as_completed(futs):
        (worker, c, g, p, rate) = fut.result()
        best_c, best_g, best_p, best_rate = update_param(c,g,p,rate,best_c,best_g,best_p,best_rate,worker,False,options)
        db.append((c, g, rate))
        if options.with_gnuplot:
            redraw(db, [best_c, best_g, best_rate], gnuplot, options)
            redraw(db, [best_c, best_g, best_rate], gnuplot, options, to_file=True)

    if options.out:
        result_file.close()

    best_param, best_cgp  = {}, []

    if best_c != None:
        best_param['c'] = 2.0**best_c
        best_cgp.append(2.0**best_c)

    if best_g != None:
        best_param['g'] = 2.0**best_g
        best_cgp.append(2.0**best_g)

    if best_p != None:
        best_param['p'] = 2.0**best_p
        best_cgp.append(2.0**best_p)

    logging.info(f'{" ".join(map(str, best_cgp))} {best_rate}')

    return best_rate, best_param


@click.command()
@click.argument('dataset', type=click.Path(exists=True))
@click.option('--svm-type', default=0, help="""\b
set type of SVM (default 0):
0 -- C-SVC (multi-class classification)
1 -- nu-SVC (multi-class classification)
2 -- one-class SVM
3 -- epsilon-SVR (regression)
4 -- nu-SVR (regression)""")
@click.option('--log2c', nargs=3, default=(-1, 6, 1), help='c_range = 2^{begin,...,begin+k*step,...,end}')
@click.option('--log2g', nargs=3, default=(0, -8, -1), help='g_range = 2^{begin,...,begin+k*step,...,end}')
@click.option('--log2p', nargs=3, default=(-8, -1, 1), help='p_range = 2^{begin,...,begin+k*step,...,end}')
@click.option('--c/--no-c', default=True, help='Disable the usage of log2c')
@click.option('--g/--no-g', default=True, help='Disable the usage of log2g')
@click.option('--p/--no-p', default=True, help='Disable the usage of log2p')
@click.option('-v', type=int, default=5, help='n fold validation')
@click.option('--out-pathname', type=click.Path(), help='set output file path and name')
@click.option('--out/--no-out', default=True, help='Disable out file')
@click.option('--resume', type=click.Path(exists=True), help='Existing old out file to resume from.')
@click.option('--svm-options', multiple=True, help='additionals svm options')
@click.option('--nb-process', default=1)
def main(**params):
    ray.init()
    find_parameters(params)


if __name__ == '__main__':
    main()
