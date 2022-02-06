__all__ = ['find_parameters', 'evaluate_svm_classifier']

import os
import sys
import re
import click
import numpy as np
import logging
import ray
import concurrent.futures
import dataclasses
from itertools import product
from libsvm import svmutil
from pygnuplot import gnuplot as pygnuplot
from typing import Callable, Iterable, Iterator, NamedTuple, Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')


class GridRange(NamedTuple):
    begin: float
    end: float
    step: float


@dataclasses.dataclass
class GridHyperParameter:
    log2c: float
    log2g: float
    log2p: float
    rate: float = None

    def __eq__(self, ghp) -> bool:
        return (self.log2c, self.log2g, self.log2p) == (ghp.log2c, ghp.log2g, ghp.log2p)

    def __hash__(self) -> int:
        return hash((self.log2c, self.log2g, self.log2p))

    @property
    def c(self) -> Optional[float]:
        return 2.0**self.log2c if self.log2c is not None else None

    @property
    def g(self) -> Optional[float]:
        return 2.0**self.log2g if self.log2g is not None else None

    @property
    def p(self) -> Optional[float]:
        return 2.0**self.log2p if self.log2p is not None else None


class GridOption:

    DEFAULTS = {
        "fold": 5,
        "svm_type": 0,
        "c_range":(-1, 6, 1),
        "g_range":(0, -8, -1),
        "p_range":(-8, -1, 1),
        "with_c": True,
        "with_g": True,
        "with_p": True,
        "with_output": True,
        "nb_process": 1
    }

    def __init__(self, **options: dict[str, object]) -> None:
        self.fold: int = options.get("fold", self.DEFAULTS.get("fold"))
        self.svm_type: int = options.get('svm_type', self.DEFAULTS.get("svm_type"))
        self.c_range: GridRange = GridRange(*options.get('c_range', self.DEFAULTS.get('c_range')))
        self.g_range: GridRange = GridRange(*options.get('g_range', self.DEFAULTS.get('g_range')))
        self.p_range: GridRange = GridRange(*options.get('p_range', self.DEFAULTS.get('p_range')))
        self.grid_with_c: bool = options.get('with_c', self.DEFAULTS.get('with_c'))
        self.grid_with_g: bool = options.get('with_g', self.DEFAULTS.get('with_g'))
        self.grid_with_p: bool = options.get('with_p', self.DEFAULTS.get('with_p')) and self.svm_type in [3, 4]
        self.dataset = svmutil.svm_read_problem(options.get('dataset'))
        self.dataset_title: str = os.path.split(options.get('dataset'))[1]
        self.with_output: bool = options.get('with_output', self.DEFAULTS.get("with_output"))
        self.out_pathname: str = '{0}.out'.format(self.dataset_title if options.get('out_pathname') is None else options.get('out_pathname'))
        self.png_pathname: str = '{0}.png'.format(self.dataset_title if options.get('out_pathname') is None else options.get('out_pathname'))
        self.resume_pathname: str = options.get('resume_pathname')
        self.svm_options: str = " ".join(options.get('svm_options'))
        self.with_gnuplot: bool = self.grid_with_c and self.grid_with_g
        self.nb_process: int = options.get('nb_process', self.DEFAULTS.get("nb_process"))
        self.validate_options()

    def validate_options(self) -> None :
        if not any([self.grid_with_c, self.grid_with_g, self.grid_with_p]):
            raise ValueError('-c , -g and -p should not be disabled simultaneously')

    def evaluate_rate(self, rate: float, best_rate: float) -> bool:
        if self.rate_kind == 'rate':
            return rate > best_rate
        return rate < best_rate

    @property
    def rate_kind(self) -> str:
        return 'mse' if self.svm_type in [3, 4] else 'rate'


def redraw(db, best_param, options, to_file=False) -> None:
    gnuplot = pygnuplot.Gnuplot()

    if len(db) == 0: return
    begin_level = round(max(x.rate for x in db)) - 3
    step_size = 0.5
    # if newly obtained c, g, or cv values are the same,
    # then stop redrawing the contour.
    if all(x.log2c == db[0].log2c for x in db): return
    if all(x.log2g == db[0].log2g for x in db): return
    if all(x.rate == db[0].rate for x in db): return

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
    gnuplot.write("set xrange [{0}:{1}]\n".format(options.c_range.begin,options.c_range.end).encode())
    gnuplot.write("set yrange [{0}:{1}]\n".format(options.g_range.begin,options.g_range.end).encode())
    gnuplot.write(b"set contour\n")
    gnuplot.write("set cntrparam levels incremental {0},{1},100\n".format(begin_level,step_size).encode())
    gnuplot.write(b"unset surface\n")
    gnuplot.write(b"unset ztics\n")
    gnuplot.write(b"set view 0,0\n")
    gnuplot.write("set title \"{0}\"\n".format(options.dataset_title).encode())
    gnuplot.write(b"unset label\n")
    gnuplot.write("set label \"Best log2(C) = {0}  log2(gamma) = {1}  accuracy = {2}%\" \
                  at screen 0.5,0.85 center\n". \
                  format(best_param.log2c, best_param.log2g, best_param.rate).encode())
    gnuplot.write("set label \"C = {0}  gamma = {1}\""
                  " at screen 0.5,0.8 center\n".format(best_param.c, best_param.g).encode())
    gnuplot.write(b"set key at screen 0.9,0.9\n")
    gnuplot.write(b"splot \"-\" with lines\n")

    db.sort(key = lambda x:(x.log2c, -x.log2g))

    prevc = db[0].log2c
    for line in db:
        if prevc != line.log2c:
            gnuplot.write(b"\n")
            prevc = line.log2c
        gnuplot.write(f"{line.log2c} {line.log2g} {line.rate}\n".encode())
    gnuplot.write(b"e\n")
    gnuplot.write(b"\n") # force g back to prompt when term set failure
    gnuplot.flush()


def calculate_jobs(options: GridOption) -> tuple[Iterator[GridHyperParameter], Iterator[GridHyperParameter]]:
    c_seq = np.random.permutation(np.arange(*options.c_range))
    g_seq = np.random.permutation(np.arange(*options.g_range))
    p_seq = np.random.permutation(np.arange(*options.p_range))

    if not options.grid_with_c:
        c_seq = [None]
    if not options.grid_with_g:
        g_seq = [None]
    if not options.grid_with_p:
        p_seq = [None]

    jobs = map(lambda x: GridHyperParameter(*x), product(c_seq, g_seq, p_seq))
    resumed_jobs = tuple()

    if options.resume_pathname is None:
        return jobs, resumed_jobs

    with open(options.resume_pathname, 'r') as resume:
        resumed_jobs = map(
            lambda x: GridHyperParameter(*map(lambda y: float(y) if y else None, x)),
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
    return jobs, resumed_jobs


@ray.remote
def evaluate_svm_classifier(name: str, hyperparam: GridHyperParameter, options: GridOption) -> tuple[str, GridHyperParameter]:
    cmdline = ['-q', f'-s {options.svm_type}']

    if options.grid_with_c:
        cmdline.append(f'-c {hyperparam.c}')

    if options.grid_with_g:
        cmdline.append(f'-g {hyperparam.g}')

    if options.grid_with_p:
        cmdline.append(f'-p {hyperparam.p}')

    cmdline.append(f'-v {options.fold} {options.svm_options}')
    hyperparam.rate = svmutil.svm_train(*options.dataset, " ".join(cmdline))
    return (name, hyperparam)


def update_param(hyperparam: GridHyperParameter, best_hyperparam: GridHyperParameter, options: GridOption) -> GridHyperParameter:
    if (
        options.evaluate_rate( hyperparam.rate, best_hyperparam.rate )
        or (
            hyperparam.rate == best_hyperparam.rate
            and hyperparam.log2g == best_hyperparam.log2g
            and hyperparam.log2p == best_hyperparam.log2p
            and hyperparam.log2c < hyperparam.log2c
        )
    ):
        best_hyperparam = hyperparam

    return best_hyperparam


def find_parameters(**params: dict[str, object]) -> GridHyperParameter:

    def write_param(hyperparam: GridHyperParameter, best_hyperparam: GridHyperParameter, worker: str, resumed: bool, options: GridOption):
        stdout_str = [f'[{worker}] {" ".join(str(x) for x in [hyperparam.log2c, hyperparam.log2g, hyperparam.log2p] if x is not None)} {hyperparam.rate} (best']
        output_str = []
        if options.grid_with_c:
            stdout_str.append(f'c={best_hyperparam.c},')
            output_str.append(f'log2c={hyperparam.log2c}')

        if options.grid_with_g:
            stdout_str.append(f'g={best_hyperparam.g},')
            output_str.append(f'log2g={hyperparam.log2g}')

        if options.grid_with_p:
            stdout_str.append(f'p={best_hyperparam.p},')
            output_str.append(f'log2p={hyperparam.log2p}')

        stdout_str.append(f'{options.rate_kind}={best_hyperparam.rate})')
        logging.info(" ".join(stdout_str))
        if options.out_pathname and not resumed:
            output_str.append(f'{options.rate_kind}={hyperparam.rate}\n')
            result_file.write(" ".join(output_str))
            result_file.flush()

    options = GridOption(**params)
    jobs, resumed_jobs = calculate_jobs(options)

    workers = [evaluate_svm_classifier.remote('local', hyperparam, options) for hyperparam in jobs if hyperparam not in resumed_jobs]

    if options.with_output:
        if options.resume_pathname:
            result_file = open(options.out_pathname, 'a')
        else:
            result_file = open(options.out_pathname, 'w')
    db = []

    best_hyperparam = GridHyperParameter(None, None, None, float('+inf') if options.svm_type in [3, 4] else -1)
    for hyperparam in resumed_jobs:
        best_hyperparam = update_param(hyperparam, best_hyperparam ,options)
        write_param(hyperparam, best_hyperparam, 'resumed', True ,options)
        db.append(hyperparam)


    futs = [worker.future() for worker in workers]
    for fut in concurrent.futures.as_completed(futs):
        (worker, hyperparam) = fut.result()
        best_hyperparam = update_param(hyperparam, best_hyperparam ,options)
        write_param(hyperparam, best_hyperparam, worker, False, options)
        db.append(hyperparam)

    if options.with_gnuplot:
        redraw(db, best_hyperparam, options)
        redraw(db, best_hyperparam, options, to_file=True)

    if options.with_output:
        result_file.close()

    logging.info(f'{" ".join(map(str, (best_hyperparam.c, best_hyperparam.g, best_hyperparam.p)))} {best_hyperparam.rate}')

    return best_hyperparam


@click.command()
@click.argument('dataset', type=click.Path(exists=True))
@click.option('--svm-type', 'svm_type', default=GridOption.DEFAULTS.get("svm_type"), help="""\b
set type of SVM (default 0):
0 -- C-SVC (multi-class classification)
1 -- nu-SVC (multi-class classification)
2 -- one-class SVM
3 -- epsilon-SVR (regression)
4 -- nu-SVR (regression)""")
@click.option('--log2c', 'c_range', nargs=3, default=GridOption.DEFAULTS.get('c_range'), help='c_range = 2^{begin,...,begin+k*step,...,end}')
@click.option('--log2g', 'g_range', nargs=3, default=GridOption.DEFAULTS.get('g_range'), help='g_range = 2^{begin,...,begin+k*step,...,end}')
@click.option('--log2p', 'p_range', nargs=3, default=GridOption.DEFAULTS.get('p_range'), help='p_range = 2^{begin,...,begin+k*step,...,end}')
@click.option('--c/--no-c', 'with_c', default=GridOption.DEFAULTS.get('with_c'), help='Disable the usage of log2c')
@click.option('--g/--no-g', 'with_g', default=GridOption.DEFAULTS.get('with_g'), help='Disable the usage of log2g')
@click.option('--p/--no-p', 'with_p', default=GridOption.DEFAULTS.get('with_p'), help='Disable the usage of log2p')
@click.option('-v', 'fold', type=int, default=GridOption.DEFAULTS.get('fold'), help='n fold validation')
@click.option('--out-pathname', type=click.Path(), help='set output file path and name')
@click.option('--out/--no-out', 'with_output', default=GridOption.DEFAULTS.get('with_output'), help='Disable out file')
@click.option('--resume', 'resume_pathname', type=click.Path(exists=True), help='Existing old out file to resume from.')
@click.option('--svm-options', multiple=True, help='additionals svm options')
@click.option('--nb-process', default=GridOption.DEFAULTS.get('nb_process'))
def main(**params: dict[str, object]):
    ray.init()
    find_parameters(**params)
