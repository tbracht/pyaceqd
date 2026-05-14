project('pyaceqd', 'c', 'cpp', version: '0.0.1')

py = import('python').find_installation(pure: false)

install_subdir(
  'pyaceqd',
  install_dir: py.get_install_dir(),
  exclude_directories: ['__pycache__', 'pyaceqd.egg-info']
)

ext_suffix = run_command(
  py, '-c', 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")',
  check: true
).stdout().strip()

build_dir = meson.current_build_dir()

timebin_out = join_paths(build_dir, 'timebin_tl' + ext_suffix)
timebin_cmd = [
  py, '-m', 'numpy.f2py', '-c', '--f90flags="-O3 -fopenmp"',
  '-m', 'timebin_tl', files('pyaceqd/timebin/timebin_tl.f90'),
  '-lopenblas', '-lgomp'
]
timebin_build = run_command(
  py, '-c',
  'import os,subprocess,sys; os.chdir(sys.argv[1]); subprocess.check_call(sys.argv[2:])',
  build_dir,
  timebin_cmd,
  check: false
)
if timebin_build.returncode() == 0
  install_data(
    timebin_out,
    install_dir: join_paths(py.get_install_dir(), 'pyaceqd', 'timebin')
  )
else
  message('Skipping timebin_tl (f2py build failed).')
endif

propagate_out = join_paths(build_dir, 'propagate_tau_module' + ext_suffix)
propagate_cmd = [
  py, '-m', 'numpy.f2py', '-c', '--f90flags="-O3 -fopenmp"',
  '-m', 'propagate_tau_module', files('pyaceqd/two_time/propagate_tau.f90'),
  '-lopenblas', '-lgomp'
]
propagate_build = run_command(
  py, '-c',
  'import os,subprocess,sys; os.chdir(sys.argv[1]); subprocess.check_call(sys.argv[2:])',
  build_dir,
  propagate_cmd,
  check: false
)
if propagate_build.returncode() == 0
  install_data(
    propagate_out,
    install_dir: join_paths(py.get_install_dir(), 'pyaceqd', 'two_time')
  )
else
  message('Skipping propagate_tau_module (f2py build failed).')
endif

# C++ nanobind module: TimeLocalPhononfree / calc_G1
nb_pkg = run_command(
  py, '-c', 'import nanobind, os; print(os.path.dirname(nanobind.__file__))',
  check: true
).stdout().strip()
nb_inc     = nb_pkg + '/include'
nb_src_dir = nb_pkg + '/src'
nb_tsl     = nb_pkg + '/ext/robin_map/include'
nb_py_inc  = run_command(
  py, '-c', 'import sysconfig; print(sysconfig.get_path("include"))',
  check: true
).stdout().strip()

propagate_nb = py.extension_module(
  'propagate_tau_nb_module',
  [
    'pyaceqd/two_time/propagate_tau_nb.cpp',
    nb_src_dir + '/nb_combined.cpp',
  ],
  cpp_args: [
    '-O3', '-fopenmp', '-std=c++17',
    #'-O3', '-g', '-fno-omit-frame-pointer', '-fopenmp', '-std=c++17',
    '-I' + nb_inc,
    '-I' + nb_src_dir,
    '-I' + nb_tsl,
    '-I' + nb_py_inc,
    '-I/usr/include/x86_64-linux-gnu/openblas-pthread',
  ],
  link_args: ['-fopenmp', '-lopenblas'],
  install: true,
  install_dir: join_paths(py.get_install_dir(), 'pyaceqd', 'two_time'),
)

# Eigen-based nanobind module: TimeLocalPhononfreeEigen / calc_G1
eigen_inc = '/usr/include/eigen3'

propagate_nb_eigen = py.extension_module(
  'propagate_tau_nb_eigen_module',
  [
    'pyaceqd/two_time/propagate_tau_nb_eigen.cpp',
    nb_src_dir + '/nb_combined.cpp',
  ],
  cpp_args: [
    '-O3', '-fopenmp', '-std=c++17',
    '-I' + nb_inc,
    '-I' + nb_src_dir,
    '-I' + nb_tsl,
    '-I' + nb_py_inc,
    '-I' + eigen_inc,
  ],
  link_args: ['-fopenmp'],
  install: true,
  install_dir: join_paths(py.get_install_dir(), 'pyaceqd', 'two_time'),
)
