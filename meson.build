project('pyaceqd', 'c', version: '0.0.1')

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
