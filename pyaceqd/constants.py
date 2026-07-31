import os

hbar = 0.6582119569  # meV*ps
c_light = 299.792e3  # nm/ps
temp_dir = ""  # Directory for temporary files
kB = 0.08617333262  # meV/K

# Global default directory for reusable process tensor (PT) files.
# Can be set once per process via pyaceqd.set_pt_dir(...) / pyaceqd.constants.set_pt_dir(...),
# or pre-set via the PYACEQD_PT_DIR environment variable.
pt_dir = os.environ.get("PYACEQD_PT_DIR", "")


def set_pt_dir(path, create=False):
    """
    Globally set the default directory used for process tensor (PT) files.

    Any GeneralSystemACE-based class instantiated afterwards without an explicit
    pt_dir argument will use this directory. Call this once at the top of a script, e.g.:

        import pyaceqd
        pyaceqd.set_pt_dir("/path/to/shared/pt_files")

    Parameters
    ----------
    path : str
        Directory to use for PT files. Use "" to reset to the current working directory.
    create : bool, optional
        If True, create the directory (including parents) if it does not exist yet.
        If False (default), raise a NotADirectoryError when the directory is missing.
    """
    global pt_dir
    if path:
        if not os.path.isdir(path):
            if create:
                os.makedirs(path, exist_ok=True)
            else:
                raise NotADirectoryError(
                    "pt_dir '{}' does not exist. Create it first, or call set_pt_dir(path, create=True).".format(path)
                )
    pt_dir = path


def get_pt_dir():
    """Return the currently configured global default PT directory."""
    return pt_dir