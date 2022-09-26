import numpy as np

def export_csv(filename, *arg, precision=4, delimit=','):
    """
    Exportiert Arrays als .csv Datei
    :param delimit: delimiter 
    :param filename: filename
    :param precision: number of decimal places after which the number is truncated
    :return: null
    """
    p = '%.{k}f'.format(k=precision)
    ps = []
    for arguments in arg:
        ps.append(p)
    try:
        np.savetxt(
            filename,
            np.c_[arg],
            fmt=ps,
            delimiter=delimit,
            newline='\n',
            # footer='end of file',
            # comments='# ',
            # header='X , MW'
        )
        # print("[i] csv saved to {}".format(filename))
    except TypeError:
        print("TypeError occured")
        for arguments in arg:
            print(arguments)
