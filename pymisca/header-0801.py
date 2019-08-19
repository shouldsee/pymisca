# NCORE=4
import pymisca.ext as pyext
# del NCORE
pyext.mpl__setBackend('agg')
with pyext.Suppress():

#     import synotil
#     with pyext.getPathStack([pyext.module__getPath('synotil')]):
#     #     ! cat 'headers/header__import.py'
#         execfile('headers/header__import.py')
    pyext.self__install()
    
    import pymisca.vis_util as pyvis
    import pymisca.module_wrapper 
    plt = pyvis.plt
    pd = pyext.pd
    np = pyext.np


    # def _start():
    if 1:
        with pyext.getAttoDirectory(['memory'],force=1) as stack:
            if not pyext.file__notEmpty(stack.d/'DB_RES.json'):
                DB_RES = [None] *1000
                pyext.printlines([pyext.ppJson(DB_RES)],'DB_RES.json')
            if not pyext.file__notEmpty(stack.d/'DATA.npy'):
                DATA = {}
                np.save('DATA.npy',DATA)

            DB_RES = pyext.readData('DB_RES.json')
            DATA =pyext.readData('DATA.npy',allow_pickle=True).tolist()



    def _end():
        with pyext.getAttoDirectory(['memory']):
            pyext.printlines([pyext.ppJson(DB_RES)],'DB_RES.json')
            np.save('DATA.npy',DATA)
        pyext.shellexec("python2 compile.py &>COMPILE.log && echo DONE")
        pyext.MDFile('OUT.md.html')        


    #### plotting
#     plt.set_cmap('Greens');
#     import matplotlib.pyplot as plt
    plt.rcParams['image.cmap'] = 'Greens'

    
    #### dataFrame display
    pd.set_option('display.max_colwidth', 500)

    # df = DATA['uniTab']
    # df.head()

    # with pyext.getAttoDirectory([DB_RES[0]['LAST_DIR']]):
    #     np.save('OUTPUT/DATA.npy',DATA)    
    # df.query('Gene_Gene_Name.duplicated()')