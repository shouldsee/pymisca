from pymisca.lazydicts.stepper import Stepper
import pymisca.atto_string
import pymisca.header as pyext

Copier = Stepper.redefine()
Copier['kw'] = {}
Copier['OUTDIR'] = '/tmp'
Copier['INPUTDIR'] = None
Copier['DRY'] = 0
Copier['FORCE'] = 0

Copier['_copier'] = lambda self,k,kw,OUTDIR,FORCE,INPUTDIR,DRY: pymisca.atto_string.CopyTo(
    **{k:v for  k,v in kw.items()+[('OUTDIR',OUTDIR),
                                   ('INPUTDIR',INPUTDIR),
                                   ('DRY',DRY),
                                   ('force',FORCE)]}
)
# del Copier['callback/init']
@pyext.setItem(Copier,'callback/init')
def _func(self, key, _copier):
    self.values['callback/main'] = pyext.PlainFunction( 
        lambda x: (_copier.call_tuple(x[:2]),) + x[2:] )
# Copier['callback']
    
