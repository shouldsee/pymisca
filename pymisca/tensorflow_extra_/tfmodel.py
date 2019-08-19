import pymisca.numpy_extra as pynp
np = pynp
import pymisca.models
import tensorflow as tf
class tfModel(pymisca.models.BaseModel):
    
    def __init__(self,sess = None, 
                input=  None,
                output= None):
        self.sess = sess
        self.input = input
        self.output = output
        super(tfModel, self).__init__()        
        pass
    
    def save(self,ofname='testSave'):
        res = tf.saved_model.simple_save(
           self.sess,
           ofname,
           inputs={'input': self.input},
           outputs={'output':self.output})
        return ofname
#         pass

    
    def load(self, fname,
        input_key = 'input',
        output_key = 'output',
):
#         sess = self.sess
        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        if len(fname.split('/')) == 1:
            fname = './'+fname
        meta_graph_def = tf.saved_model.loader.load(
                   self.sess,
                  [tf.saved_model.tag_constants.SERVING],
                  fname)
        
        signature = meta_graph_def.signature_def
        
        x_tensor_name = signature[signature_key].inputs[input_key].name
        y_tensor_name = signature[signature_key].outputs[output_key].name

        self.input  = sess.graph.get_tensor_by_name(x_tensor_name)
        self.output = sess.graph.get_tensor_by_name(y_tensor_name)          
        pass
    
    def _input2output(self, inputData,log = 1):
        res = logP = self.sess.run( 
            self.output, 
            {self.input:
             inputData})
        if not log:
            res = P = np.exp(logP)
        return res    
    
    def _predict_proba(self,inputData,log = 1):
        res = logP = self._input2output(inputData)
        if not log:
            res = P = np.exp(logP)
        return res
        
#     def predict( self, ):
#         res = self
#         return res
# def 
