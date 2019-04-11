import collections

class TreeDict_0410(collections.OrderedDict):    
    '''
    TreeDict[['a','b','c']] = 
    '''
    def __init__(self,  *args, **kwargs):
        self._sep = '.'
        self.debug = 0
        super(TreeDict, self).__init__(*args,**kwargs)

    def set__sep(self,sep):
        self._sep = sep
    @staticmethod
    def _key_sanitise(key):
        key0 = key
        if not isinstance(key,tuple):
            if isinstance(key,list):
                key = tuple(key)
            else:
                ### wrap non-tuple as a singleton
                key = (key,)
        assert isinstance(key,tuple)
#         assert len(key),'key cannot be empty:%s' %(key0,)
        return key
            
# def __init__(self, *args, **kwargs):
#     super(MyUpdateDict, self).__init__() #<-- HERE call to super __init__
#         self.update(*args, **kwargs)        
    
    def getRoute(self, key):
        route = key.split(self._sep)
        route = tuple(route)
        if self.debug:
            print (route)
        return route
    
    def new__node(self,parent):
        return collections.OrderedDict() 
    #         self.__dict__ = self
    def get(self,key,default=None):
        key = self._key_sanitise(key)
#         root = res = parent = super(TreeDict, self)
#         parent = self
        parent = self
        if self.debug:
            print ('[get.key]',key)
#             assert isinstance(key,tuple)
#                 assert 0
#         print key
        for i, _key in enumerate(key):
            if parent is self:
                getter = super(TreeDict,parent).get
            else:
                getter = parent.get          
            child = getter(_key,None)
                
            if child is None:
#                 if parent is self:
#                     parent[(_key,)] = 
                child = parent[_key] = self.new__node(parent)
            parent = child
            
        return parent
    

        
    def __setitem__(self,key,value):
        key = self._key_sanitise(key)
#         assert len(key)

        parent = self.get(key[:-1])
        if self.debug:
            print '[parent]',parent
        if parent is self:
            setter = super(TreeDict,parent).__setitem__
        else:
            setter = parent.__setitem__
        setter(key[-1],  value)
        
    def __delitem__(self,key):
        key = self._key_sanitise(key)
#         assert len(key)
        parent = self.get(key[:-1])
        
        if parent is self:
            deller = super(TreeDict,parent).__delitem__
        else:
            deller = parent.__delitem__
        deller(key[-1])
    def getFlat(self, key, default=None):
        '''
        self.getFlat('a.b.c') == self[['a','b','c']]
        '''
        key = self.getRoute(key)
        return self.get( key,default=default)
    
    def setFlat(self, key, value):
        '''
        shortcut to a node
        '''
        if self.debug:
            print ('self.setFlat("%s","%s")'%(key,value))
        assert key,'[key must be a string]%s'%key
        key = self.getRoute(key)
        self[key] = value
#         self.__setitem__(key,value)
        
    @classmethod
    def from_flatDict(cls, d, sep=None):
        '''
        transform a flat 
        '''
        tdict = cls()
        if sep is not None:
            tdict.set__sep(sep)
        it = d.iteritems()                
        for k, v in it:
#             print k
            tdict.setFlat(k,v)
        return tdict

    

class TreeDict(collections.OrderedDict):
    """Implementation of perl's autovivification feature."""
    def __init__(self,*a,**kw):
        super(TreeDict,self).__init__(*a,**kw)
        self._sep='.'
	self._parent = None
        
    def __missing__(self, key):
        '''
        An instrumental symplification taken from 
        https://stackoverflow.com/a/6781411/8083313
        Original (probably) SO:
        https://stackoverflow.com/q/635483/8083313
        '''
        value = self[key] = type(self)()
        return value
    def set__sep(self,sep):
        self._sep=sep

    def walk(self, route, ):
        d = self
        for key in route:
            res = d[key]
            if isinstance(res,dict):
                if not isinstance(res,TreeDict):
                    #### cast any dict-like object to a TreeDict
                    d[key] = res = type(self)(res)
            d = res
        return d
        
    def getFlatLeaf(self, flatPath):
        route = flatPath.split(self._sep)
        return self.walk(route)
        
    def setFlatLeaf(self,flatPath, value):
        route = flatPath.split(self._sep)
        d = self.walk(route[:-1])
        d[route[-1]] = value
        
    @classmethod
    def from_flatPathDict(cls, flatPathDict,sep=None):
        self = cls()
        if sep is not None:
            self.set__sep(sep)
        for k,v in flatPathDict.items():
            self.setFlatLeaf( k, v)
        return self