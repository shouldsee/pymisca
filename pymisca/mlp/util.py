import pymisca.ext as pyext
import numpy as np      
import matplotlib as mpl


    
def getMasks__Random(strides=(2,2),size=(4,4),debug=0,maskSize=None,seed=None):
#     lst = []
    np.random.seed(seed)
    maskAll = np.zeros((8,8),dtype='bool')
    if maskSize is None:
        maskSize = np.prod(size)
    for i in range(50):
#         nodes = zip(*np.where(~maskAll))
        if np.all(maskAll):
            break
        nodes = np.transpose(np.where(~maskAll))
        if i ==0:
#             toAddIdx = np.random.randint(len(nodes),size=maskSize,)
#             toAdd = nodes[toAddIdx]
#             toAddIdx = np.random.choice(range(len(nodes)),size=maskSize,replace=False)
            toAdd = pyext.list__randomChoice(nodes,size=maskSize,replace=False)
            maskAdd = toAdd
#             mask[toAdd.T] = True
            lastAdd = pyext.list__randomChoice(toAdd,size=maskSize//2,replace=False)
        else:
            toAdd = pyext.list__randomChoice(nodes,size=maskSize//2,replace=False)
            maskAdd = np.vstack([lastAdd,toAdd])
            lastAdd = toAdd
        mask = np.zeros((8,8),dtype='bool')
        mask[ [tuple(x) for x in maskAdd.T] ] = True
        maskAll |= mask
        print maskAll.sum()
        yield mask,0,0
            




def getMasks__CNN(strides=(2,2),size=(4,4),debug=0):
#     lst = []

    i = 0 
    j = 0
#     strides = (2,2)
#     size = (4,4)
    # datas += [data]
#     datas = []
    maskAll = np.zeros((8,8),dtype='bool')
    for i in range(8/strides[0]):
        i = i*strides[0]
        for j in range(8/strides[0]):
            j = j*strides[1]
    #         print (i,j)
            mask = np.zeros((8,8))
            mask[i:i+size[0],j:j+size[1]] = 1
            mask = mask.astype(bool)
            if np.sum(mask)!=np.prod(size):
                continue
            maskAll |= mask
            if debug:
                print maskAll.sum()
                
            yield mask,i,j
            
if __name__ == '__main__':
    def test():
        import pymisca.vis_util as pyvis
        plt = pyvis.plt

        list(getMasks__Random())[:1]            

        getMasks = getMasks__CNN        
        ax = plt.gca()
        cmap = pyvis.get__cmap__defaultCycle()
        size=(4,4)
        for mask,i,j in getMasks(size=size):
            _i = i+np.random.random()*0.05
            _j = j+np.random.random()*0.05
            color =  cmap(i+j)
            patch = mpl.patches.Rectangle(xy=(_i,_j),
                                          width=size[0] - 0.5,height=size[1]-0.5,alpha=0.5,facecolor=color,
                                          linewidth=2.2,edgecolor='black',
                                          )
            ax.add_patch(patch)        

        ax.autoscale()
    test()

# plt.plot()