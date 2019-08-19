def minidom__iter(parent):
    '''
    return all nodes of a minidom.Node() instance
    '''
    _this_func = minidom__walk
    for e in parent.childNodes:
        yield e
        for e in _this_func(e):
            yield e
            
minidom__walk = minidom__iter
def minidom__iterElementByAttributeValue(dom,attr,value):
    for e in minidom__iter(dom):
        if e.nodeType == e.ELEMENT_NODE:
            if e.getAttribute(attr)==value:
                yield e            
