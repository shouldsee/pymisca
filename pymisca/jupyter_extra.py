def nbjson__iterNodes__OutputData(d):
    for _cell in d['cells']:
        ob = _cell.get('outputs') or [{}]
        for _ob in ob:
            _data = _ob.get('data',{})
            if _data:
                for _dnkv in _data.iteritems():
                    yield ( _dnkv, _data, _ob, _cell)