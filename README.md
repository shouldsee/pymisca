# pymisca

Miscellaneous utilities and visualisation snippets oriented towards functional programming

## Installation

Make sure you have git installed by

```sh
sudo apt-get install -y git
```

Install `pymisca` dev-version.

```sh
pip2 install git+https://github.com/shouldsee/pymisca -U
```

### For dev and test

```sh
git clone --depth 1 https://github.com/shouldsee/pymisca
pip install -r requirements.txt ## --user
python setup.py install ## --user  for local install
```
### Examples

Make sure you are in the git source tree

```sh
python2 examples/run-pipeline-0608.py
```

### [Deprecated_function] Example Usage 

Run a pipeline and output to "$PWD/results"

```bash
python2 test/test__pipeline.py
```

Filename: `test/test__pipeline.py`

```python
#!/usr/bin/env python
import pymisca.header as pyheader
import pymisca.ext as pyext

### make sure $BASE is set within environment, or default to current direct$
pyheader.base__check() 

suc,res = pyext.job__baseScript('test/pipeline-src/init__seq.sh',silent=1)
assert suc,res

suc,res = pyext.job__baseScript('test/pipeline-src/histogram.py',silent=1)
assert suc,res

### we can also output to plots/ instead of results/ 
suc,res = pyext.job__baseScript('test/pipeline-src/histogram.py',silent=1,
                            prefix='plots')
assert suc,res

### job__baseScript calls job__script() internally, where all keywords are specified
### Documentation welcomed

print ('[DONE]')
```
