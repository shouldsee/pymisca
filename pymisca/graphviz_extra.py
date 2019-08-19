import sys
try:
    from graphviz import backend as _backend
    from graphviz.backend import FORMATS
except Exception as e:
    sys.stderr.write("%s\n"%e)
import os
import pymisca.tree
def graphviz__render(self,filepath, format=None,**kw):
    filepath = os.path.realpath(filepath)
    bname = filepath.rsplit('.',1)[0]
    assert format is None
    #### fragile
    sp = filepath.rsplit('.',1)
    assert len(sp) ==2
    bname,format = sp        
    filepath = type(filepath)(bname)
    assert format in FORMATS,"Unknown format: %s" % format
#     with pymisca.tree.getPathStack([os.path.dirname(filepath)]):
    rendered = _backend.render( self._engine, format, bname,
#                                cwd = os.path.dirname(filepath),
                               **kw)
    return rendered

gv__render = graphviz__render

if 0:
    # from graphviz.backend import command,os,run
    import pymisca.header as pyext
    from graphviz.backend import *
    @pyext.setAttr
    def command(engine, format_, filepath=None, renderer=None, formatter=None):
        """Return args list for ``subprocess.Popen`` and name of the rendered file."""
        if formatter is not None and renderer is None:
            raise RequiredArgumentError('formatter given without renderer')

        if engine not in ENGINES:
            raise ValueError('unknown engine: %r' % engine)
        if format_ not in FORMATS:
            raise ValueError('unknown format: %r' % format_)
        if renderer is not None and renderer not in RENDERERS:
            raise ValueError('unknown renderer: %r' % renderer)
        if formatter is not None and formatter not in FORMATTERS:
            raise ValueError('unknown formatter: %r' % formatter)

        output_format = [f for f in (format_, renderer, formatter) if f is not None]
        cmd = [engine, '-T%s' % ':'.join(output_format)]
        rendered = None

        if filepath is not None:
            cmd.extend(['-O', filepath])
            suffix = '.'.join(reversed(output_format))
            rendered = '%s.%s' % (filepath, suffix)

        return cmd, rendered

    @pyext.setAttr(_backend)
    def render(engine, format, filepath, renderer=None, formatter=None, quiet=False):
        """Render file with Graphviz ``engine`` into ``format``,  return result filename.

        Args:
            engine: The layout commmand used for rendering (``'dot'``, ``'neato'``, ...).
            format: The output format used for rendering (``'pdf'``, ``'png'``, ...).
            filepath: Path to the DOT source file to render.
            renderer: The output renderer used for rendering (``'cairo'``, ``'gd'``, ...).
            formatter: The output formatter used for rendering (``'cairo'``, ``'gd'``, ...).
            quiet (bool): Suppress ``stderr`` output from the layout subprocess.
        Returns:
            The (possibly relative) path of the rendered file.
        Raises:
            ValueError: If ``engine``, ``format``, ``renderer``, or ``formatter`` are not known.
            graphviz.RequiredArgumentError: If ``formatter`` is given but ``renderer`` is None.
            graphviz.ExecutableNotFound: If the Graphviz executable is not found.
            subprocess.CalledProcessError: If the exit status is non-zero.

        The layout command is started from the directory of ``filepath``, so that
        references to external files (e.g. ``[image=...]``) can be given as paths
        relative to the DOT source file.
        """
        dirname, filename = os.path.split(filepath)
        cmd, rendered = command(engine, format, filename, renderer, formatter)
        if dirname:
            cwd = dirname
            rendered = os.path.join(dirname, rendered)
        else:
            cwd = None
        run(cmd, capture_output=True, cwd=cwd, check=True, quiet=quiet)
        return rendered