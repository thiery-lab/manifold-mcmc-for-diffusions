import sympy as sym
from sympy.printing.pycode import NumPyPrinter
import numpy as np


def to_tuple(lst):
    """Recursively convert nested lists to nested tuples."""
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


def flatten(seq):
    """Flatten a sequence of arrays to a flat list."""
    flattened = []
    shapes = []
    for el in seq:
        if isinstance(el, sym.Array):
            size = np.prod(el.shape)
            flattened += el.reshape(size).tolist()
            shapes.append((el.shape, size))
        else:
            flattened.append(el)
            shapes.append(())
    def unflatten(flattened):
        unflattened = []
        i = 0
        for s in shapes:
            if s == ():
                unflattened.append(flattened[i])
                i += 1
            else:
                shape, size = s
                unflattened.append(sym.Array(flattened[i:i+size], shape))
                i += size
        return unflattened
    return flattened, unflatten


class SymbolTuple(tuple):
    """Tuple of symbols with common name-prefix."""
    
    def __new__(cls, name, length):
        return tuple.__new__(
            cls, sym.symbols([f'{name}[{i}]' for i in range(length)]))
    
    def __init__(self, name, length):
        self.name = name
        self.length = length
    
    def __str__(self):
        return self.name
             

class TupleArrayPrinter(NumPyPrinter):
    """SymPy printer which uses nested tuples for array literals.
    
    Rather than printing array-like objects as numpy.array calls with a nested
    list argument (which is not numba compatible) a nested tuple argument is
    used instead.
    """
    
    def _print_arraylike(self, expr):
        exp_str = self._print(to_tuple(expr.tolist()))
        return f'numpy.array({exp_str}, dtype=numpy.float64)'
    
    _print_NDimArray = _print_arraylike
    _print_DenseNDimArray = _print_arraylike
    _print_ImmutableNDimArray = _print_arraylike
    _print_ImmutableDenseNDimArray = _print_arraylike
    _print_MatrixBase = _print_arraylike


def generate_code(inputs, exprs, func_name='generated_function', printer=None):
    """Generate code for a Python function from symbolic expression(s)."""
    if printer is None:
        printer = TupleArrayPrinter()
    if not isinstance(exprs, list) and not isinstance(exprs, tuple):
        exprs = [exprs]
    flat_exprs, unflatten = flatten(exprs)
    intermediates, flat_outputs = sym.cse(
        flat_exprs, symbols=sym.numbered_symbols('_i'), optimizations='basic')
    outputs = unflatten(flat_outputs)
    code = f'def {func_name}({", ".join([str(sym) for sym in inputs])}):\n    '
    code += '\n    '.join([f'{printer.doprint(i[0])} = {printer.doprint(i[1])}' 
                           for i in intermediates])
    code += '\n    return (\n        '
    code += ',\n        '.join([f'{printer.doprint(output)}' 
                                for output in outputs])
    code += '\n    )'
    return code


def generate_func(inputs, exprs, func_name='generated_function', printer=None,
                  numpy_module=None):
    """Generate a Python function from symbolic expression(s)."""
    code = generate_code(inputs, exprs, func_name, printer)
    namespace = {'numpy': np if numpy_module is None else numpy_module}
    exec(code, namespace)
    func = namespace[func_name]
    func.__doc__ = f'Automatically generated {func_name} function.\n\n{code}'
    return func