#include <Python.h>
#include <stdio.h>

static PyObject *bar(PyObject *module, PyObject *args) {
  printf("\ncall foo.bar\n");
  Py_RETURN_NONE;
}

static PyMethodDef foo_methods[] = {{
                                        "bar",
                                        bar,
                                        METH_NOARGS,
                                        "",
                                    },
                                    {
                                        NULL,
                                        NULL,
                                    }};

static PyModuleDef _foomodule = {
    PyModuleDef_HEAD_INIT,
    "foo",
    "foo doc",
    -1,
    foo_methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit__eval_frame(void) { return PyModule_Create(&_foomodule); }
