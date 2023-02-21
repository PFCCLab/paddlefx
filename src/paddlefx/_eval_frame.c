#include <Python.h>
#include <frameobject.h>
#include <pystate.h>

// see https://bugs.python.org/issue35886
#if PY_VERSION_HEX >= 0x03080000
#define Py_BUILD_CORE
#include "internal/pycore_pystate.h"
#undef Py_BUILD_CORE
#endif

#define unlikely(x) __builtin_expect((x), 0)

static Py_tss_t eval_frame_callback_key = Py_tss_NEEDS_INIT;

inline static PyObject *eval_frame_callback_get(void) {
  void *result = PyThread_tss_get(&eval_frame_callback_key);
  if (unlikely(result == NULL)) {
    Py_RETURN_NONE;
  } else {
    return (PyObject *)result;
  }
}

inline static void eval_frame_callback_set(PyObject *obj) {
  PyThread_tss_set(&eval_frame_callback_key, obj);
}

inline static PyObject *eval_frame_default(PyThreadState *tstate,
                                           PyFrameObject *frame,
                                           int throw_flag) {
  return _PyEval_EvalFrameDefault(frame, throw_flag);
}

static inline PyObject *call_callback(PyObject *callable, PyObject *frame,
                                      long cache_len) {
  PyObject *args = Py_BuildValue("(Ol)", frame, cache_len);
  PyObject *result = PyObject_CallObject(callable, args);
  Py_DECREF(args);
  return result;
}

static PyObject *_custom_eval_frame(PyThreadState *tstate, PyFrameObject *frame,
                                    int throw_flag, PyObject *callback) {
  if (PyFrame_FastToLocalsWithError(frame) < 0) {
    return NULL;
  }

  if (callback == Py_False) {
    return eval_frame_default(tstate, frame, throw_flag);
  }

  eval_frame_callback_set(Py_None);

  PyObject *args = Py_BuildValue("(O)", frame);
  PyObject *result = PyObject_CallObject(callback, args);
  if (result == NULL) {
    // internal exception
    return NULL;
  } else if (result != Py_None) {
    Py_DECREF(result);
    eval_frame_callback_set(callback);
    return eval_frame_default(tstate, frame, throw_flag);
  } else {
    Py_DECREF(result);
    // Re-enable custom behavior
    eval_frame_callback_set(callback);
    return eval_frame_default(tstate, frame, throw_flag);
  }
}

static PyObject *_custom_eval_frame_shim(PyThreadState *tstate,
                                         PyFrameObject *frame, int throw_flag) {
  PyObject *callback = eval_frame_callback_get();

  if (callback == Py_None) {
    return eval_frame_default(tstate, frame, throw_flag);
  }

  return _custom_eval_frame(tstate, frame, throw_flag, callback);
}

static PyObject *custom_eval_frame_shim(PyFrameObject *frame, int throw_flag) {
  PyThreadState *tstate = PyThreadState_GET();
  return _custom_eval_frame_shim(tstate, frame, throw_flag);
}

static int active_dynamo_threads = 0;

static PyObject *increment_working_threads(PyThreadState *tstate) {
  active_dynamo_threads = active_dynamo_threads + 1;
  if (active_dynamo_threads > 0) {
    if (tstate->interp->eval_frame != &custom_eval_frame_shim) {
      // First call
      tstate->interp->eval_frame = &custom_eval_frame_shim;
    }
  }
  Py_RETURN_NONE;
}

static PyObject *decrement_working_threads(PyThreadState *tstate) {
  if (active_dynamo_threads > 0) {
    active_dynamo_threads = active_dynamo_threads - 1;
    if (active_dynamo_threads == 0) {
      if (tstate->interp->eval_frame != &_PyEval_EvalFrameDefault) {
        // First call
        tstate->interp->eval_frame = &_PyEval_EvalFrameDefault;
      }
    }
  }
  Py_RETURN_NONE;
}

static PyObject *set_eval_frame(PyObject *new_callback, PyThreadState *tstate) {
  // Change the eval frame callback and return the old one
  PyObject *old_callback = eval_frame_callback_get();

  // owned by caller
  Py_INCREF(old_callback);

  if (old_callback != Py_None && new_callback == Py_None) {
    decrement_working_threads(tstate);
  } else if (old_callback == Py_None && new_callback != Py_None) {
    increment_working_threads(tstate);
  }

  Py_INCREF(new_callback);
  Py_DECREF(old_callback);

  eval_frame_callback_set(new_callback);

  return old_callback;
}

static PyObject *set_eval_frame_py(PyObject *dummy, PyObject *args) {
  PyObject *callback = NULL;
  if (!PyArg_ParseTuple(args, "O:callback", &callback)) {
    return NULL;
  }
  if (callback != Py_None && callback != Py_False &&
      !PyCallable_Check(callback)) {
    PyErr_SetString(PyExc_TypeError, "expected a callable");
    return NULL;
  }
  return set_eval_frame(callback, PyThreadState_GET());
}

static PyMethodDef foo_methods[] = {
    {"set_eval_frame", set_eval_frame_py, METH_VARARGS, NULL},
    {NULL, NULL},
};

static PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "_eval_frame",
    "_eval_frame doc",
    -1,
    foo_methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit__eval_frame(void) {
  int result = PyThread_tss_create(&eval_frame_callback_key);

  Py_INCREF(Py_None);
  eval_frame_callback_set(Py_None);

  return PyModule_Create(&_module);
}
