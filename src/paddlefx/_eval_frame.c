#include <Python.h>
#include <code.h>
#include <frameobject.h>
#include <object.h>
#include <pystate.h>

// NOTE: This file is a simplified version of
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/dynamo/eval_frame.c
// https://github.com/pytorch/torchdynamo/blob/1.13/torchdynamo/_eval_frame.c

// see https://bugs.python.org/issue35886
#if PY_VERSION_HEX >= 0x03080000
#define Py_BUILD_CORE
#include "internal/pycore_pystate.h"
#undef Py_BUILD_CORE
#endif

#ifdef PADDLEFX_DEBUG

#define DEBUG_CHECK(cond) CHECK(cond)
#define DEBUG_NULL_CHECK(val) NULL_CHECK(val)
#define DEBUG_TRACE(msg, ...)                                                  \
  fprintf(stderr, "TRACE[%s:%d] " msg "\n", __func__, __LINE__, __VA_ARGS__)
#define DEBUG_TRACE0(msg)                                                      \
  fprintf(stderr, "TRACE[%s:%d] " msg "\n", __func__, __LINE__)

#else

#define DEBUG_CHECK(cond)
#define DEBUG_NULL_CHECK(val)
#define DEBUG_TRACE(msg, ...)
#define DEBUG_TRACE0(msg)

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
#if PY_VERSION_HEX >= 0x03090000
  if (tstate == NULL) {
    tstate = PyThreadState_GET();
  }
  return _PyEval_EvalFrameDefault(tstate, frame, throw_flag);
#else
  return _PyEval_EvalFrameDefault(frame, throw_flag);
#endif
}

inline static PyObject *eval_custom_code(PyThreadState *tstate,
                                         PyFrameObject *frame,
                                         PyCodeObject *code, int throw_flag) {
  Py_ssize_t ncells = 0;
  Py_ssize_t nfrees = 0;
  Py_ssize_t nlocals_new = code->co_nlocals;
  Py_ssize_t nlocals_old = frame->f_code->co_nlocals;

  if ((code->co_flags & CO_NOFREE) == 0) {
    ncells = PyTuple_GET_SIZE(code->co_cellvars);
    nfrees = PyTuple_GET_SIZE(code->co_freevars);
  }

  // DEBUG_NULL_CHECK(tstate);
  // DEBUG_NULL_CHECK(frame);
  // DEBUG_NULL_CHECK(code);
  // DEBUG_CHECK(ncells == PyTuple_GET_SIZE(frame->f_code->co_cellvars));
  // DEBUG_CHECK(nfrees == PyTuple_GET_SIZE(frame->f_code->co_freevars));
  // DEBUG_CHECK(nlocals_new >= nlocals_old);

  PyFrameObject *shadow = PyFrame_New(tstate, code, frame->f_globals, NULL);
  if (shadow == NULL) {
    return NULL;
  }

  PyObject **fastlocals_old = frame->f_localsplus;
  PyObject **fastlocals_new = shadow->f_localsplus;

  for (Py_ssize_t i = 0; i < nlocals_old; i++) {
    Py_XINCREF(fastlocals_old[i]);
    fastlocals_new[i] = fastlocals_old[i];
  }

  for (Py_ssize_t i = 0; i < ncells + nfrees; i++) {
    Py_XINCREF(fastlocals_old[nlocals_old + i]);
    fastlocals_new[nlocals_new + i] = fastlocals_old[nlocals_old + i];
  }

  PyObject *result = eval_frame_default(tstate, shadow, throw_flag);
  Py_DECREF(shadow);
  return result;
}

static PyObject *_custom_eval_frame(PyThreadState *tstate, PyFrameObject *frame,
                                    int throw_flag, PyObject *callback) {
  // TODO: why need this line?
  // https://peps.python.org/pep-0558/#fast-locals-proxy-implementation-details
  // https://devguide.python.org/internals/interpreter/#all-sorts-of-variables
  if (PyFrame_FastToLocalsWithError(frame) < 0) {
    return NULL;
  }

  // We don't run the current custom_eval_frame behavior for guards.
  // So we temporarily set the callback to Py_None to drive the correct behavior
  // in the shim.
  eval_frame_callback_set(Py_None);

  PyObject *args = Py_BuildValue("(O)", frame);
  PyObject *result = PyObject_CallObject(callback, args);
  // result: GuardedCode
  if (result == NULL) {
    // internal exception
    return NULL;
  } else if (result != Py_None) {
    //  NOTE: Cache is not supported now
    PyCodeObject *code = (PyCodeObject *)PyObject_GetAttrString(result, "code");
    // Re-enable custom behavior
    eval_frame_callback_set(callback);
    return eval_custom_code(tstate, frame, code, throw_flag);
  } else {
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

#if PY_VERSION_HEX >= 0x03090000
static PyObject *custom_eval_frame_shim(PyThreadState *tstate,
                                        PyFrameObject *frame, int throw_flag) {
  return _custom_eval_frame_shim(tstate, frame, throw_flag);
}
#else
static PyObject *custom_eval_frame_shim(PyFrameObject *frame, int throw_flag) {
  PyThreadState *tstate = PyThreadState_GET();
  return _custom_eval_frame_shim(tstate, frame, throw_flag);
}
#endif

static PyObject *set_eval_frame(PyObject *new_callback, PyThreadState *tstate) {
  // Change the eval frame callback and return the old one
  //  - None: disables Dynamo
  //  - Python callable(): enables Dynamo
  //  NOTE: Cache is not supported now
  PyObject *old_callback = eval_frame_callback_get();

#if PY_VERSION_HEX >= 0x03090000
  void *old_eval_frame = _PyInterpreterState_GetEvalFrameFunc(tstate->interp);
#else
  void *old_eval_frame = tstate->interp->eval_frame;
#endif

  // NOTE: multi-threading is not supported now
  if (old_callback != Py_None && new_callback == Py_None) {
    if (old_eval_frame != &_PyEval_EvalFrameDefault) {
      DEBUG_TRACE0("set _PyEval_EvalFrameDefault");
#if PY_VERSION_HEX >= 0x03090000
      _PyInterpreterState_SetEvalFrameFunc(tstate->interp,
                                           &_PyEval_EvalFrameDefault);
#else
      tstate->interp->eval_frame = &_PyEval_EvalFrameDefault;
#endif
    }
  } else if (old_callback == Py_None && new_callback != Py_None) {
    if (old_eval_frame != &custom_eval_frame_shim) {
      DEBUG_TRACE0("set custom_eval_frame_shim");
#if PY_VERSION_HEX >= 0x03090000
      _PyInterpreterState_SetEvalFrameFunc(tstate->interp,
                                           &custom_eval_frame_shim);
#else
      tstate->interp->eval_frame = &custom_eval_frame_shim;
#endif
    }
  }

  Py_INCREF(new_callback);
  eval_frame_callback_set(new_callback);

  return old_callback;
}

static PyObject *set_eval_frame_py(PyObject *dummy, PyObject *args) {
  PyObject *callback = NULL;
  if (!PyArg_ParseTuple(args, "O:callback", &callback)) {
    DEBUG_TRACE0("arg error");
    return NULL;
  }
  if (callback != Py_None && callback != Py_False &&
      !PyCallable_Check(callback)) {
    DEBUG_TRACE0("arg error");
    PyErr_SetString(PyExc_TypeError, "expected a callable");
    return NULL;
  }
  return set_eval_frame(callback, PyThreadState_GET());
}

static PyMethodDef _methods[] = {
    {"set_eval_frame", set_eval_frame_py, METH_VARARGS, NULL},
    {NULL, NULL},
};

static PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "_eval_frame",
    "_eval_frame doc",
    -1,
    _methods,
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
