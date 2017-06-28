// include directly and indirectly dependent libraries
#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif


#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>


// declare C++ function
void remove_highlights_orig(  blitz::Array<float ,3> &img,
                              blitz::Array<float ,3> &diff,
                              blitz::Array<float ,3> &sfi,
                              blitz::Array<float ,3> &residue,
                              float  epsilon);

// declare the function
static PyObject* PyRemoveHighlightsOrig(PyObject*, PyObject* args, PyObject* kwargs) {

  BOB_TRY

  static const char* const_kwlist[] = {"array", "startEps", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* array;
  double epsilon  = 0.5f;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|d", kwlist,
                                  &PyBlitzArray_Converter, &array,
                                  &epsilon)) return 0;

  // check that the array has the expected properties
  if (array->type_num != NPY_FLOAT32|| array->ndim != 3){
    PyErr_Format(PyExc_TypeError,
                "remove_highlights : Only 3D arrays of type float32 are allowed");
    return 0;
  }

  // extract the actual blitz array from the Python type
  blitz::Array<float ,3> img = *PyBlitzArrayCxx_AsBlitz<float , 3>(array);

  // results
  int dim_x = img.shape()[2];
  int dim_y = img.shape()[1];

  blitz::Array<float ,3> diffuse_img(3, dim_y, dim_x);
  blitz::Array<float ,3> speckle_free_img(3, dim_y, dim_x);
  blitz::Array<float ,3> speckle_img(3, dim_y, dim_x);

  diffuse_img       = 0;
  speckle_free_img  = 0;
  speckle_img       = 0;

  // call the C++ function
  remove_highlights_orig(img, diffuse_img, speckle_free_img, speckle_img, (float)epsilon);

  // convert the blitz array back to numpy and return it
  PyObject *ret_tuple = PyTuple_New(3);
  PyTuple_SetItem(ret_tuple, 0, PyBlitzArrayCxx_AsNumpy(speckle_free_img));
  PyTuple_SetItem(ret_tuple, 1, PyBlitzArrayCxx_AsNumpy(diffuse_img));
  PyTuple_SetItem(ret_tuple, 2, PyBlitzArrayCxx_AsNumpy(speckle_img));

  return ret_tuple;

  // handle exceptions that occurred in this function
  BOB_CATCH_FUNCTION("remove_highlights_orig", 0)
}


//////////////////////////////////////////////////////////////////////////
/////// Python module declaration ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

// module-wide methods
static PyMethodDef module_methods[] = {
  {
    "remove_highlights_orig",
    (PyCFunction)PyRemoveHighlightsOrig,
    METH_VARARGS|METH_KEYWORDS,
    "remove_highlights [doc]"
  },
  {NULL}  // Sentinel
};

// module documentation
PyDoc_STRVAR(module_docstr, "Exemplary Python Bindings");

// module definition
#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  BOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods,
  0, 0, 0, 0
};
#endif

// create the module
static PyObject* create_module (void) {

# if PY_VERSION_HEX >= 0x03000000
  PyObject* module = PyModule_Create(&module_definition);
  auto module_ = make_xsafe(module);
  const char* ret = "O";
# else
  PyObject* module = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
  const char* ret = "N";
# endif
  if (!module) return 0;

  if (PyModule_AddStringConstant(module, "__version__", BOB_EXT_MODULE_VERSION) < 0) return 0;

  /* imports bob.blitz C-API + dependencies */
  if (import_bob_blitz() < 0) return 0;

  return Py_BuildValue(ret, module);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
