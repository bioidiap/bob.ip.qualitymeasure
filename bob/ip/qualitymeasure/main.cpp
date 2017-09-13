// include directly and indirectly dependent libraries
#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif


#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/documentation.h>


// declare C++ function
void remove_highlights(   blitz::Array<float ,3> &img,
                          blitz::Array<float ,3> &diff,
                          blitz::Array<float ,3> &sfi,
                          blitz::Array<float ,3> &residue,
                          float epsilon,
                          bool  skip_diffuse,
                          bool  check_nan_inf);

// use the documentation classes to document the function
static bob::extension::FunctionDoc remove_highlights_doc = bob::extension::FunctionDoc(
  "remove_highlights",
  "This function implements a specular highlight removal algorithm.",
  "This function implements a specular highlight removal algorithm which, by \
     using an iterative process, separates the specular component from the \
     diffuse component of the picture. It returns a color incorect specular free \
     image, the diffuse component and the specular residue, respectively. It is \
     based on the original code by Robby T. Tan \
     reference: \
     separating reflection components of textured surfaces using a single image \
     by Robby T. Tan, Katsushi Ikeuchi, \
     IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), \
     27(2), pp.179-193, February, 2005"
)
.add_prototype("image", "specular_free_image, diffuse_image, specular_residue, epsilon")
.add_parameter("image", "array_like (3D, float32)", "The image to process")
.add_return("specular_free_image", "array_like (3D, float32)", "Specular free image with altered colors.")
.add_return("diffuse_image", "array_like (3D, float32)", "Diffuse component image, free of specularity.")
.add_return("specular_residue", "array_like (3D, float32)", "Specular residue of the image.")
.add_return("epsilon", "float32", "Parameter that specifies the number of iterations.")
;


// declare the function
static PyObject* PyRemoveHighlights(PyObject*, PyObject* args, PyObject* kwargs) {

  BOB_TRY

  static const char* const_kwlist[] =
    {"array", "startEps", "skip_diffuse", "check_nan_inf", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* array;
  double  epsilon  = 0.5f;
  bool    skip_diffuse  = false;
  bool    check_nan_inf = false;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|dii", kwlist,
                                  &PyBlitzArray_Converter, &array,
                                  &epsilon, &skip_diffuse, &check_nan_inf))
    return 0;

  auto array_ = make_safe(array);

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
  remove_highlights(img, diffuse_img, speckle_free_img, speckle_img,
                    (float)epsilon, skip_diffuse, check_nan_inf);

  // convert the blitz array back to numpy and return it
  PyObject *ret_tuple = PyTuple_New(3);
  PyTuple_SetItem(ret_tuple, 0, PyBlitzArrayCxx_AsNumpy(speckle_free_img));
  PyTuple_SetItem(ret_tuple, 1, PyBlitzArrayCxx_AsNumpy(diffuse_img));
  PyTuple_SetItem(ret_tuple, 2, PyBlitzArrayCxx_AsNumpy(speckle_img));

  return ret_tuple;

  // handle exceptions that occurred in this function
  BOB_CATCH_FUNCTION("remove_highlights", 0)
}


//////////////////////////////////////////////////////////////////////////
/////// Python module declaration ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

// module-wide methods
static PyMethodDef module_methods[] = {
  {
    "remove_highlights",
    (PyCFunction)PyRemoveHighlights,
    METH_VARARGS|METH_KEYWORDS,
    remove_highlights_doc.doc()
  },
  {NULL}  // Sentinel
};

// module documentation
PyDoc_STRVAR(module_docstr, "Tan Specular Highlights");

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
