# Data Formats

## In-memory Data

### Basic Concepts

The main format we're using within plottr is the {py:class}`DataDict <labcore.data.datadict.DataDict>`. While most of the actual numeric data will typically live in numpy arrays (or lists, or similar), they don't typically capture easily arbitrary metadata and relationships between arrays. Say, for example, we have some data ``z`` that depends on two other variables, ``x`` and ``y``. This information has be stored somewhere, and numpy doesn't offer readily a solution here. There are various extensions, for example [xarray](http://xarray.pydata.org) or the [MetaArray class](https://scipy-cookbook.readthedocs.io/items/MetaArray.html). Those however typically have a grid format in mind, which we do not want to impose. Instead, we use a wrapper around the python dictionary that contains all the required meta information to infer the relevant relationships, and that uses numpy arrays internally to store the numeric data. Additionally we can store any other arbitrary meta data.

A DataDict container (a `dataset`) can contain multiple `data fields` (or variables), that have values and can contain their own meta information. Importantly, we distinct between independent fields (the `axes`) and dependent fields (the `data`).

Despite the naming, `axes` is not meant to imply that the `data` have to have a certain shape (but the degree to which this is true depends on the class used). A list of classes for different shapes of data can be found below.

The basic structure of data conceptually looks like this (we inherit from `dict`):

```
{
    'data_1' : {
        'axes' : ['ax1', 'ax2'],
        'unit' : 'some unit',
        'values' : [ ... ],
        '__meta__' : 'This is very important data',
        ...
    },
    'ax1' : {
        'axes' : [],
        'unit' : 'some other unit',
        'values' : [ ... ],
        ...,
    },
    'ax2' : {
        'axes' : [],
        'unit' : 'a third unit',
        'values' : [ ... ],
        ...,
    },
    '__globalmeta__' : 'some information about this data set',
    '__moremeta__' : 1234,
    ...
}
```

In this case we have one dependent variable, ``data_1``, that depends on two axes, ``ax1`` and ``ax2``. This concept is restricted only in the following way:

  * A dependent can depend on any number of independents.
  * An independent cannot depend on other fields itself.
  * Any field that does not depend on another, is treated as an axis.

Note that meta information is contained in entries whose keys start and end with double underscores. Both the DataDict itself, as well as each field can contain meta information.

In the most basic implementation, the only restriction on the data values is that they need to be contained in a sequence (typically as list, or numpy array), and that the length of all values in the data set (the number of `records`) must be equal. Note that this does not preclude nested sequences!

#### Relevant Data Classes

**DataDictBase:** 
The main base class. Only checks for correct dependencies. Any requirements on data structure is left to the inheriting classes. The class contains methods for easy access to data and metadata.

**DataDict:**
The only requirement for valid data is that the number of records is the same for all data fields. Contains some tools for expansion of data.

**MeshgridDataDict:** For data that lives on a grid (not necessarily regular).

## Datadict

!!! note
    Because DataDicts are [python dictionaries](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) , we highly recommend becoming familiar with them before utilizing DataDicts.

### Basic Use

We can start by creating an empty DataDict like any other python object:

```python
>>> data_dict = DataDict()
>>> data_dict
{}
```

We can create the structure of the data_dict by creating dictionary items and populating them like a normal python dictionary:

```python
>>> data_dict['x'] = dict(unit='m')
>>> data_dict
{'x': {'unit': 'm'}}
```

We can also start by creating a DataDict that has the structure of the data we are going to record:

```python
>>> data_dict = DataDict(x=dict(unit='m'), y = dict(unit='m'), z = dict(axes=['x', 'y']))
>>> data_dict
{'x': {'unit': 'm'}, 'y': {'unit': 'm'}, 'z': {'axes': ['x', 'y']}}
```

The DataDict that we just created contains no data yet, only the structure and relationship of the data fields. We have also specified the unit of ``x`` and ``y`` and which variables are independent variables (``x``, ``y``), or how we will call them from now on, ``axes`` and dependent variables (``z``), or, ``dependents``.

### Structure

From the basic and empty DataDict we can already start to inspect its structure. To see the entire structure of a DataDict we can use the {py:meth}`structure <labcore.data.datadict.DataDictBase.structure>` method:

```python
>>> data_dict = DataDict(x=dict(unit='m'), y = dict(unit='m'), z = dict(axes=['x', 'y']))
>>> data_dict.structure()
{'x': {'unit': 'm', 'axes': [], 'label': ''},
 'y': {'unit': 'm', 'axes': [], 'label': ''},
 'z': {'axes': ['x', 'y'], 'unit': '', 'label': ''}}
```

We can check for specific things inside the DataDict. We can look at the axes:

```python
>>> data_dict.axes()
['x', 'y']
```

We can look at all the dependents:

```python
>>> data_dict.dependents()
['z']
```

We can also see the shape of a DataDict by using the {py:meth}`shapes <labcore.data.datadict.DataDictBase.shapes>` method:

```python
>>> data_dict.shapes()
{'x': (0,), 'y': (0,), 'z': (0,)}
```

### Populating the DataDict

One of the only "restrictions" that DataDict implements is that every data field must have the same number of records (items). However, restrictions is in quotes because there is nothing that is stopping you from having different data fields have different number of records, this will only make the DataDict invalid. We will explore what his means later.

There are 2 different ways of safely populating a DataDict, adding data to it or appending 2 different DataDict to each other.

!!! note
    You can always manually update the item ``values`` any data field like any other item of a python dictionary, however, populating the DataDict this way can result in an invalid DataDict if you are not being careful. Both population methods presented below contains checks to make sure that the new data being added will not create an invalid DataDict.

We can add data to an existing DataDict with the {py:meth}`add_data <labcore.data.datadict.DataDict.add_data>` method:

```python
>>> data_dict = DataDict(x=dict(unit='m'), y = dict(unit='m'), z = dict(axes=['x', 'y']))
>>> data_dict.add_data(x=[0,1,2], y=[0,1,2], z=[0,1,4])
>>> data_dict
{'x': {'unit': 'm', 'axes': [], 'label': '', 'values': array([0, 1, 2])},
 'y': {'unit': 'm', 'axes': [], 'label': '', 'values': array([0, 1, 2])},
 'z': {'axes': ['x', 'y'],  'unit': '',  'label': '',  'values': array([0, 1, 4])}}
```

We now have a populated DataDict. It is important to notice that this method will also add any of the missing special keys that a data field doesn't have (`values`, `axes`, `unit`, and `label`). Populating the DataDict with this method will also ensure that every item has the same number of records and the correct shape, either by adding ``nan`` to the other data fields or by nesting the data arrays so that the outer most dimension of every data field has the same number of records.

We can see this in action if we add a single record to a data field with items but no the rest:

```python
>>> data_dict.add_data(x=[9])
>>> data_dict
{'x': {'unit': 'm', 'axes': [], 'label': '', 'values': array([0, 1, 2, 9])},
 'y': {'unit': 'm', 'axes': [], 'label': '', 'values': array([ 0.,  1.,  2., nan])},
 'z': {'axes': ['x', 'y'], 'unit': '', 'label': '', 'values': array([ 0.,  1.,  4., nan])}}
```

As we can see, both ``y`` and ``z`` have an extra ``nan`` record in them. We can observe the change of dimension if we do not add the same number of records to all data fields:

```python
>>> data_dict = DataDict(x=dict(unit='m'), y = dict(unit='m'), z = dict(axes=['x', 'y']))
>>> data_dict.add_data(x=[0,1,2], y=[0,1,2],z=[0])
>>> data_dict
{'x': {'unit': 'm', 'axes': [], 'label': '', 'values': array([[0, 1, 2]])},
 'y': {'unit': 'm', 'axes': [], 'label': '', 'values': array([[0, 1, 2]])},
 'z': {'axes': ['x', 'y'], 'unit': '', 'label': '', 'values': array([0])}}
```

If we want to expand our DataDict by appending another one, we need to make sure that both of our DataDicts have the same inner structure. We can check that by utilizing the static method {py:meth}`same_structure <labcore.data.datadict.DataDictBase.same_structure>`:

```python
>>> data_dict_1 = DataDict(x=dict(unit='m'), y=dict(unit='m'), z=dict(axes=['x','y']))
>>> data_dict_2 = DataDict(x=dict(unit='m'), y=dict(unit='m'), z=dict(axes=['x','y']))
>>> data_dict_1.add_data(x=[0,1,2], y=[0,1,2], z=[0,1,4])
>>> data_dict_2.add_data(x=[3,4], y=[3,4], z=[9,16])
>>> DataDict.same_structure(data_dict_1, data_dict_2)
True
```

!!! note
    Make sure that both DataDicts have the exact same structure. This means that every item of every data field that appears when using the method {py:meth}`same_structure <labcore.data.datadict.DataDictBase.same_structure>` (`unit`, `axes`, and `label`) are identical to one another, except for `values`. Any slight difference will make this method fail due to conflicting structures.    

The {py:meth}`append <labcore.data.datadict.DataDict.append>` method will do this check before appending the 2 DataDict, and will only append them if the check returns ``True``. Once we know that the structure is the same we can append them:

```python
>>> data_dict_1.append(data_dict_2)
>>> data_dict_1
{'x': {'unit': 'm', 'axes': [], 'label': '', 'values': array([0, 1, 2, 3, 4])},
 'y': {'unit': 'm', 'axes': [], 'label': '', 'values': array([0, 1, 2, 3, 4])},
 'z': {'axes': ['x', 'y'], 'unit': '', 'label': '', 'values': array([ 0,  1,  4,  9, 16])}}
```

### Meta Data

One of the advantages DataDicts have over regular python dictionaries is their ability to contain meta data. Meta data can be added to the entire DataDict or to individual data fields. Any object inside a ``DataDict`` whose key starts and ends with two underscores is considered meta data.

We can simply add meta data manually by adding an item with the proper notation:

```python
>>> data_dict['__metadata__'] = 'important meta data'
```

Or we can use the {py:meth}`add_meta <labcore.data.datadict.DataDictBase.add_meta>` method:

```python
>>> data_dict.add_meta('sample_temperature', '10mK')
>>> data_dict
{'x': {'unit': 'm', 'axes': [], 'label': '', 'values': array([0, 1, 2])},
 'y': {'unit': 'm', 'axes': [], 'label': '', 'values': array([0, 1, 2])},
 'z': {'axes': ['x', 'y'], 'unit': '', 'label': '', 'values': array([0, 1, 4])},
 '__metadata__': 'important meta data',
 '__sample_temperature__': '10mK'}
```

We can also add meta data to a specific data field by passing its name as the last argument:

```python
>>> data_dict.has_meta('sample_temperature')
True
```

We can retrieve the meta data with the {py:meth}`meta_val <labcore.data.datadict.DataDictBase.meta_val>` method:

```python
>>> data_dict.meta_val('sample_temperature')
'10mK'
```

We can also ask for a meta value from a specific data field by passing the data field as the second argument:

```python
>>> data_dict.meta_val('extra_metadata','x')
'important meta data'
```

We can delete a specific meta field by using the {py:meth}`delete_meta <labcore.data.datadict.DataDictBase.delete_meta>` method:

```python
>>> data_dict.delete_meta('metadata')
>>> data_dict.has_meta('metadata')
False
```

This also work for meta data in data fields by passing the data field as the last argument:

```python
>>> data_dict.delete_meta('extra_metadata', 'x')
>>> data_dict['x']
{'unit': 'm', 'axes': [], 'label': '', 'values': array([0, 1, 2])}
```

We can delete all the meta data present in the DataDict with the {py:meth}`clear_meta <labcore.data.datadict.DataDictBase.clear_meta>` method:

```python
>>> data_dict.add_meta('metadata', 'important meta data')
>>> data_dict.add_meta('extra_metadata', 'important meta data', 'x')
>>> data_dict.clear_meta()
>>> data_dict
{'x': {'unit': 'm', 'axes': [], 'label': '', 'values': array([0, 1, 2])},
 'y': {'unit': 'm', 'axes': [], 'label': '', 'values': array([0, 1, 2])},
 'z': {'axes': ['x', 'y'], 'unit': '', 'label': '', 'values': array([0, 1, 4])}}

```

!!! note
    There are 3 helper functions in the datadict module that help converting from meta data name to key. These are: 

      * {py:func}`is_meta_key <labcore.data.datadict.is_meta_key>`,
      * {py:func}`meta_key_to_name <labcore.data.datadict.meta_key_to_name>`
      * {py:func}`meta_name_to_key <labcore.data.datadict.meta_name_to_key>`


### Meshgrid DataDict

A dataset where the axes form a grid on which the dependent values reside.

This is a more special case than DataDict, but a very common scenario. To support flexible grids, this class requires that all axes specify values for each datapoint, rather than a single row/column/dimension.

For example, if we want to specify a 3-dimensional grid with axes `x`, `y`, `z`, the values of `x`, `y`, `z` all need to be 3-dimensional arrays; the same goes for all dependents that live on that grid. Then, say, `x[i,j,k]` is the x-coordinate of point `i`,`j`,`k` of the grid.

This implies that a MeshgridDataDict can only have a single shape, i.e., all data values share the exact same nesting structure.

For grids where the axes do not depend on each other, the correct values for the axes can be obtained from [np.meshgrid](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html) (hence the name of the class).

Example: a simple uniform 3x2 grid might look like this; x and y are the coordinates of the grid, and z is a function of the two:

```
    x = [[0, 0],
         [1, 1],
         [2, 2]]

    y = [[0, 1],
         [0, 1],
         [0, 1]]

    z = x * y =
        [[0, 0],
         [0, 1],
         [0, 2]]

```

!!! note
    Internally we will typically assume that the nested axes are ordered from slow to fast, i.e., dimension 1 is the most outer axis, and dimension N of an N-dimensional array the most inner (i.e., the fastest changing one). This guarantees, for example, that the default implementation of `np.reshape` has the expected outcome. If, for some reason, the specified axes are not in that order (e.g., we might have ``z`` with ``axes = ['x', 'y']``, but ``x`` is the fast axis in the data). In such a case, the guideline is that at creation of the meshgrid, the data should be transposed such that it conforms correctly to the order as given in the ``axis = [...]`` specification of the data. The function {py:func}`datadict_to_meshgrid <labcore.data.datadict.datadict_to_meshgrid>` provides options for that.

This implementation of {py:class}`DataDictBase <labcore.data.datadict.DataDictBase>` consists only of three extra methods:

  * {py:attr}`shape <labcore.data.datadict.MeshgridDataDict.shape>`
  * {py:meth}`validate <labcore.data.datadict.MeshgridDataDict.validate>`
  * {py:meth}`reorder_axes <labcore.data.datadict.MeshgridDataDict.reorder_axes>`

So the only way of populating it is by manually modifying the ``values`` object of each data field since the tools for populating the DataDict are specific to the {py:class}`DataDict <labcore.data.datadict.DataDict>` implementation.

## DataDict Storage

The datadict_storage.py module offers tools to help with saving DataDicts into disk by storing them in DDH5 files ([HDF5 files](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) that contains DataDicts inside).

### Description of the HDF5 Storage Format

We use a simple mapping from DataDict to the HDF5 file. Within the file, a single DataDict is stored in a (top-level) group of the file.
The data fields are datasets within that group.

Global meta data of the DataDict are attributes of the group; field meta data are attributes of the dataset (incl., the `unit` and `axes` values). The meta data keys are given exactly like in the DataDict, i.e., includes the double underscore pre- and suffix.

For more specific information on how HDF5 works please read the [following documentation](https://docs.hdfgroup.org/hdf5/v1_14/_h5_d_m__u_g.html#subsec_data_model_intro)

### Working With DDH5 Files

When we are working with data, the first thing we usually want to do is to save it in disk. We can directly save an already existing DataDict into disk by calling the function {py:func}`datadict_to_hdf5 <labcore.data.datadict_storage.datadict_to_hdf5>`.

```python
>>> data_dict = DataDict(x=dict(values=np.array([0,1,2]), axes=[], __unit__='cm'), y=dict(values=np.array([3,4,5]), axes=['x']))
>>> data_dict
{'x': {'values': array([0, 1, 2]), 'axes': [], '__unit__': 'cm'},
 'y': {'values': array([3, 4, 5]), 'axes': ['x']}}
>>> datadict_to_hdf5(data_dict, 'folder\data.ddh5')
```

{py:func}`datadict_to_hdf5 <labcore.data.datadict_storage.datadict_to_hdf5>` will save data_dict in a file named 'data.ddh5' in whatever directory is passed to it, creating new folders if they don't already exists. The file will contain all of the data fields as well as all the metadata,
with some more metadata generated to specify when the DataDict was created.

!!! note
    Meta data is only written during initial writing of the dataset. If we're appending to existing datasets, we're not setting meta data anymore.

!!! warning
    For this method to properly work the objects that are being saved in the ``values`` key of a data field must by a numpy array, or numpy array like.

Data saved on disk is useless however if we do not have a way of accessing it. To do this we use the {py:func}`datadict_from_hdf5 <labcore.data.datadict_storage.datadict_from_hdf5>`:

```python
>>> loaded_data_dict = datadict_from_hdf5('folder\data.ddh5')
>>> loaded_data_dict
{'__creation_time_sec__': 1651159636.0,
 '__creation_time_str__': '2022-04-28 10:27:16',
 'x': {'values': array([0, 1, 2]),
  'axes': [],
  '__shape__': (3,),
  '__creation_time_sec__': 1651159636.0,
  '__creation_time_str__': '2022-04-28 10:27:16',
  '__unit__': 'cm',
  'unit': '',
  'label': ''},
 'y': {'values': array([3, 4, 5]),
  'axes': ['x'],
  '__shape__': (3,),
  '__creation_time_sec__': 1651159636.0,
  '__creation_time_str__': '2022-04-28 10:27:16',
  'unit': '',
  'label': ''}}
```

We can see that the DataDict is the same one we saved earlier with the added metadata that indicates the time it was created.

By default both {py:func}`datadict_to_hdf5 <labcore.data.datadict_storage.datadict_to_hdf5>` and {py:func}`datadict_from_hdf5 <labcore.data.datadict_storage.datadict_from_hdf5>` save and load the datadict in the 'data' group of the DDH5. Both of these can by changed by passing another group to the argument 'groupname'. We can see this if we manually create a second group and save a new DataDict there:

```python
>>> data_dict2 = DataDict(a=dict(values=np.array([0,1,2]), axes=[], __unit__='cm'), b=dict(values=np.array([3,4,5]), axes=['a']))
>>> with h5py.File('folder\data.ddh5', 'a') as file:
>>>    file.create_group('other_data')
>>> datadict_to_hdf5(data_dict2, 'folder\data.ddh5', groupname='other_data')
```

If we then load the DDH5 file like before we only see the first DataDict:

```python
>>> loaded_data_dict = datadict_from_hdf5('folder\data.ddh5', 'data')
>>> loaded_data_dict
{'__creation_time_sec__': 1651159636.0,
 '__creation_time_str__': '2022-04-28 10:27:16',
 'x': {'values': array([0, 1, 2]),
  'axes': [],
  '__shape__': (3,),
  '__creation_time_sec__': 1651159636.0,
  '__creation_time_str__': '2022-04-28 10:27:16',
  '__unit__': 'cm',
  'unit': '',
  'label': ''},
 'y': {'values': array([3, 4, 5]),
  'axes': ['x'],
  '__shape__': (3,),
  '__creation_time_sec__': 1651159636.0,
  '__creation_time_str__': '2022-04-28 10:27:16',
  'unit': '',
  'label': ''}}
```

To see the other DataDict we can specify the group in the argument 'groupname':

```python
>>> loaded_data_dict = datadict_from_hdf5('folder\data.ddh5', 'other_data')
>>> loaded_data_dict
{'a': {'values': array([0, 1, 2]),
  'axes': [],
  '__shape__': (3,),
  '__creation_time_sec__': 1651159636.0,
  '__creation_time_str__': '2022-04-28 10:27:16',
  '__unit__': 'cm',
  'unit': '',
  'label': ''},
 'b': {'values': array([3, 4, 5]),
  'axes': ['a'],
  '__shape__': (3,),
  '__creation_time_sec__': 1651159636.0,
  '__creation_time_str__': '2022-04-28 10:27:16',
  'unit': '',
  'label': ''}}
```

We can also use {py:func}`all_datadicts_from_hdf5 <labcore.data.datadict_storage.all_datadicts_from_hdf5>` to get a dictionary with all DataDicts in every group inside:

```python
>>> all_datadicts = all_datadicts_from_hdf5('folder\data.ddh5')
>>> all_datadicts
{'data': {'__creation_time_sec__': 1651159636.0,
  '__creation_time_str__': '2022-04-28 10:27:16',
  'x': {'values': array([0, 1, 2]),
   'axes': [],
   '__shape__': (3,),
   '__creation_time_sec__': 1651159636.0,
   '__creation_time_str__': '2022-04-28 10:27:16',
   '__unit__': 'cm',
   'unit': '',
   'label': ''},
  'y': {'values': array([3, 4, 5]),
   'axes': ['x'],
   '__shape__': (3,),
   '__creation_time_sec__': 1651159636.0,
   '__creation_time_str__': '2022-04-28 10:27:16',
   'unit': '',
   'label': ''}},
 'other_data': {'a': {'values': array([0, 1, 2]),
   'axes': [],
   '__shape__': (3,),
   '__creation_time_sec__': 1651159636.0,
   '__creation_time_str__': '2022-04-28 10:27:16',
   '__unit__': 'cm',
   'unit': '',
   'label': ''},
  'b': {'values': array([3, 4, 5]),
   'axes': ['a'],
   '__shape__': (3,),
   '__creation_time_sec__': 1651159636.0,
   '__creation_time_str__': '2022-04-28 10:27:16',
   'unit': '',
   'label': ''}}}
```

### DDH5 Writer

Most times we want to be saving data to disk as soon as it is generated by an experiment (or iteration), instead of waiting to have a complete DataDict. To do this, Datadict_storage also offers a [context manager](https://docs.python.org/3/library/stdtypes.html#context-manager-types) with which we can safely save our incoming data.

To use it we first need to create an empty DataDict that contains the structure of how the data is going to look like:

```python
>>> data_dict = DataDict(
>>> x = dict(unit='x_unit'),
>>> y = dict(unit='y_unit', axes=['x']))
```

With our created DataDict, we can start the {py:class}`DDH5Writer <labcore.data.datadict_storage.DDH5Writer>` context manager and add data to our DataDict utilizing the {py:meth}`add_data <plottr.data.datadict_storage.DDH5Writer.add_data>`

```python
>>> with DDH5Writer(datadict=data_dict, basedir='./data/', name='Test') as writer:
>>>    for x in range(10):
>>>        writer.add_data(x=x, y=x**2)
Data location:  data\2022-04-27\2022-04-27T145308_a986867c-Test\data.ddh5
```

The writer created the folder 'data' (because it did not exist before) and inside that folder,
created another new folder for the current day and another new folder inside of it day folder for the the DataDict that we saved with the naming structure of ``YYYY-mm-dd_THHMMSS_<ID>-<name>/<filename>.ddh5``,
where name is the name parameter passed to the writer.
The writer creates this structure such that when we run the writer again with new data, it will create another folder following the naming structure inside the current date folder.
This way each new DataDict will be saved in the date it was generated with a time stamp in the name of the folder containing it.

### Change File Extension and Time Format

Finally, datadict_storage contains 2 module variables, 'DATAFILEXT' and 'TIMESTRFORMAT'.

'DATAFILEXT' by default is 'ddh5', and it is used to specify the extension file of all of the module saving functions. Change this variable if you want your HDF5 to have a different extension by default, instead of passing it everytime.

'TIMESTRFORMAT' specifies how the time is formated in the new metadata created when saving a DataDict. The default is: ``"%Y-%m-%d %H:%M:%S"``, and it follows the structure of [strftime](https://docs.python.org/3/library/time.html#time.strftime>).
