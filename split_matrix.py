from scipy.sparse import vstack, hstack, csr_matrix, coo_matrix
# from easyinfo.utils import # lprint, # vprint

def add_file_suffix(filename, suf):
  name, ext = os.path.splitext(filename)
  return name+"_"+str(suf)+ext

def calc_split_offsets(matrix_len, num_splits):
  offsets = [0] # Start
  if num_splits > 1:
    split_portion = matrix_len / num_splits
    curr_offset = int(split_portion) # End of first split
    offsets.append(curr_offset)
    for i in range(num_splits-2): # For all the middle splits
      curr_offset += int(split_portion)
      offsets.append(curr_offset)
  offsets.append(matrix_len) # End of last split
  return offsets

# Transform matrix
def transpose_job(matrix, **kwargs):
  return matrix.transpose(), kwargs

# Transform matrix
def transpose_first_job(matrix, **kwargs):
  return matrix.transpose(), kwargs

# Transform second_matrix in kwargs
def transpose_second_job(matrix, **kwargs):
  kwargs['second'] = kwargs['second'].transpose()
  return matrix, kwargs

def normalize_job(matrix, **kwargs):
  return normalize(matrix), kwargs

def multiply_2op_job(matrix, **kwargs):
  pass

# Multiplies matrix by first argument in kwargs
def multiply_job(matrix, **kwargs):
  try:
    second_matrix = kwargs['second']
    prod = matrix * second_matrix
    lprint(matrix)
    lprint(second_matrix)
    lprint(prod)
    return prod, kwargs
  except ValueError as e:
    eprint("ValueError in multiply_job")
    eprint(e)
    lprint(matrix)
    lprint(second_matrix)
  return None

# Flip the first matrix (matrix) and second (kwargs['second'])
#   in preparation for future operations.
def flip_matrices_job(matrix, **kwargs):
  if 'second' not in kwargs:
    eprint("Flip job requires a 'second' matrix")
  first = kwargs['second']
  kwargs['second'] = matrix
  print("After flip:")
  lprint(first)
  lprint(matrix, 'second')
  return first, kwargs

# Return top k rows for each column (axis=0)
def sort_rows_k_job(matrix, **kwargs):
  return topn(matrix, kwargs['k'], axis=0), kwargs
  
# Return matrix with sorted columns (axis=0)
def sort_rows_job(matrix, **kwargs):
  return np.argsort(matrix, axis=0)[::-1], kwargs

# Return top k columns for each row (axis=1)
def sort_cols_k_job(matrix, **kwargs):
  return topn(matrix, kwargs['k'], axis=1), kwargs
  
# Return matrix with sorted rows (axis=1)
def sort_cols_job(matrix, **kwargs):
  return np.flip(np.argsort(matrix, axis=1), axis=1), kwargs

# Convert to a dense array
def toarray_job(matrix, **kwargs):
  if hasattr(matrix, 'toarray'):
    return matrix.toarray(), kwargs
  else:
    return matrix, kwargs

# Convert to a dense array
def asarray_job(matrix, **kwargs):
  if hasattr(matrix, 'asarray'):
    return matrix.asarray(), kwargs
  else:
    return matrix, kwargs

# Convert to a csr sparse matrix
def to_csr_job(matrix, **kwargs):
  return csr_matrix(matrix), kwargs

# Assign a random uniform float to every cell in the matrix
# Constructs a new numpy array of the same size
def random_uniform_job(matrix, **kwargs):
  uniform_matrix = np.random.uniform(size=matrix.shape)
  return uniform_matrix, kwargs

def random_uniform_product_2op_job(matrix, **kwargs):
  pass

# Assign a random uniform float to every cell in the matrix
# Constructs a new numpy array with size of the product
# of matrix and second
# uniform_matrix = matrix_rows x second_columns
def random_uniform_product_job(matrix, **kwargs):
  second = kwargs['second']
  uniform_matrix = np.random.uniform(size=(vlen(matrix), vwid(second)))
  return uniform_matrix, kwargs

def duplicate_rows_2op_job(matrix, **kwargs):
  pass

# second here is a matrix with 1 or more rows to be duplicated
# The rows of second are duplicated by the length of the first matrix
def duplicate_rows_job(matrix, **kwargs):
  second = kwargs['second']
  second, kwargs = toarray_job(second, **kwargs)
  expanded_matrix = np.tile(second, (vlen(matrix), 1))
  return expanded_matrix, kwargs

def offsets_from_filename(base_filename, num_splits):
  offsets = set()
  dirname = os.path.dirname(base_filename)
  base_filename, _ = os.path.splitext(os.path.basename(base_filename))
  base_filename += '_'
  for subdir, dirs, files in os.walk(dirname):
    for filename in files:
      if filename.startswith(base_filename):
        filename, _ = os.path.splitext(filename)
        offset_splits = filename[len(base_filename):].split('-') # e.g. filename_0-200
       
        try:
          for offset_part in offset_splits:
            offset_num = to_int(offset_part)
            if len(str(offset_num)) < len(offset_part):
              continue
            offsets.add(offset_num)
        except (TypeError, ValueError):
          vprint(offset_splits, name='Could not convert to ints')
          vprint(filename)
  
  offsets = sorted(list(offsets))
  try:
    matrix_len = offsets[-1]
  except IndexError:
    # vprint(base_filename, "Found no offsets with given base_filename")
    return []
  offsets = calc_split_offsets(matrix_len, num_splits)
  return offsets

def load_rows(load_filename, num_splits, rows, save_transformed=False, save_filename=None, axis=0):
  offsets = offsets_from_filename(load_filename, num_splits)
  transformed_matrix = None
  for offset_i in range(len(offsets)-1):
    begin_i = offsets[offset_i]
    end_i = offsets[offset_i+1]
    start()
    split_filename = add_file_suffix(load_filename, str(begin_i)+"-"+str(end_i))
    with open(split_filename, 'rb') as bin_file:
      transformed_split = pickle.load(bin_file)
      if isinstance(transformed_split, str): # In some cases, what was stored was a filename!
        transformed_split = load_splits(transformed_split, kwargs['num_splits_second'])
      transformed_split = coo_matrix(transformed_split)
      if transformed_matrix is None:
        transformed_matrix = transformed_split
      else:
        if axis == 1:
          transformed_matrix = hstack([transformed_matrix, transformed_split])
        else:
          transformed_matrix = vstack([transformed_matrix, transformed_split])
      end("stacked transformed split from "+split_filename)
      del transformed_split
  lprint(transformed_matrix)
  if not save_filename:
    save_filename = load_filename
  if save_transformed and save_filename:
    with open(save_filename, 'wb') as bin_file:
      pickle.dump(transformed_matrix, bin_file)
  return csr_matrix(transformed_matrix)

# Load and stack split matrix from files
def load_splits(load_filename, num_splits, save_transformed=False, save_filename=None, axis=0, rows=None, cols=None):
  offsets = offsets_from_filename(load_filename, num_splits)
  transformed_matrix = None
  for offset_i in range(len(offsets)-1):
    begin_i = offsets[offset_i]
    end_i = offsets[offset_i+1]
    start()
    split_filename = add_file_suffix(load_filename, str(begin_i)+"-"+str(end_i))
    with open(split_filename, 'rb') as bin_file:
      transformed_split = coo_matrix(pickle.load(bin_file))
      if transformed_matrix is None:
        transformed_matrix = transformed_split
      else:
        if axis == 1:
          transformed_matrix = hstack([transformed_matrix, transformed_split])
        else:
          transformed_matrix = vstack([transformed_matrix, transformed_split])
      end("stacked transformed split from "+split_filename)
      del transformed_split
  # lprint(transformed_matrix)
  if not save_filename:
    save_filename = load_filename
  if save_transformed and save_filename:
    with open(save_filename, 'wb') as bin_file:
      pickle.dump(transformed_matrix, bin_file)
  return csr_matrix(transformed_matrix)

# Find and return list of (split_filepath, (start_index, end_index))
def list_splits(load_filename, num_splits, matrix_len=None):
  if matrix_len:
    offsets = calc_split_offsets(matrix_len, num_splits)
  else:
    offsets = offsets_from_filename(load_filename, num_splits)

  splits = []
  for offset_i in range(len(offsets)-1):
    begin_i = offsets[offset_i]
    end_i = offsets[offset_i+1]
    split_filename = add_file_suffix(load_filename, str(begin_i)+"-"+str(end_i))
    splits.append((split_filename, (begin_i, end_i)))

  return splits

# For a second matrix, end arguments with '_second'.
def split_second(pipeline, matrix, **kwargs):
  if not hasattr(pipeline, '__iter__'):
      pipeline = [pipeline]
  kwargs_second = {}
  for arg in kwargs:
    if arg.endswith('_second') and not arg.startswith('pipeline'):
      kwargs_second[arg[:-7]] = kwargs[arg] # In second land, these arguments are now normal arguments (first)
  if 'axis' not in kwargs_second:
    kwargs_second['axis'] = 1 # Change default axis to 1 for second matrix.
  if 'stack_transformed' not in kwargs_second:
    # If you're using a split second matrix, you probably want the whole stacked matrix
    # Which is probably the result of a split on the first matrix
    kwargs_second['stack_transformed'] = True
  if 'save_transformed' not in kwargs_second:
    kwargs_second['save_transformed'] = False
  # First split the second matrix as if it was the first (with split_matrix()),
  #   then flip for future operations
  pipeline_second = []
  if 'pipeline_second' in kwargs:
    pipeline_second.extend(kwargs['pipeline_second']) # Any custom jobs to be done on each split of second
  pipeline_second.append(flip_matrices_job) # Flip back and forth for each split of the first matrix
  pipeline_second.extend(pipeline) # The original job
  # pipeline_second.append(flip_matrices_job) # Flip back and forth for each split of the first matrix
  kwargs_second['second'] = matrix # Flip the matrices, to be flipped back after splitting
  kwargs_second['splitting_second'] = True
  # return flip_matrices_job(split_matrix(pipeline_second, large_matrix=kwargs['second'], **kwargs_second), **kwargs_second)
  return split_matrix(pipeline_second, large_matrix=kwargs['second'], **kwargs_second), kwargs

# Splits large_matrix by axis (default split by rows).
# Use kwargs to arguments needed by jobs in pipeline and for split operations on a second matrix.
# For a second matrix, end arguments with '_second'.
# verbose can be False, None, or True. None is default and the recommended setting.
def split_matrix(pipeline, save_filename=False, num_splits=1, large_matrix=None, matrix_len=None, 
                           load_filename="", save_transformed=True, stack_transformed=False, axis=0,
                            verbose=None, splitting_second=False, **kwargs):
  vprint(pipeline)
  if not load_filename:
    load_filename = ""
  # vprint(kwargs, name='kwargs')
  if 'second' in kwargs and not isinstance(kwargs['second'], str): # Not a file
    kwargs['second'] = csr_matrix(kwargs['second'])
  if (splitting_second or ('splitting_second' in kwargs and kwargs['splitting_second'])):
    original_first = kwargs['second'] # in the original split_matrix call, this was the first matrix

  offsets = []
  if large_matrix is None and num_splits == 1:
    with open(load_filename, 'rb') as bin_file:
      large_matrix = pickle.load(bin_file)
  
  if isinstance(large_matrix, str):
    load_filename = large_matrix
    large_matrix = None

  if large_matrix is not None:
    large_matrix = csr_matrix(large_matrix)
    if verbose:
      lprint(large_matrix)
    if hasattr(large_matrix, 'shape'):
      matrix_len = large_matrix.shape
      if isinstance(matrix_len, (list, tuple)):
        matrix_len = matrix_len[axis] 
    else:
      if axis == 1:
        matrix_len == len(large_matrix[0])
      elif axis == 0:
        matrix_len = len(large_matrix)

    offsets = calc_split_offsets(matrix_len, num_splits)
  elif num_splits > 1:
    if matrix_len:
      offsets = calc_split_offsets(matrix_len, num_splits)
    else:
      offsets = offsets_from_filename(load_filename, num_splits)
  # vprint(offsets)
    
  if not save_filename:
    save_filename = add_file_suffix(load_filename, 'transformed')

  transformed_matrix = None
  split_i = -1
  for offset_i in range(len(offsets)-1):
    begin_i = offsets[offset_i]
    end_i = offsets[offset_i+1]
    split_i += 1
    
    split_matrix = None
    if large_matrix is None:
      if num_splits > 1:
          split_load_filename = add_file_suffix(load_filename, str(begin_i)+"-"+str(end_i))
          if verbose:
            start()
          with open(split_load_filename, 'rb') as load_bin_file:
            split_matrix = pickle.load(load_bin_file)
            if isinstance(split_matrix, str): # In some cases, what was stored was a filename!
              split_matrix = load_splits(split_matrix, kwargs['num_splits_second'])
          if verbose:
            end("Loaded "+os.path.basename(split_load_filename))
    else:
      if axis == 0:
        split_matrix = large_matrix[begin_i:end_i]
      elif axis == 1:
        split_matrix = large_matrix[:, begin_i:end_i]
      else:
        split_matrix = np.split(np.asarray(large_matrix), num_splits, axis=axis)

    start()
    transformed_split = csr_matrix(split_matrix)
    # lprint(transformed_split)
    if not hasattr(pipeline, '__iter__'):
      pipeline = [pipeline]

    # This is where the jobs in pipeline are executed
    # original_first = None
    lprint(transformed_split, 'next split')
    if (splitting_second or ('splitting_second' in kwargs and kwargs['splitting_second'])) and split_i > 0: # Need to switch matrices for every split, after the first split (already flipped)
      # transformed_split, kwargs = flip_matrices_job(transformed_split, **kwargs)
      # lprint(original_first, 'Saving original_first')
      kwargs['second'] = original_first
    for job in pipeline:
      if '_2op_' in job.__name__: # Does this job involve a second matrix?
        # Remove _2op from function name and add function to pipeline
        second_pipeline = [globals()[job.__name__.replace('_2op', '')]]
        transformed_split, kwargs = split_second(second_pipeline, transformed_split, **kwargs)
      else:
        transformed_split, kwargs = job(transformed_split, **kwargs)
        lprint(transformed_split, 'performed job '+str(job.__name__))
    #     aprint(transformed_split, 'split output')
    # aprint(transformed_split, 'final split output')
    # if original_first is not None:
    #   kwargs['second'] = original_first
    del split_matrix
    if num_splits > 1:
      split_save_filename = add_file_suffix(save_filename, str(begin_i)+"-"+str(end_i))
      if save_transformed and save_filename:
        with open(split_save_filename, 'wb') as save_bin_file:
          pickle.dump(transformed_split, save_bin_file)
        if verbose:
          end("Ran pipeline and stored result to file "+split_save_filename+"\n\t")
        
      else: # If the matrix isn't being saved, you must want it to be stacked and returned?
        if transformed_matrix is None:
          transformed_matrix = transformed_split
        else:
          if axis == 1:
            transformed_matrix = hstack([transformed_matrix, transformed_split])
          else:
            transformed_matrix = vstack([transformed_matrix, transformed_split])
        if verbose:
          end("Ran pipeline and stacked result\n\t")
        # lprint(transformed_matrix)
      if verbose is not False:
        end("Completed matrix indices "+str(begin_i)+" to "+str(end_i)+" on axis "+str(axis))
      del transformed_split
    else: 
      transformed_matrix = transformed_split
      # end("Ran pipeline for whole matrix")
      split_save_filename = save_filename
      if save_transformed and save_filename:
        with open(split_save_filename, 'wb') as save_bin_file:
          pickle.dump(transformed_matrix, save_bin_file)
      return transformed_matrix

  if stack_transformed and save_transformed and num_splits > 1:
    transformed_matrix = None
    for offset_i in range(len(offsets)-1):
      begin_i = offsets[offset_i]
      end_i = offsets[offset_i+1]
      if verbose:
        start()
      split_filename = add_file_suffix(save_filename, str(begin_i)+"-"+str(end_i))
      if os.path.isfile(split_filename):
        transformed_split = vload(split_filename)
      else:
        transformed_split = load_splits(transformed_split, num_splits)
      if isinstance(transformed_split, str): # In some cases, what was stored was a filename!
        transformed_split = load_splits(transformed_split, kwargs['num_splits_second'])
      if transformed_matrix is None:
        transformed_matrix = transformed_split
      else:
        try:
          if isinstance(transformed_split, np.ndarray):
            transformed_matrix = np.stack((np.asarray(transformed_matrix), transformed_split), axis=axis)
          else:
            if axis == 1:
              transformed_matrix = hstack([transformed_matrix, transformed_split])
            else:
              transformed_matrix = vstack([transformed_matrix, transformed_split])
        except TypeError as e:
          eprint(e)
          # aprint(transformed_matrix)
          # aprint(transformed_split)
      if verbose:
        end("Stacked transformed split from "+split_filename)
        # aprint(transformed_matrix)
        # aprint(transformed_split)
      del transformed_split
    # lprint(transformed_matrix)
    if save_transformed and save_filename:
      with open(save_filename, 'wb') as bin_file:
        pickle.dump(transformed_matrix, bin_file)
    return transformed_matrix
  elif transformed_matrix is not None:
    return transformed_matrix
  else:
    return save_filename
