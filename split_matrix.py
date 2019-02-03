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

# Multiplies matrix by first argument in kwargs
def multiply_job(matrix, **kwargs):
  try:
    print("multiply_job")
    second_matrix = kwargs['second']
    # lprint(matrix)
    # lprint(second_matrix)
    if isinstance(second_matrix, str):
      if 'num_splits_second' in kwargs and kwargs['num_splits_second'] and kwargs['num_splits_second'] > 1:
        return split_second(multiply_flip_job, matrix, **kwargs)
      else:
        with open(second_matrix, 'rb') as bin_file:
          second_matrix = pickle.load(bin_file)
    # # lprint(matrix)
    # # lprint(second_matrix)
    # if hasattr(matrix, 'dot'):
    #   prod = matrix.dot(second_matrix)
    # else:
    prod = matrix * second_matrix
    del second_matrix
    # # lprint(prod)
    return prod, kwargs
  except ValueError as e:
    eprint("ValueError in multiply_job")
    eprint(e)
    # lprint(matrix)
    # lprint(kwargs['second'], name='kwargs[second]')
    # lprint(second_matrix)
  return None

# Multiplies first argument in kwargs by matrix
# Flips the typical lhs and rhs
def multiply_flip_job(matrix, **kwargs):
  try:
    second_matrix = kwargs['second']
    if isinstance(second_matrix, str):
      if 'num_splits_second' in kwargs and kwargs['num_splits_second'] and kwargs['num_splits_second'] > 1:
        return split_second(multiply_job, matrix, **kwargs)
      else:
        with open(second_matrix, 'rb') as bin_file:
          second_matrix = pickle.load(bin_file)
    # # lprint(matrix)
    # # lprint(second_matrix)
    # if hasattr(second_matrix, 'dot'):
    #   prod = second_matrix.dot(matrix)
    # else:
    prod = second_matrix * matrix
    del second_matrix
    # # lprint(prod)
    return prod, kwargs
  except ValueError as e:
    eprint("ValueError in multiply_job")
    eprint(e)
    # lprint(kwargs['second'], name='kwargs[second]')
    # lprint(second_matrix)
    # lprint(matrix)
  return None

# Flip the first matrix (matrix) and second (kwargs['second'])
#   in preparation for future operations.
def flip_matrices_job(matrix, **kwargs):
  first = kwargs['second']
  kwargs['second'] = matrix
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
          # vprint(offset_splits, name='Could not convert to ints')
          # vprint(filename)
  
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
  # pipeline_second.append(flip_matrices_job)
  # Instead of flip_matrix_job, use a flip specific job.
  pipeline_second.extend(pipeline)
  # pipeline_second.append(flip_matrices_job) # Flip back and forth each split
  kwargs_second['second'] = matrix # Flip the matrices, to be flipped back after splitting
  return split_matrix(pipeline_second, large_matrix=kwargs['second'], **kwargs_second), kwargs

# Splits large_matrix by axis (default split by rows).
# Use kwargs to arguments needed by jobs in pipeline and for split operations on a second matrix.
# For a second matrix, end arguments with '_second'.
def split_matrix(pipeline, save_filename=False, num_splits=1, large_matrix=None, matrix_len=None, 
                           load_filename="", save_transformed=True, stack_transformed=False, axis=0, **kwargs):

  # vprint(pipeline)
  # vprint(kwargs, name='kwargs')
  if 'second' in kwargs and not isinstance(kwargs['second'], str): # Not a file
    kwargs['second'] = csr_matrix(kwargs['second'])

  offsets = []
  if large_matrix is None and num_splits == 1:
    with open(load_filename, 'rb') as bin_file:
      large_matrix = pickle.load(bin_file)
  
  if isinstance(large_matrix, str):
    load_filename = large_matrix
    large_matrix = None

  if large_matrix is not None:
    large_matrix = csr_matrix(large_matrix)
    # lprint(large_matrix, name='large matrix')
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
  for offset_i in range(len(offsets)-1):
    begin_i = offsets[offset_i]
    end_i = offsets[offset_i+1]
    
    split_matrix = None
    if large_matrix is None:
      if num_splits > 1:
          split_load_filename = add_file_suffix(load_filename, str(begin_i)+"-"+str(end_i))
          start()
          with open(split_load_filename, 'rb') as load_bin_file:
            split_matrix = pickle.load(load_bin_file)
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
    if not hasattr(pipeline, '__iter__'):
      pipeline = [pipeline]
    for job in pipeline:
      transformed_split, kwargs = job(transformed_split, **kwargs)
    del split_matrix
    if num_splits > 1:
      split_save_filename = add_file_suffix(save_filename, str(begin_i)+"-"+str(end_i))
      if save_transformed and save_filename:
        with open(split_save_filename, 'wb') as save_bin_file:
          pickle.dump(transformed_split, save_bin_file)
        end("Ran pipeline and stored result to file "+split_save_filename+"\n\t")
        
      else: # If the matrix isn't being saved, you must want it to be stacked and returned?
        if transformed_matrix is None:
          transformed_matrix = transformed_split
        else:
          if axis == 1:
            transformed_matrix = hstack([transformed_matrix, transformed_split])
          else:
            transformed_matrix = vstack([transformed_matrix, transformed_split])
        end("Ran pipeline and stacked result\n\t")
        # lprint(transformed_matrix)
      del transformed_split
    else: 
      transformed_matrix = transformed_split
      end("Ran pipeline for whole matrix")
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
      start()
      split_filename = add_file_suffix(save_filename, str(begin_i)+"-"+str(end_i))
      with open(split_filename, 'rb') as bin_file:
        transformed_split = pickle.load(bin_file)
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
    if save_transformed and save_filename:
      with open(save_filename, 'wb') as bin_file:
        pickle.dump(transformed_matrix, bin_file)
    return transformed_matrix
  elif transformed_matrix is not None:
    return transformed_matrix
  else:
    return save_filename
