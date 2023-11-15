import polars as pl
import numpy as np
import tensorflow as tf
import tqdm as tqdm

# data generator function for time series data
class DataGenerator:
    def __init__(self, data, lookback, horizon, min_index, max_index, training, encoder, batch_size, step, n_input, n_output, col_range_start, col_range_end):
        self.data = data
        self.lookback = lookback
        self.horizon = horizon
        self.min_index = min_index
        self.max_index = max_index
        self.training = training
        self.encoder = encoder
        self.batch_size = batch_size
        self.step = step
        self.n_input = n_input
        self.n_output = n_output
        self.col_range_start = col_range_start
        self.col_range_end = col_range_end
        self.index = 0

        if self.training:
            self.name = "data/train_indices.npz"
        else:
            self.name = "data/test_indices.npz"
        
        import os.path
        if os.path.isfile(self.name):
            # load all_indices from cache if it exists
            self.all_indices = np.load(self.name)["arr_0"]

        else:
            # generate all the possible indices in 3 dimentional array
            self.all_indices = []
            from tqdm import tqdm
            for i in tqdm(range(self.min_index, self.max_index + 1), desc="Generating indices"):
                row_info = self.data[i, ["year", "state", "place"]]
                indices = self.data.filter(
                    (pl.col("year") <= (row_info["year"][0] + self.horizon)) &
                    (pl.col("year") >= (row_info["year"][0] - self.lookback)) &
                    (pl.col("state") == row_info["state"][0]) &
                    (pl.col("place") == row_info["place"][0])
                )[:, self.col_range_start:self.col_range_end].to_numpy() 
                self.all_indices.append(indices)

            # transform the list to numpy array
            self.all_indices = np.array(self.all_indices)
            # save self.all_indices as parquet file
            np.savez_compressed(self.name, self.all_indices)

        self.random_range = np.arange(0, self.all_indices.shape[0])

    def __call__(self):
        while True:
            if self.training: # if training, use random sampling
                if len(self.random_range) < self.batch_size:
                    rows = self.random_range
                    remaining = self.batch_size - len(self.random_range)
                    self.random_range = np.arange(0, self.all_indices.shape[0])
                    remaining_rows = np.random.choice(
                        self.random_range, remaining)
                    rows = np.append(rows, remaining_rows)
                    self.random_range = np.setdiff1d(
                        self.random_range, remaining_rows)
                else:
                    rows = np.random.choice(self.random_range, self.batch_size)
                    self.random_range = np.setdiff1d(self.random_range, rows)
            else: # if validation, use sequential sampling
                if self.index >= self.all_indices.shape[0]:
                    self.index = 0
                if self.index + self.batch_size - 1 >= self.all_indices.shape[0]:
                    rows = np.arange(self.index, self.all_indices.shape[0])
                    rows = np.append(rows, np.arange(
                        0, (self.batch_size - len(rows))))
                else:
                    rows = np.arange(self.index, min(
                        self.index + self.batch_size , self.all_indices.shape[0]))

                self.index += self.batch_size
            lookback_count = self.lookback // self.step
            x = self.all_indices[rows, 0:lookback_count + 1, :]
            y = self.all_indices[rows, lookback_count + 1:lookback_count + self.horizon + 1, :]         

            
            if (self.encoder == True):
                yield x

            elif (self.horizon == 1):
                y = y.reshape((self.batch_size, self.n_output))
                yield x, y

            else:
                yield x, y

    def reset(self):
        self.index = 0
        self.random_range = np.arange(0, self.all_indices.shape[0])



def tf_init_config():
    # limit the use of GPU memory by tensorflow
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    # make sure tensorflow utilize all threads
    tf.config.threading.set_inter_op_parallelism_threads(15)
    tf.config.threading.set_intra_op_parallelism_threads(15)

