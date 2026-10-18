import h5py
import numpy as np
import pandas as pd
import CONF1DENCE
import os
import json
import tarfile
import gzip
import torch

#Contains helpful data organizing functions

class Data_prep:
    #functions used to help prepare the training data
    def __init__(self, **kwargs):
        self.SEM_type = kwargs.get('SEM_type','DSEP_iso')
        self.verbose = kwargs.get('verbose',False)
        #Define path
        self.tracks_folder_path = kwargs.get('tracks_folder_path',None)
        if self.tracks_folder_path is None:
            self.tracks_folder_path = CONF1DENCE.__abspath__+'data/tracks/'
        self.vars_folder_path = kwargs.get('vars_folder_path',None)
        if self.vars_folder_path is None:
            self.vars_folder_path = CONF1DENCE.__abspath__+'data/MC_vars/'
            
        #read format file
        format_path = CONF1DENCE.__abspath__+'data/format/' + self.SEM_type + '_format.json'
        with open(format_path, 'r') as json_file:
            self.Data_format = json.load(json_file)
        
        #read DSEE_var if needed
        self.DSEE_var_path = kwargs.get('DSEE_var_path',None)
        if self.DSEE_var_path != None:
            with h5py.File(self.DSEE_var_path, 'r') as f:
                data = f['data'][:]
                labels = np.array([x.decode('utf-8') for x in f['columns'][:]])
            #find all columns needed except for age
            columns_need = self.Data_format['label_header']
            #rearange index
            elements_to_exclude = ['Age','mass']
            indices = []
            for x in columns_need:
                if x in elements_to_exclude:
                    continue
                if x not in labels:
                    raise ValueError(f"Element '{x}' needed is not found in var file.")
                #print(x, labels)
                indices.append(np.where(labels == x)[0][0])

            #rearange the data
            self.DSEE_var = pd.DataFrame(data=data[:,indices], columns=labels.astype(str)[indices])
            #if mass column exist, store it seperately
            if 'mass' in labels:
                mass_idx = np.where(labels == 'mass')[0][0]
                self.DSEE_mass = np.array(data[:,mass_idx])
            #store mc_num in an array for idx search
            mc_idx = np.where(labels == 'mc_num')[0][0]
            self.DSEE_mc_num = np.array(data[:,mc_idx]).astype(int)

    def quite_print(self, statement):
        if self.verbose == True:
            print(statement)

    def file_name_construct(self, file_type):
        file_name = ''
        if file_type == 'tracks':
            format_need = 'tracks_sample_format'
        elif file_type == 'var':
            format_need = 'var_sample_format'
        elif file_type == 'track':
            format_need = 'track_sample_format'
        elif file_type == 'training':
            format_need = 'training_sample_format'
        else:
            raise Exception('Sample format not specified')
        
        if format_need in self.Data_format:
            for name_piece in self.Data_format[format_need]:
                if name_piece in self.filename_vars:
                    file_name += self.filename_vars[name_piece]
                else:
                    file_name += name_piece
        return file_name
    
    def read_tracks(self,mass_range):
        if self.SEM_type == 'DSEP_iso':
            tracks_filename = self.file_name_construct('tracks')
            if os.path.exists(self.tracks_folder_path + tracks_filename) == False:
                print("{} empty or not found".format(tracks_filename))
                return None, None
            tracks_df = []
            mass_found = []
            #DSEP_iso files are stored in a .tar file and need to be extracted first
            with tarfile.open(self.tracks_folder_path + tracks_filename,'r') as tracks:
                #we need to go through the mass_range
                for mass in mass_range:
                    self.filename_vars['mass'] = mass
                    track_filename = self.file_name_construct('track')
                    try:
                        #track files are stored as .gz file and need to be extracted first
                        track_gz = tracks.extractfile(track_filename)
                        with gzip.open(track_gz, 'r') as track:
                            track_df = pd.read_csv(track, skiprows = 5, sep = r'\s+', names=self.Data_format['track_header'])
                            tracks_df.append(track_df)
                            mass_found.append(mass)
                    except:
                        print("{} empty or not found".format(track_filename))
            return tracks_df, mass_found
        if self.SEM_type == 'DSEE_iso':
            tracks_filename = self.file_name_construct('tracks')
            if os.path.exists(self.tracks_folder_path + tracks_filename) == False:
                print("{} empty or not found".format(tracks_filename))
                return None, None
            #DSEP_iso files are stored in a .tar file and need to be extracted first
            with tarfile.open(self.tracks_folder_path + tracks_filename,'r') as tracks:
                track_filename = self.file_name_construct('track')
                try:
                    #track files are stored as .gz file and need to be extracted first
                    track_gz = tracks.extractfile(track_filename)
                    with gzip.open(track_gz, 'r') as track:
                        track_df = pd.read_csv(track, skiprows = 5, sep = r'\s+', names=self.Data_format['track_header'])
                except:
                    print("{} empty or not found".format(track_filename))
            return track_df
        else:
            raise Exception("Method to read {} track files undefined".format(self.SEM_type))
    
    def read_var(self):
        if self.SEM_type == 'DSEP_iso':
            var_filename = self.file_name_construct('var')
            if os.path.exists(self.vars_folder_path + var_filename) == False:
                print("{} empty or not found".format(var_filename))
                return None
            var_vals = []
            with open(self.vars_folder_path + var_filename, 'r') as var:
                for line in var:
                    var_vals.append(float(line[:14]))
            return np.array(var_vals)
        else:
            raise Exception("Method to read {} var files undefined".format(self.SEM_type))

    def convert_to_hdf5_compressed(self, L_replace = False, **kwargs):
        #outdated, not able to be used for training
        #convert to densly compacted h5py file with all the information
        if self.SEM_type == 'DSEP_iso':
            #currently we only use the following structure
            # |-- mc_num            Group distinguished by MC_num
            #     |-- m0200         track file distinguished by inital mass
            #                       MC_vars are set as attributes for each group
            
            #define the set of files to read and convert
            self.filename_vars = kwargs.get('filename_vars',{'[Fe/H]':'227','mc_num':'10000','sign':'-','GC_name':'M15'})
            mass_range = kwargs.get(
                'mass_range',['0200', '0240', '0280', '0320', '0360', '0400', '0440', '0480', '0520', '0560', '0600', '0640', '0680', '0700', '0750', '0800', '0850', '0900', '0950', '1000', '1050', '1100', '1150', '1200', '1250', '1300', '1350', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2200', '2400', '2600', '2800', '3000']
            )
            #define the path to store HDF5 file
            self.HDF5_path = kwargs.get('HDF5_path',None)
            if self.HDF5_path == None:
                self.HDF5_path = CONF1DENCE.__abspath__+'data/training_data/'
            #reading tracks
            tracks_df = self.read_tracks(mass_range)
            var_vals = self.read_var()
            #construct hdf5 file
            training_filename = self.file_name_construct('training')
            with h5py.File(self.HDF5_path + training_filename, 'a') as training:
                #define group name
                grp_name = 'mc' + self.filename_vars['mc_num']
                if grp_name in training:
                    if L_replace == True:
                        del training[grp_name]
                        self.quite_print("Group {} existed and is replaced".format(grp_name))
                    else:
                        self.quite_print("Group {} existed and is not replaced".format(grp_name))
                        return 
                grp = training.create_group(grp_name)
                #set MC params as attribution
                for i, mc_param in enumerate(self.Data_format['MC_params']):
                    grp.attrs[mc_param] = var_vals[i]
                #record all the data
                for i, mass in enumerate(mass_range):
                    try:
                        grp.create_dataset('m'+mass,data=tracks_df[i].values)
                    except:
                        self.quite_print("mc{}'s m{} file is empty or missing".format(self.filename_vars['mc_num'],mass))
        else:
            raise Exception("Method to read {} track files undefined".format(self.SEM_type))
    
    def convert_to_hdf5_training(self, **kwargs):
        self.chunk_size = kwargs.get('chunk_size',100)
        if self.SEM_type == 'DSEP_iso':
            #define the set of files to read and convert
            self.filename_vars = kwargs.get('filename_vars',{'[Fe/H]':'227','mc_num':'10000','sign':'-','GC_name':'M15'})
            mass_range = kwargs.get(
                'mass_range',['0200', '0240', '0280', '0320', '0360', '0400', '0440', '0480', '0520', '0560', '0600', '0640', '0680', '0700', '0750', '0800', '0850', '0900', '0950', '1000', '1050', '1100', '1150', '1200', '1250', '1300', '1350', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2200', '2400', '2600', '2800', '3000']
            )
            #define the path to store HDF5 file
            self.HDF5_path = kwargs.get('HDF5_path',None)
            if self.HDF5_path == None:
                self.HDF5_path = CONF1DENCE.__abspath__+'data/training_data/'
            #reading tracks
            tracks_df, mass_found = self.read_tracks(mass_range)
            if tracks_df != None:
                var_vals = self.read_var()
                #construct hdf5 file
                training_filename = self.file_name_construct('training')
                data_cols = sum(2 if scaler == 'mantissa_exponent' else 1 for scaler in self.Data_format['data_scaler'])
                label_cols = sum(2 if scaler == 'mantissa_exponent' else 1 for scaler in self.Data_format['label_scaler'])
                #determine the index of SBC:
                SBC_idx = self.Data_format['MC_params'].index('Surface Boundary Condition')

                with h5py.File(self.HDF5_path + training_filename, 'a') as training:
                    #define 3 groups based on the Surface boundary condition
                    grp_names = ['PHOENIX', 'Eddington', 'Krishna-Swamy']
                    for grp_name in grp_names:
                        if grp_name not in training:
                            grp = training.create_group(grp_name)
                            grp.create_dataset(
                                    'data',
                                    shape=(0, data_cols),  # Initial shape
                                    maxshape=(None, data_cols),  # Can grow indefinitely along the first dimension
                                    chunks=(self.chunk_size, data_cols),  # Store data in chunks of chunk_rows x cols
                                    dtype='float32'  # Data type
                            )
                            grp.create_dataset(
                                    'label',
                                    shape=(0, label_cols),  # Initial shape
                                    maxshape=(None, label_cols),  # Can grow indefinitely along the first dimension
                                    chunks=(self.chunk_size, label_cols),  # Store data in chunks of chunk_rows x cols
                                    dtype='float32'  # Data type
                            )

                    #now determine which dataset to append based on the surface boundary condition
                    if int(var_vals[SBC_idx]) == 5:
                        grp = training['PHOENIX']
                    elif int(var_vals[SBC_idx]) == 0:
                        grp = training['Eddington']
                    elif int(var_vals[SBC_idx]) == 1:
                        grp = training['Krishna-Swamy']
                    dataset = grp['data']
                    labelset = grp['label']

                    # pre-calculate total rows for all masses and resize once
                    total_new_rows = sum(len(df) for df in tracks_df)
                    start_idx = dataset.shape[0]
                    dataset.resize((start_idx + total_new_rows, dataset.shape[1]))
                    labelset.resize((start_idx + total_new_rows, labelset.shape[1]))

                    offset = start_idx
                    for i, mass in enumerate(mass_found):
                        try:
                            track = tracks_df[i]
                            ages = track['Age'].values
                            data_append = data_transform.norm_data(
                                track[self.Data_format['data_header']].values,
                                self.Data_format['data_scaler'],
                                self.Data_format['data_scale']
                            ).astype('float32')
                            label_vals = np.column_stack([
                                np.repeat(np.delete(var_vals, SBC_idx)[None, :], len(ages), axis=0),
                                ages.reshape(len(ages), 1),
                                np.full((len(ages), 1), int(mass) / 1000)
                            ])
                            label_append = data_transform.norm_data(
                                label_vals,
                                self.Data_format['label_scaler'],
                                self.Data_format['label_scale']
                            ).astype('float32')

                            dataset[offset:offset + len(ages)] = data_append
                            labelset[offset:offset + len(ages)] = label_append
                            offset += len(ages)
                        except Exception:
                            pass
        elif self.SEM_type == 'DSEE_iso':
            #there is no fixed grid of mass. We read each track individually
            #define the set of files to read and convert
            self.filename_vars = kwargs.get('filename_vars',{'mc_num':'10000', 'mass':'0200','GC_name':'DSEE'})
            #define the path to store HDF5 file
            self.HDF5_path = kwargs.get('HDF5_path',None)
            if self.HDF5_path == None:
                self.HDF5_path = CONF1DENCE.__abspath__+'data/training_data/'
            #reading tracks
            track_df = self.read_tracks(None)
            if isinstance(track_df, pd.DataFrame) and not track_df.empty:
                var_idx = np.where(int(self.filename_vars['mc_num']) == self.DSEE_mc_num)[0]
                var_vals = self.DSEE_var.iloc[var_idx].values
                mass = self.DSEE_mass[var_idx]
                #construct hdf5 file
                training_filename = self.file_name_construct('training')
                data_cols = sum(2 if scaler == 'mantissa_exponent' else 1 for scaler in self.Data_format['data_scaler'])
                label_cols = sum(2 if scaler == 'mantissa_exponent' else 1 for scaler in self.Data_format['label_scaler'])

                with h5py.File(self.HDF5_path + training_filename, 'a') as training:
                    #define 3 groups based on the Surface boundary condition
                    if 'data' not in training:
                        training.create_dataset(
                                'data',
                                shape=(0, data_cols),  # Initial shape
                                maxshape=(None, data_cols),  # Can grow indefinitely along the first dimension
                                chunks=(self.chunk_size, data_cols),  # Store data in chunks of chunk_rows x cols
                                dtype='float32'  # Data type
                        )
                    if 'label' not in training:
                        training.create_dataset(
                                'label',
                                shape=(0, label_cols),  # Initial shape
                                maxshape=(None, label_cols),  # Can grow indefinitely along the first dimension
                                chunks=(self.chunk_size, label_cols),  # Store data in chunks of chunk_rows x cols
                                dtype='float32'  # Data type
                        )

                    dataset = training['data']
                    labelset = training['label']

                    try:
                        # Age is not in the label.h5 file. We need to add that to the label
                        ages = track_df['Age'].values
                        data_append = data_transform.norm_data(
                            track_df[self.Data_format['data_header']].values,
                            self.Data_format['data_scaler'],
                            self.Data_format['data_scale']
                        ).astype('float32')
                        label_append = data_transform.norm_data(
                            np.hstack((
                                np.tile(var_vals, (len(data_append), 1)),
                                ages.reshape(len(ages), 1),
                                np.tile(np.array([mass]), (len(data_append), 1)))
                            ),
                            self.Data_format['label_scaler'],
                            self.Data_format['label_scale']
                        ).astype('float32')    

                        #append data
                        new_shape = (dataset.shape[0] + data_append.shape[0], dataset.shape[1])
                        dataset.resize(new_shape)
                        dataset[-data_append.shape[0]:] = data_append
                        new_shape = (labelset.shape[0] + label_append.shape[0], labelset.shape[1])
                        labelset.resize(new_shape)
                        labelset[-label_append.shape[0]:] = label_append
                    except Exception:
                        pass
        else:
            raise Exception("Method to read {} track files undefined".format(self.SEM_type))
        
class performance:
    def __init__(self):
        pass
    
    @staticmethod
    def check_cuda(verbose=False):
        """
        Check available computing devices and return the best option.
        Priority: CUDA > MPS > CPU
        
        Note: MPS (Metal Performance Shaders) is for Apple Silicon Macs.
        Some operations may not be fully supported on MPS yet.
        """
        # Check CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            print(f"✓ CUDA is available. Number of devices: {torch.cuda.device_count()}")
            if verbose:
                for i in range(torch.cuda.device_count()):
                    print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            device = torch.device('cuda:0')
            print(f'Using device: {device}')
            return device
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✓ MPS (Apple Silicon) is available")
            print("⚠️  WARNING: MPS support in PyTorch is still maturing.")
            print("   If you experience issues, consider using CPU by setting")
            print("   device='cpu' manually or disabling MPS.")
            device = torch.device('mps')
            print(f'Using device: {device}')
            return device
        
        # Fallback to CPU
        print("ℹ️  No GPU acceleration available. Using CPU.")
        print("   This is normal for systems without NVIDIA or Apple Silicon GPUs.")
        print("   Performance will be slower but the code should still work.")
        device = torch.device('cpu')
        print(f'Using device: {device}')
        return device

    @staticmethod
    def get_available_gpu_memory(device):
        """
        Returns available GPU memory for the specified device.
        Only works for CUDA devices.
        """
        if not isinstance(device, torch.device):
            device = torch.device(device)
            
        if device.type != 'cuda':
            raise ValueError(f"get_available_gpu_memory only works with CUDA devices, got {device.type}")
            
        torch.cuda.empty_cache()
        gpu_properties = torch.cuda.get_device_properties(device)
        total_memory = gpu_properties.total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = total_memory - reserved_memory - allocated_memory
        return free_memory

    def estimate_optimal_batch_size(self, device, model, input_sample, **kwargs):
        """
        Estimates the largest possible batch size to fit in GPU memory.
        Only works for CUDA devices. Returns a default batch size for CPU/MPS.

        Parameters:
        - device: torch device (cuda, cpu, or mps)
        - model: your trained generative model (torch.nn.Module).
        - input_sample: a sample input tensor of shape [1, *input_shape].
        - safety_margin: fraction of memory to actually use (to prevent OOM errors).

        Returns:
        - Optimal batch size as an integer.
        """
        if not isinstance(device, torch.device):
            device = torch.device(device)
            
        # For non-CUDA devices, return a reasonable default batch size
        if device.type != 'cuda':
            default_batch_size = kwargs.get('default_batch_size', 512)
            print(f"ℹ️  Batch size estimation only works for CUDA. Using default: {default_batch_size}")
            return default_batch_size

        safety_margin = kwargs.get('safety_margin',0.9)

        # Measure memory per sample
        with torch.no_grad():
            torch.cuda.empty_cache()
            initial_mem = torch.cuda.memory_allocated(device)
            
            input_sample = input_sample.to(device)
            output = model(input_sample)
            
            final_mem = torch.cuda.memory_allocated(device)
            memory_per_sample = final_mem - initial_mem
            
            # Safety check if memory_per_sample is extremely small or zero
            if memory_per_sample <= 0:
                memory_per_sample = 1e6  # Approximate minimum to avoid division by zero

        available_mem = self.get_available_gpu_memory(device) * safety_margin

        optimal_batch_size = int(available_mem / memory_per_sample)

        return optimal_batch_size

class data_transform:    

    #define the mantissa exponent method to convert the age to 2*fp32 variables to avoid accuracy lost during training
    @staticmethod
    def mantissa_exponent_forward(col, exp_min, exp_max):
        """
        Apply mantissa-exponent transformation to avoid precision loss.
        Works with both NumPy arrays and PyTorch tensors.
        
        Returns:
            tuple: (mantissa_scaled, exponent_scaled)
        """
        # Detect if input is PyTorch tensor or NumPy array
        is_torch = hasattr(col, 'is_cuda')
        
        if is_torch:
            exponent = torch.floor(torch.log10(torch.clamp(col, min=1e-10)))
            mantissa = col / torch.pow(10.0, exponent)
        else:
            exponent = np.floor(np.log10(np.clip(col, 1e-10, None)))
            mantissa = col / (10 ** exponent)
        
        mantissa_scaled = ((mantissa - 1) / 9) * 10 - 5
        exponent_scaled = ((exponent - exp_min) / (exp_max - exp_min)) * 10 - 5
        
        return mantissa_scaled, exponent_scaled
    
    @staticmethod
    def mantissa_exponent_inverse(mantissa_scaled, exponent_scaled, exp_min, exp_max):
        """
        Inverse mantissa-exponent transformation.
        Works with both NumPy arrays and PyTorch tensors.
        
        Returns:
            original values
        """
        # Detect if input is PyTorch tensor or NumPy array
        is_torch = hasattr(mantissa_scaled, 'is_cuda')
        
        mantissa = ((mantissa_scaled + 5) / 10) * 9 + 1
        exponent = ((exponent_scaled + 5) / 10) * (exp_max - exp_min) + exp_min
        
        if is_torch:
            return mantissa * torch.pow(10.0, exponent)
        else:
            return mantissa * (10 ** exponent)
    
    @staticmethod
    def norm_data(data, scalers, scalers_params):
        norm_data_list = []
        for i, (scaler, scale_params) in enumerate(zip(scalers, scalers_params)):
            col = data[:, i]
            if scaler == 'log':
                norm_col = np.log10(np.clip(col, 1e-10, None))
                norm_data_list.append(norm_col[:, None])
            elif scaler == 'divide_log':
                norm_col = np.log10(np.clip(col / scale_params[0], 1e-10, None))
                norm_data_list.append(norm_col[:, None])
            elif scaler == 'log_rescale':
                norm_col = (np.log10(np.clip(col, 1e-10, None)) - scale_params[0]) / (scale_params[1] - scale_params[0])
                norm_data_list.append(norm_col[:, None])
            elif scaler == 'linear_rescale':
                norm_col = (col - scale_params[0]) / (scale_params[1] - scale_params[0])
                norm_data_list.append(norm_col[:, None])
            elif scaler == 'normal':
                norm_col = (col - scale_params[0]) / scale_params[1]
                norm_data_list.append(norm_col[:, None])
            elif scaler == 'mantissa_exponent':
                exp_min, exp_max = scale_params[0], scale_params[1]
                mantissa_scaled, exponent_scaled = data_transform.mantissa_exponent_forward(col, exp_min, exp_max)
                norm_data_list.extend([mantissa_scaled[:, None], exponent_scaled[:, None]])
            elif scaler == 'None':
                norm_data_list.append(col[:, None])
            else:
                raise ValueError(f"Scaler {scaler} not defined.")
        return np.hstack(norm_data_list)

    @staticmethod
    def inverse_norm_data(norm_data, scalers, scalers_params):
        inv_data_list = []
        idx = 0
        for scaler, scale_params in zip(scalers, scalers_params):
            if scaler == 'log':
                inv_col = np.power(10.0, norm_data[:, idx])
                inv_data_list.append(inv_col[:, None])
                idx += 1
            elif scaler == 'divide_log':
                inv_col = np.power(10.0, norm_data[:, idx]) * scale_params[0]
                inv_data_list.append(inv_col[:, None])
                idx += 1
            elif scaler == 'log_rescale':
                inv_col = np.power(10.0, norm_data[:, idx] * (scale_params[1] - scale_params[0]) + scale_params[0])
                inv_data_list.append(inv_col[:, None])
                idx += 1
            elif scaler == 'linear_rescale':
                inv_col = norm_data[:, idx] * (scale_params[1] - scale_params[0]) + scale_params[0]
                inv_data_list.append(inv_col[:, None])
                idx += 1
            elif scaler == 'normal':
                inv_col = norm_data[:, idx] * scale_params[1] + scale_params[0]
                inv_data_list.append(inv_col[:, None])
                idx += 1
            elif scaler == 'mantissa_exponent':
                mantissa_scaled = norm_data[:, idx]
                exponent_scaled = norm_data[:, idx + 1]
                exp_min, exp_max = scale_params[0], scale_params[1]
                inv_col = data_transform.mantissa_exponent_inverse(mantissa_scaled, exponent_scaled, exp_min, exp_max)
                inv_data_list.append(inv_col[:, None])
                idx += 2
            elif scaler == 'None':
                inv_data_list.append(norm_data[:, idx][:, None])
                idx += 1
            else:
                raise ValueError(f"Inverse scaler {scaler} not defined.")
        return np.hstack(inv_data_list)

    @staticmethod
    def norm_data_torch(data, scalers, scalers_params):
        norm_data_list = []
        for i, (scaler, scale_params) in enumerate(zip(scalers, scalers_params)):
            col = data[:, i]
            if scaler == 'log':
                norm_col = torch.log10(torch.clamp(col, min=1e-10))
                norm_data_list.append(norm_col.unsqueeze(1))
            elif scaler == 'divide_log':
                norm_col = torch.log10(torch.clamp(col / scale_params[0], min=1e-10))
                norm_data_list.append(norm_col.unsqueeze(1))
            elif scaler == 'log_rescale':
                norm_col = (torch.log10(torch.clamp(col, min=1e-10)) - scale_params[0]) / (scale_params[1] - scale_params[0])
                norm_data_list.append(norm_col.unsqueeze(1))
            elif scaler == 'linear_rescale':
                norm_col = (col - scale_params[0]) / (scale_params[1] - scale_params[0])
                norm_data_list.append(norm_col.unsqueeze(1))
            elif scaler == 'normal':
                norm_col = (col - scale_params[0]) / scale_params[1]
                norm_data_list.append(norm_col.unsqueeze(1))
            elif scaler == 'mantissa_exponent':
                exp_min, exp_max = scale_params[0], scale_params[1]
                mantissa_scaled, exponent_scaled = data_transform.mantissa_exponent_forward(col, exp_min, exp_max)
                norm_data_list.extend([mantissa_scaled.unsqueeze(1), exponent_scaled.unsqueeze(1)])
            elif scaler == 'None':
                norm_data_list.append(col.unsqueeze(1))
            else:
                raise ValueError(f"Scaler {scaler} not defined.")
        return torch.cat(norm_data_list, dim=1)

    @staticmethod
    def inverse_norm_data_torch(norm_data, scalers, scalers_params):
        inv_data_list = []
        idx = 0
        for scaler, scale_params in zip(scalers, scalers_params):
            if scaler == 'log':
                inv_col = torch.pow(10.0, norm_data[:, idx])
                inv_data_list.append(inv_col.unsqueeze(1))
                idx += 1
            elif scaler == 'divide_log':
                inv_col = torch.pow(10.0, norm_data[:, idx]) * scale_params[0]
                inv_data_list.append(inv_col.unsqueeze(1))
                idx += 1
            elif scaler == 'log_rescale':
                inv_col = torch.pow(10.0, norm_data[:, idx] * (scale_params[1] - scale_params[0]) + scale_params[0])
                inv_data_list.append(inv_col.unsqueeze(1))
                idx += 1
            elif scaler == 'linear_rescale':
                inv_col = norm_data[:, idx] * (scale_params[1] - scale_params[0]) + scale_params[0]
                inv_data_list.append(inv_col.unsqueeze(1))
                idx += 1
            elif scaler == 'normal':
                inv_col = norm_data[:, idx] * scale_params[1] + scale_params[0]
                inv_data_list.append(inv_col.unsqueeze(1))
                idx += 1
            elif scaler == 'mantissa_exponent':
                mantissa_scaled = norm_data[:, idx]
                exponent_scaled = norm_data[:, idx + 1]
                exp_min, exp_max = scale_params[0], scale_params[1]
                inv_col = data_transform.mantissa_exponent_inverse(mantissa_scaled, exponent_scaled, exp_min, exp_max)
                inv_data_list.append(inv_col.unsqueeze(1))
                idx += 2
            elif scaler == 'None':
                inv_data_list.append(norm_data[:, idx].unsqueeze(1))
                idx += 1
            else:
                raise ValueError(f"Inverse scaler {scaler} not defined.")
        return torch.cat(inv_data_list, dim=1)


    def easy_transform(self,data, scalers = None, tensor = False, data_type='input', SEM_type = 'DSEE_iso', **kwargs):
        SEM_type = kwargs.get('SEM_type','DSEE_iso')
        Data_format = kwargs.get('Data_format',None)
        if Data_format == None:
            format_path = kwargs.get('format_path',CONF1DENCE.__abspath__+'data/format/' + SEM_type + '_format.json')
            #read format file
            with open(format_path, 'r') as json_file:
                Data_format = json.load(json_file)
                
        #make sure data is 2D
        if np.array(data).ndim == 1:
            data = np.array(data).reshape(1,-1)

        if scalers == None:
            #check if the length is the same
            if data_type == 'input':
                input_len = np.shape(data)[-1]
                n_labels = len(Data_format['label_header'])
                if n_labels != input_len:
                    raise ValueError(f"Input data has '{input_len}' columns which does not match number of label headers. Make sure the shape of the data is correct or try to set scalers instead.")
                else:
                    if tensor:
                        return self.norm_data_torch(data, Data_format['label_scaler'], Data_format['label_scale'])
                    else:
                        return self.norm_data(data, Data_format['label_scaler'], Data_format['label_scale'])
            else:
                output_len = np.shape(data)[-1]
                n_data = len(Data_format['data_header'])
                if n_data != output_len:
                    raise ValueError(f"Output data has '{output_len}' columns which does not match number of data headers. Make sure the shape of the data is correct or try to set scalers instead.")
                else:
                    if tensor:
                        return self.inverse_norm_data_torch(data, Data_format['data_scaler'], Data_format['data_scale'])
                    else:
                        return self.inverse_norm_data(data, Data_format['data_scaler'], Data_format['data_scale'])
        
        else:
            #cross match proposed scaler names with format lists
            if data_type == 'input':
                indices = [Data_format['label_header'].index(item) for item in data]
                if tensor:
                    return self.norm_data_torch(data, Data_format['label_scaler'][indices], Data_format['label_scale'][indices])
                else:
                    return self.norm_data(data, Data_format['label_scaler'][indices], Data_format['label_scale'][indices])
            else:
                indices = [Data_format['data_header'].index(item) for item in data]
                if tensor:
                    return self.inverse_norm_data_torch(data, Data_format['data_scaler'][indices], Data_format['data_scale'][indices])
                else:
                    return self.inverse_norm_data(data, Data_format['data_scaler'][indices], Data_format['data_scale'][indices])