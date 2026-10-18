import torch
import zuko
import CONF1DENCE
import json
import numpy as np
from ..utils.helper import data_transform, performance
from ..train.train import TrainMod
import os
import pandas as pd



class observables:
    def __init__(self, **kwargs):
        #read other files for mock observation for convinience
        # Store device if provided (will be set by emulator subclass)
        self._device = kwargs.get('device', None)
        
        bc_table_path = kwargs.get('BC_table_path', None)
        bc_format_path = kwargs.get('BC_format_path', None)
        if (bc_table_path is not None) and (bc_format_path is not None):
            self.BC_table(bc_table_path, bc_format_path)
        
        
        as_comp_path = kwargs.get('AS_comp_path', None)
        as_phot_paths = kwargs.get('AS_phot_paths', None)
        as_format_path = kwargs.get('AS_format_path', None)
        if (as_phot_paths is not None) and (as_format_path is not None):
            self.AS_test(as_phot_paths, as_format_path, as_comp_path = as_comp_path)
    
    def BC_table(self,bc_table_path, bc_format_path):
        #load trained BC_table
        # Use the device from emulator if available, otherwise determine locally
        device = self._device if self._device is not None else performance.check_cuda()
        self.bc_table = torch.load(bc_table_path, map_location=device, weights_only=False)
        # Move model to correct device
        if hasattr(self.bc_table, 'to'):
            self.bc_table = self.bc_table.to(device)
        #load bc_format
        with open(bc_format_path, 'r') as json_file:
            self.bc_format = json.load(json_file)
    
    def AS_test(self, as_phot_paths, as_format_path, as_comp_path = None):
        #load trained AS test results
        # Use the device from emulator if available, otherwise determine locally
        device = self._device if self._device is not None else performance.check_cuda()
        self.as_phots = [torch.load(as_phot_path, map_location=device, weights_only=False) for as_phot_path in as_phot_paths]
        # Move models to correct device
        self.as_phots = [model.to(device) if hasattr(model, 'to') else model for model in self.as_phots]
        #load bc_format
        with open(as_format_path, 'r') as json_file:
            self.as_format = json.load(json_file)
        if as_comp_path != None:
            #load completeness test result
            self.as_comp = torch.load(as_comp_path, map_location=device, weights_only=False)
            # Move model to correct device
            if hasattr(self.as_comp, 'to'):
                self.as_comp = self.as_comp.to(device)

class emulator(observables):
    #The core emulator. Emulate stellar evolution models based on the input labels.
    def __init__(self, **kwargs):
        self.SEM_type = kwargs.get('SEM_type','DSEE_iso')

        # determine path to trained continous stellar evolution model
        flow_path = kwargs.get('flow_path', CONF1DENCE.__abspath__+'data/SEM/DSEE.model')
        
        #read format file
        SEM_format_path = CONF1DENCE.__abspath__+'data/format/' + self.SEM_type + '_format.json'
        with open(SEM_format_path, 'r') as json_file:
            self.SEM_format = json.load(json_file)

        #assign device - set this BEFORE calling super().__init__()
        # Allow manual device override, otherwise auto-detect
        if 'device' in kwargs:
            # User specified device - ensure it's a torch.device object
            device = kwargs['device']
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        else:
            # Auto-detect best available device
            self.device = performance.check_cuda()
            kwargs['device'] = self.device  # Pass device to parent class
        
        # Initialize parent class (observables) with device info
        super().__init__(**kwargs)

        #define dimensions for input and output parameters
        n_features = sum(2 if scaler == 'mantissa_exponent' else 1 for scaler in self.SEM_format['data_scaler'])
        n_context = sum(2 if scaler == 'mantissa_exponent' else 1 for scaler in self.SEM_format['label_scaler'])

        #setup zuko
        self.flow = zuko.flows.NSF(features=n_features,  context=n_context, transforms=10, hidden_features=[256] * 10).to(self.device)

        #read trained continous stellar evolution model
        checkpoint = torch.load(flow_path, map_location=self.device, weights_only=False)

        # Handle both saving formats: dict with state_dict or direct model
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format: saved as dictionary with model_state_dict
            self.flow.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Old format: saved as model directly
            self.flow = checkpoint.to(self.device)

    #buiding piplines to generate Mock observations
    #Currently, we can only generate synthetic CMD

    def search_data_and_labels(self, type, header):
        data_header = getattr(self, type)['data_header']
        label_header = getattr(self, type)['label_header']
        try:
            # Check if the header is in data_header
            index = data_header.index(header)
            return 'data', index
        except ValueError:
            pass  # Element not in list1, continue to check list2
        
        try:
            # Check if the element is in list2
            index = label_header.index(header)
            return 'label', index
        except:
            raise ValueError("{} is not part of {} data".format(header, type))

    def gen_theory_models(self, label, **kwargs):
        #generate stars in theoretical models
        #check whether labels are on the same device as the model
        if label.device != self.device:
            label = label.to(self.device, non_blocking=True)
        if hasattr(self, 'nsample'):
            data = self.flow(label).sample((self.nsample,)).permute(1, 0, 2)
            data = data.reshape(-1, data.shape[2])
        else:
            data = self.flow(label).sample()
        output_headers = kwargs.get('output_header','all')
        if output_headers == 'all':
            return data_transform.inverse_norm_data_torch(data,self.SEM_format['data_scaler'],self.SEM_format['data_scale'])
        else:
            combined_cols = []
            for header in output_headers:
                header_in, i = self.search_data_and_labels('SEM_format', header)
                source = data if header_in == 'data' else label
                combined_cols.append(data_transform.inverse_norm_data_torch(source[:, i].unsqueeze(1),[self.SEM_format[header_in + '_scaler'][i]] , [self.SEM_format[header_in +'_scale'][i]]))
            return torch.cat(combined_cols, dim=1)

    #Define a truncated mass distribution function
    def truncated_exponential_sample(self, pdmf, minmass, maxmass, size):
        m1 = maxmass**(pdmf+1.) - minmass**(pdmf+1.)
        m2 = minmass**(pdmf+1.)
        powmass = (1.0/(pdmf+1.) )
        return np.power(m1*np.random.rand(size) + m2, powmass)

    def gen_sCMD(self, labels, ages, n_samples, pdmf, radius_dist, distance, max_mass = 0.95, min_mass = 0.4,**kwargs):
        r''' Generate a synthetic Color Magnitude Diagram

        Arguments:
            labels(array): labels (except mass) being used as conditions for the stellar evolution models to emulate isochrone
            ages(array): age of the models
            n_samples(int): number of stars on the CMD
            pdmf(float): Present date mass function
            radius_dist(function): Sample from the distribution of distance from center
            distance(float): distance modulus of the cluster

        kwarg arguments:
            bc_table(function): Bolometric Correction table used
            bc_format(list): Format for BC tables
            as_phot(function): Photometric uncertainty for the observation
            as_comp(function): Photometric complentness for the observation 
            as_format(list): Format for  photometric uncertainty
            bf(float): visual binary fraction
            red(float): Extinction
            max_mass(float): maximum mass allowed
            min_mass(float): minimum mass allowed
        
            #need to add support for red
        '''

        #replace modules if indicated
        if 'bc_table' in kwargs:
            self.bc_table = kwargs['bc_table']
        if 'bc_format' in kwargs:
            self.bc_format = kwargs['bc_format']
        if 'as_phot' in kwargs:
            self.as_phot = kwargs['as_phot']
        if 'as_comp' in kwargs:
            self.as_comp = kwargs['as_comp']
        if 'as_format' in kwargs:
            self.as_format = kwargs['as_format']

        #check all colors exist for both bc_tables and as_phot
        try:
            bc_color_idx = [self.bc_format['colors'].index(color) for color in self.as_format['colors']]
        except:
            raise ValueError("CMD color is not included in Bolometric Correction Tables")
            
        #currently only support truncated expontial distribution, will add more in the future
        #Need to find better way to determine the boundary of mass to avoid extrapolation problem
        mass = self.truncated_exponential_sample(pdmf, min_mass, max_mass, n_samples)

        #generate stars in theoretical models
        #for sCMD we only need input for BC tables and Luminosity
        theory_models = self.gen_theory_models(data_transform.norm_data_torch(torch.tensor(np.hstack((np.repeat(labels[np.newaxis],n_samples,axis=0), ages.reshape(n_samples,1), mass.reshape(n_samples,1))),dtype=torch.float32).to(self.device), self.SEM_format['label_scaler'], self.SEM_format['label_scale']), output_header=self.bc_format['label_header'] + ['Log_L'])
        #convert to magnitudes
        #no need to sample because BC tables uses MLP instead of CNF
        #convert back to magnitude
        bc_models = 4.75 - 2.5*theory_models[:,[-1]] - data_transform.inverse_norm_data_torch(self.bc_table(data_transform.norm_data_torch(theory_models[:,:-1],self.bc_format['label_scaler'],self.bc_format['label_scale'])), self.bc_format['data_scaler'], self.bc_format['data_scale'])[:,bc_color_idx]
        bf = kwargs.get('bf',0)
        if bf != 0:
            #sample stars and generate binaries for them
            n_binary = int(n_samples*bf)
            binary_mass = mass[:n_binary] * (0.5 + 0.5*np.random.rand(n_binary))
            binary_theory_models = self.gen_theory_models(data_transform.norm_data_torch(torch.tensor(np.hstack((np.repeat(labels[np.newaxis],n_binary,axis=0), ages[:n_binary].reshape(n_binary,1), binary_mass.reshape(n_binary,1))),dtype=torch.float32).to(self.device), self.SEM_format['label_scaler'], self.SEM_format['label_scale']), output_header=self.bc_format['label_header'] + ['Log_L'])
            binary_bc_models = 4.75 - 2.5*binary_theory_models[:,[-1]] -  data_transform.inverse_norm_data_torch(self.bc_table(data_transform.norm_data_torch(binary_theory_models[:,:-1],self.bc_format['label_scaler'],self.bc_format['label_scale'])), self.bc_format['data_scaler'], self.bc_format['data_scale'])[:,bc_color_idx]
            #Add magnitude together 
            bc_models[:n_binary, :].copy_(torch.log10(torch.clamp(torch.pow(10.0, bc_models[:n_binary, :]) + torch.pow(10.0, binary_bc_models), min=1e-10)))
        mag_idx = [self.as_format['label_header'].index('Input' + color) for color in self.as_format['colors']]
        input_mags = bc_models+ distance
        bc_models.copy_(data_transform.norm_data_torch(bc_models+ distance, [self.as_format['label_scaler'][i] for i in mag_idx], [self.as_format['label_scale'][i] for i in mag_idx]))
        #sample radius from radius distribution and rescale for AS test
        r_idx = self.as_format['label_header'].index('Input' + 'R')
        radius = data_transform.norm_data_torch(torch.tensor(radius_dist(n_samples),dtype=torch.float32).unsqueeze(1).to(self.device), [self.as_format['label_scaler'][r_idx]],[self.as_format['label_scale'][r_idx]])  

        #if completeness is defined, perform completeness test
        if self.as_comp is not None:
            comp_test_result = self.as_comp(torch.cat((bc_models, radius), dim=1))
            bc_models = bc_models[comp_test_result[:, 0] < comp_test_result[:, 1]]
            radius = radius[comp_test_result[:, 0] < comp_test_result[:, 1]]

        #Add photometric uncertainty
        for i in range(len(mag_idx)):
            bc_models[:,i] = self.as_phots[i](torch.cat((bc_models[:,[i]], radius), dim=1)).sample()[:,0]
        return data_transform.inverse_norm_data_torch(torch.cat((bc_models, radius), dim=1), self.as_format['data_scaler'], self.as_format['data_scale'])

class validate(emulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        #check if automatic optimization is needed
        auto_opt = kwargs.get('auto_opt',False)

        #define sample number
        self.nsample = kwargs.get('nsample',10000)

        #define file path
        file_path = kwargs.get('output_path',None)
        if file_path == None:
            raise Exception ('Need to define output_path.')


        if not auto_opt:
            batch_size = kwargs.get('batch_size', self.nsample)
        else:
            #load validation dataset
            kwargs['batch_size'] = 1
            kwargs['epoch'] = 1
            self.data_loader = TrainMod(**kwargs).create_data_loader()
            _, labels = next(iter(self.data_loader))
            #make sure batch_size * nsample fills up the GPU memory
            batch_size = performance().estimate_optimal_batch_size(self.device, self.flow, labels, safety_margin=0.07,**kwargs) // self.nsample

        #reload validation dataset
        kwargs['batch_size'] = batch_size

        self.data_loader = TrainMod(**kwargs).create_data_loader()

        header_names = (
            self.SEM_format['label_header'] +
            self.SEM_format['data_header'] +
            [h + '_output_median' for h in self.SEM_format['data_header']] +
            [h + '_output_std' for h in self.SEM_format['data_header']]
        )
        header_str = ','.join(header_names)

        #create file and save header first
        file_exists = os.path.exists(file_path)
        if file_exists:
            df = pd.read_csv(file_path)
            sidx = len(df)
        else:
            sidx = 0
        with open(file_path, 'a' if file_exists else 'w') as f:
            cidx = 0
            for i, (data, labels) in enumerate(self.data_loader):
                if cidx + batch_size > sidx:
                    emulated = self.gen_theory_models(labels, **kwargs).reshape(data.shape[0], self.nsample, data.shape[1])
                    
                    median_values = emulated.median(dim=1,keepdim=True).values

                    std_values = 1.4826*(torch.abs(emulated - median_values)).median(dim=1).values

                    retval = torch.cat([labels, data, median_values.squeeze().cpu(), std_values.cpu()], dim=1).numpy()

                    if i == 0 and not file_exists:
                        np.savetxt(f, retval, delimiter=',', header=header_str, comments='', fmt='%.8f')
                    else:
                        np.savetxt(f, retval, delimiter=',', fmt='%.8f')
                cidx += batch_size                          
