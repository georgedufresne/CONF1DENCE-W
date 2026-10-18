import torch
import zuko
import CONF1DENCE
import json
from torch.utils.data import Sampler, TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import h5py
from ..utils import read_tracks
from ..utils.helper import data_transform, performance
import time
import psutil
import math
import random

class TrainMod:
    def __init__(self, **kwargs):
        self.SEM_type = kwargs.get('SEM_type','DSEP_iso')

        # determine path to atm files
        self.SEM_path = kwargs.get('SEM_path',None)
        if self.SEM_path is None:
            self.SEM_path = CONF1DENCE.__abspath__+'data/HDF5_data/DSEP_iso_default.h5'
        
        #read format file
        format_path = CONF1DENCE.__abspath__+'data/format/' + self.SEM_type + '_format.json'
        with open(format_path, 'r') as json_file:
            self.Data_format = json.load(json_file)
        
        #get index
        self.index = kwargs.get('index',{'mc_num':'all', 'mass':'all'})

        #define dimensions for input and output parameters
        n_features = sum(2 if scaler == 'mantissa_exponent' else 1 for scaler in self.Data_format['data_scaler'])
        n_context = sum(2 if scaler == 'mantissa_exponent' else 1 for scaler in self.Data_format['label_scaler'])

        # Use consistent device checking - allow manual override
        if 'device' in kwargs:
            device = kwargs['device']
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        else:
            self.device = performance.check_cuda()

        #determine batch size
        self.batch_size = kwargs.get('batch_size', 4096)

        #determine the number of epochs
        self.n_epoch = kwargs.get('n_epoch', 1000)

        #determine whether to shuffle
        self.shuffle = kwargs.get('shuffle',False)
        self.block_size = kwargs.get('block_size',10000000)

        #define number of training workers. Default is max - 1
        self.num_workers = kwargs.get('num_workers',psutil.cpu_count(logical=True) - 1)
        self.prefetch_factor = kwargs.get('prefetch_factor',4)

        #setup zuko
        self.flow = zuko.flows.NSF(features=n_features,  context=n_context, transforms=10, hidden_features=[256] * 10).to(self.device)
        self.lr = kwargs.get('lr',1e-3)
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.lr)

    def create_data_loader(self):
        dataset = read_tracks.readSEM(SEM_type = self.SEM_type, SEM_path = self.SEM_path, index = self.index)
        if not self.shuffle:
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, drop_last=True,prefetch_factor=self.prefetch_factor)
        elif dataset.__len__() > self.block_size:
            #use block shuffle sampler to avoid memory explosion
            sampler = BlockShuffleSampler(dataset, block_size=self.block_size)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, sampler=sampler, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, drop_last=True,prefetch_factor=self.prefetch_factor)
        else:
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, drop_last=True,prefetch_factor=self.prefetch_factor)
        return dataloader
    
    def train(self,**kwargs):
        load_model = kwargs.get('load_model',False)
        if load_model == True:
            model_path = kwargs.get('model_path',CONF1DENCE.__abspath__+'data/model/DSEP_iso_model')

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.flow.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']  
            loss = checkpoint['loss'] 
        
        if 'epoch' not in locals():
            epoch = 0

        while epoch < self.n_epoch:
            losses = []
            n_bad_batch = 0
            n_good_batch = 0

            for x, c in self.dataloader:

                x = x.to(self.device, non_blocking=True)
                c = c.to(self.device, non_blocking=True)

                loss = -self.flow(c).log_prob(x).mean()

                #skip the batch of loss becomes nan
                if torch.isnan(loss):
                    n_bad_batch += 1
                    del loss
                    # Only clear CUDA cache if using CUDA
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                else:
                    n_good_batch += 1
                    loss.backward()
            
                    torch.nn.utils.clip_grad_norm_(self.flow.parameters(),max_norm=1.0)
                    
                    self.optimizer.step()
            
                    losses.append(loss.detach())
                
                self.optimizer.zero_grad()

            losses = torch.stack(losses)

            print(f'({epoch})', losses.mean().item(), '±', losses.std().item())

            #raise warning the ratio of bad batchs is too hight
            if n_good_batch == 0:
                raise Exception('All batches have ill-defined gradients. Training cannot continue.')
            if n_bad_batch/n_good_batch >0.05:
                print('There are too many batches with ill-defined gradient.')

            #save every 100 epoch
            if epoch % 100 == 0:
                torch.save({
                    'model_state_dict': self.flow.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,  # Optional: save the current epoch number
                    'loss': losses.mean(),    # Optional: save the current loss
                    }, model_path)
            epoch += 1

class TrainAS:
    def __init__(self, **kwargs):
        #read format file
        format_path = kwargs.get('AS_format_path',CONF1DENCE.__abspath__+'data/format/' + '47Tuc_AS_format.json')
        with open(format_path, 'r') as json_file:
            self.as_format = json.load(json_file)
        
        # Use consistent device checking - allow manual override
        if 'device' in kwargs:
            device = kwargs['device']
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        else:
            self.device = performance.check_cuda()


    #photometric uncertainty
    #We train filters seperatly due to the limited number of tests being performed
    def AS_phot(self, verbose=True, **kwargs):
        # determine path to training data
        traininng_data_path = kwargs.get('AS_phot_training',CONF1DENCE.__abspath__+'data/AS_test/' + '47Tuc_AS_phot.h5')

        #decide if to store model
        model_paths = kwargs.get('model_paths',None)
        if model_paths != None:
            if len(model_paths) != len(self.as_format['colors']):
                raise ValueError('Number models does not match up number of colors in AS test')
            

        model_filters = []
        
        rad_idx = self.as_format['label_header'].index('InputR')
        for i, color in enumerate(self.as_format['colors']):
            color_idx = self.as_format['label_header'].index('Input' + color)
            
            #Define NN
            n_features = 2
            n_context = 2
            model = kwargs.get('model',zuko.flows.NSF(features=n_features,  context=n_context, transforms=5, hidden_features=[64] * 5).to(self.device))

            #define other variables
            batch_size = kwargs.get('batch_size',512)
            n_epoch = kwargs.get('n_epoch',1000)
            lr = kwargs.get('lr',1e-3)
            optimizer = kwargs.get('optimizer',optim.Adam(model.parameters(), lr=lr))
            scheduler = kwargs.get('scheduler',lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.8,patience=20))

            #load data
            with h5py.File(traininng_data_path, 'r') as f:
                label = torch.tensor(data_transform.norm_data(f['label'][:,[color_idx, rad_idx]], [self.as_format["label_scaler"][i] for i in [color_idx, rad_idx]], [self.as_format["label_scale"][i] for i in [color_idx, rad_idx]]),dtype=torch.float32).to(self.device)
                data = torch.tensor(data_transform.norm_data(f['data'][:,[color_idx, rad_idx]], [self.as_format["data_scaler"][i] for i in [color_idx, rad_idx]], [self.as_format["data_scale"][i] for i in [color_idx, rad_idx]]),dtype=torch.float32).to(self.device)

            dataset = TensorDataset(data, label)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            for epoch in range(n_epoch):
                st = time.time()
                losses = []

                for x, c in dataloader:

                    loss = -model(c).log_prob(x).mean()
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    losses.append(loss.detach())
                losses = torch.stack(losses)
                et = time.time()
                if verbose==True:
                    print(f'({epoch})', losses.mean().item(), '±', losses.std().item(), f'using {et - st}s', f'with learning rate: {optimizer.param_groups[0]["lr"]}')
                scheduler.step(losses.mean().item())
            
            if model_paths == None:
                model_filters.append(model)
            else:
                torch.save(model, model_paths[i])

        if model_paths == None:
            return model_filters
        
    #photometric completeness
    def AS_comp(self, verbose=True, **kwargs):
        # determine path to training data
        traininng_data_path = kwargs.get('AS_comp_trainnig',CONF1DENCE.__abspath__+'data/AS_test/' + '47Tuc_AS_comp.h5')
        
        #Define NN
        #binary classification
        n_features = 1
        n_context = len(self.as_format["label_header"])
        model = kwargs.get('model',nn.Sequential(
            nn.Linear(n_context, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 2)
        ).to(self.device))

        #define other variables
        batch_size = kwargs.get('batch_size',512)
        n_epoch = kwargs.get('n_epoch',1000)
        lr = kwargs.get('lr',1e-3)
        optimizer = kwargs.get('optimizer',optim.Adam(model.parameters(), lr=lr))
        scheduler = kwargs.get('scheduler',lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.8,patience=20))
        loss_fct = nn.CrossEntropyLoss()

        #load data
        with h5py.File(traininng_data_path, 'r') as f:
            label = torch.tensor(data_transform.norm_data(f['label'], self.as_format["label_scaler"], self.as_format["label_scale"]),dtype=torch.float32).to(self.device)
            data = torch.tensor(f['data'],dtype=torch.float32).long().to(self.device)

        dataset = TensorDataset(data, label)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(n_epoch):
            st = time.time()
            losses = []

            for x, c in dataloader:

                loss = loss_fct(model(c), x)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                losses.append(loss.detach())
            losses = torch.stack(losses)
            et = time.time()
            if verbose==True:
                print(f'({epoch})', losses.mean().item(), '±', losses.std().item(), f'using {et - st}s', f'with learning rate: {optimizer.param_groups[0]["lr"]}')
            scheduler.step(losses.mean().item())
        
        #decide if to store model
        model_path = kwargs.get('model_path',None)
        if model_path == None:
            return model
        else:
            torch.save(model, model_path)

class TrainBC:
    def __init__(self, **kwargs):
        #read format file
        format_path = kwargs.get('BC_format_path',CONF1DENCE.__abspath__+'data/format/' + 'HSTACS_GC_format.json')
        with open(format_path, 'r') as json_file:
            self.bc_format = json.load(json_file)
        
        # Use consistent device checking - allow manual override
        if 'device' in kwargs:
            device = kwargs['device']
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        else:
            self.device = performance.check_cuda()

    #Bolometric Correction
    def BC_table(self, verbose=True, **kwargs):
        # determine path to training data
        traininng_data_path = kwargs.get('BC_table_trainnig',CONF1DENCE.__abspath__+'data/BC_table/' + 'HSTACS_GC_phx.h5')
        
        #Define NN
        #binary classification
        n_features = len(self.bc_format["data_header"])
        n_context = len(self.bc_format["label_header"])
        model = kwargs.get('model',nn.Sequential(
            nn.Linear(n_context, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, n_features)
        ).to(self.device))

        #define other variables
        batch_size = kwargs.get('batch_size',128)
        n_epoch = kwargs.get('n_epoch',1000)
        lr = kwargs.get('lr',1e-3)
        optimizer = kwargs.get('optimizer',optim.Adam(model.parameters(), lr=lr))
        scheduler = kwargs.get('scheduler',lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.8,patience=20))
        loss_fct = nn.MSELoss()

        #load data
        with h5py.File(traininng_data_path, 'r') as f:
            label = torch.tensor(data_transform.norm_data(f['label'], self.bc_format["label_scaler"], self.bc_format["label_scale"]),dtype=torch.float32).to(self.device)
            data = torch.tensor(data_transform.norm_data(f['data'], self.bc_format["data_scaler"], self.bc_format["data_scale"]),dtype=torch.float32).to(self.device)

        dataset = TensorDataset(data, label)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(n_epoch):
            st = time.time()
            losses = []

            for x, c in dataloader:

                loss = loss_fct(model(c), x)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                losses.append(loss.detach())
            losses = torch.stack(losses)
            et = time.time()
            if verbose==True:
                print(f'({epoch})', losses.mean().item(), '±', losses.std().item(), f'using {et - st}s', f'with learning rate: {optimizer.param_groups[0]["lr"]}')
            scheduler.step(losses.mean().item())
        
        #decide if to store model
        model_path = kwargs.get('model_path',None)
        if model_path == None:
            return model
        else:
            torch.save(model, model_path)

class TrainObs:
    def __init__(self, **kwargs):
        #read format file
        format_path = kwargs.get('Obs_format_path',CONF1DENCE.__abspath__+'data/format/' + '47Tuc_Obs_CMD_format.json')
        with open(format_path, 'r') as json_file:
            self.Obs_format = json.load(json_file)
        
        # Use consistent device checking - allow manual override
        if 'device' in kwargs:
            device = kwargs['device']
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        else:
            self.device = performance.check_cuda()


    #photometric uncertainty
    def Obs_CMD(self, verbose=True, **kwargs):
        # determine path to training data
        traininng_data_path = kwargs.get('Obs_CMD_trainnig',CONF1DENCE.__abspath__+'data/Obs_data/' + '47Tuc_Obs_CMD.h5')
        
        #Define NN
        n_features = len(self.Obs_format["data_header"])
        model = kwargs.get('model',zuko.flows.NSF(features=n_features,  transforms=10, hidden_features=[256] * 10).to(self.device))

        #define other variables
        batch_size = kwargs.get('batch_size',512)
        n_epoch = kwargs.get('n_epoch',1000)
        lr = kwargs.get('lr',1e-3)
        optimizer = kwargs.get('optimizer',optim.Adam(model.parameters(), lr=lr))
        scheduler = kwargs.get('scheduler',lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.8,patience=20))

        #load data
        with h5py.File(traininng_data_path, 'r') as f:
            data = torch.tensor(data_transform.norm_data(f['data'], self.Obs_format["data_scaler"], self.Obs_format["data_scale"]),dtype=torch.float32).to(self.device)

        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(n_epoch):
            st = time.time()
            losses = []

            for x in dataloader:

                loss = -model().log_prob(x[0]).mean()
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                losses.append(loss.detach())
            losses = torch.stack(losses)
            et = time.time()
            if verbose==True:
                print(f'({epoch})', losses.mean().item(), '±', losses.std().item(), f'using {et - st}s', f'with learning rate: {optimizer.param_groups[0]["lr"]}')
            scheduler.step(losses.mean().item())
        
        #decide if to store model
        model_path = kwargs.get('model_path',None)
        if model_path == None:
            return model
        else:
            torch.save(model, model_path)

class BlockShuffleSampler(Sampler):
    def __init__(self, data_source, block_size=10000000):
        """
        data_source: your Dataset
        block_size: number of indices to shuffle at once
        """
        self.data_source = data_source
        self.N = len(data_source)
        self.block_size = block_size

    def __iter__(self):
        # how many full blocks we’ll have
        num_blocks = math.ceil(self.N / self.block_size)
        blocks = list(range(num_blocks))
        random.shuffle(blocks)

        # for each block, shuffle its indices and yield them
        for b in blocks:
            start = b * self.block_size
            end = min(start + self.block_size, self.N)
            idxs = list(range(start, end))
            random.shuffle(idxs)
            for idx in idxs:
                yield idx

    def __len__(self):
        return self.N