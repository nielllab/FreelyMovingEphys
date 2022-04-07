"""
FreelyMovingEphys/src/glm_rf.py
"""
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from kornia.geometry.transform import Affine
from torch.nn.modules.activation import Softplus
from torch.nn.modules.linear import Linear

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import src.utils.save as ioh5

class LinVisNetwork(nn.Module):
    def __init__(self,
                    in_features,
                    N_cells,
                    shift_in=3,
                    shift_hidden=20,
                    shift_out=3,
                    hidden_move=15,
                    reg_alph=None,
                    reg_alphm=None,
                    move_features=None,
                    LinMix=False,
                    train_shifter=False,
                    meanbias = None,
                    device='cuda'):
        super(LinVisNetwork, self).__init__()

        self.in_features = in_features
        self.N_cells = N_cells
        self.move_features = move_features
        self.LinMix = LinMix
        self.meanbias = meanbias
        # Cell_NN = {'{}'.format(celln):nn.Sequential(nn.Linear(self.in_features, self.hidden_features),nn.Softplus(),nn.Linear(self.hidden_features, 1)) for celln in range(N_cells)}
        # self.visNN = nn.ModuleDict(Cell_NN)
        self.Cell_NN = nn.Sequential(nn.Linear(self.in_features, self.N_cells,bias=True))
        self.activations = nn.ModuleDict({'SoftPlus':nn.Softplus(),
                                          'ReLU': nn.ReLU(),})
        torch.nn.init.uniform_(self.Cell_NN[0].weight, a=-1e-6, b=1e-6)
        # torch.nn.init.constant_(self.Cell_NN[2].weight, 1)
        # if self.meanbias is not None:
        #     torch.nn.init.constant_(self.Cell_NN[0].bias, meanbias)
        
        # Initialize Regularization parameters
        self.reg_alph = reg_alph
        if self.reg_alph != None:
            self.alpha = reg_alph*torch.ones(1).to(device)
        
        # Initialize Movement parameters
        self.reg_alphm = reg_alphm
        if self.move_features != None:
            if reg_alphm != None:
                self.alpha_m = reg_alphm*torch.ones(1).to(device)
            self.posNN = nn.ModuleDict({'Layer0': nn.Linear(move_features, N_cells)})
            torch.nn.init.uniform_(self.posNN['Layer0'].weight,a=-1e-6,b=1e-6)
            torch.nn.init.zeros_(self.posNN['Layer0'].bias)

        # option to train shifter network
        self.train_shifter = train_shifter
        self.shift_in = shift_in
        self.shift_hidden = shift_hidden
        self.shift_out = shift_out
        if train_shifter:
            self.shifter_nn = nn.Sequential(
                nn.Linear(self.shift_in,shift_hidden),
                nn.Softplus(),
                nn.Linear(shift_hidden, shift_out)
            )
        
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight,a=-1e-6,b=1e-6)
            # torch.nn.init.zeros_(m.weight)
            m.bias.data.fill_(1e-6)
    
    def forward(self, inputs, move_input=None, eye_input=None, celln=None):
        if self.train_shifter: 
            batchsize, timesize, x, y = inputs.shape
            dxy = self.shifter_nn(eye_input)
            shift = Affine(angle=torch.clamp(dxy[:,-1],min=-30,max=30),translation=torch.clamp(dxy[:,:2],min=-15,max=15))
            inputs = shift(inputs)
            inputs = inputs.reshape(batchsize,-1).contiguous()
        # fowrad pass of GLM 
        x, y = inputs.shape    
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        if celln is not None:
            output = []
            for celln in range(self.N_cells):
                output.append(self.Cell_NN['{}'.format(celln)](inputs))
            output = torch.stack(output).squeeze().T
        else:
            output = self.Cell_NN(inputs)
        # Add Vs. Multiplicative
        if move_input != None:
            if self.LinMix==True:
                output = output + self.posNN['Layer0'](move_input)
            else:
                move_out = self.posNN['Layer0'](move_input)
                # move_out = self.activations['SoftPlus'](move_out)
                # move_out = torch.exp(move_out)
                output = output*move_out
        ret = self.activations['ReLU'](output)
        return ret
    
    def loss(self,Yhat, Y): 
        if self.LinMix:
            loss_vec = torch.mean((Yhat-Y)**2,axis=0)
        else:
            loss_vec = torch.mean((Yhat-Y)**2,axis=0)
            # loss_vec = torch.mean(Yhat-Y*torch.log(Yhat),axis=0)  # Log-likelihood
        if self.move_features != None:
            if self.reg_alph != None:
                l1_reg0 = self.alpha*(torch.linalg.norm(self.Cell_NN[0].weight,axis=1,ord=1))
            else: 
                l1_reg0 = 0
                l1_reg1 = 0
            if self.reg_alphm != None:
                l1_regm = self.alpha_m*(torch.linalg.norm(self.weight[:,-self.move_features:],axis=1,ord=1))
            else: 
                l1_regm = 0
            loss_vec = loss_vec + l1_reg0 + l1_reg1 + l1_regm
        else:
            if self.reg_alph != None:
                l1_reg0 = torch.stack([torch.linalg.vector_norm(NN_params,ord=1) for name, NN_params in self.Cell_NN.named_parameters() if '0.weight' in name])
                loss_vec = loss_vec + self.alpha*(l1_reg0)
        
        return loss_vec

class FitGLM():
    def __init__(self, date, animal, recording_directory, model):
        self.date = date
        self.animal = animal
        self.recording_directory = recording_directory

        # which model to train
        if type(model) == int: # if an int, should be a value from -1 to 4
            model = ['only_pos','only_vis','add','mult','head-fixed','shifter'][model]
        self.movmodel = model

    def get_model(self, input_size, output_size, meanbias,
                  device, l, a, params, NepochVis=10000, best_shifter_Nepochs=2000,
                  Kfold=0, **kwargs):

        l1 = LinVisNetwork(input_size,output_size,
                            reg_alph=params['alphas'][a],reg_alphm=params['alphas_m'][a],move_features=params['move_features'],
                            train_shifter=params['train_shifter'], shift_hidden=50,
                            LinMix=params['LinMix'], device=device,).to(device)
        if (params['train_shifter']==False) & (params['MovModel']!=0) & (params['NoShifter']==False) & (params['SimRF']==False):
            state_dict = l1.state_dict()
            best_shift = 'GLM_{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_Kfold{:01d}.pth'.format('Pytorch_BestShift',int(params['model_dt']*1000), 1, 1, best_shifter_Nepochs, Kfold)
            checkpoint = torch.load(params['save_dir']/best_shift)
            for key in state_dict.keys():
                if 'posNN' not in key:
                    if 'weight' in key:
                        state_dict[key] = checkpoint['model_state_dict'][key].repeat(1,params['nt_glm_lag'])
                    else:
                        state_dict[key] = checkpoint['model_state_dict'][key]
            l1.load_state_dict(state_dict)
        elif (params['NoShifter']==True) :
            pass
        elif (params['SimRF']==True):
            SimRF_file = params['save_dir'].parent.parent.parent/'121521/SimRF/fm1/SimRF_withL1_dt050_T01_Model1_NB10000_Kfold00_best.h5'
            SimRF_data = ioh5.load(SimRF_file)
            l1.Cell_NN[0].weight.data = torch.from_numpy(SimRF_data['sta'].astype(np.float32).T).to(device)
            l1.Cell_NN[0].bias.data = torch.from_numpy(SimRF_data['bias_sim'].astype(np.float32)).to(device)
            # pass
        else:
            if meanbias is not None:
                state_dict = l1.state_dict()
                state_dict['Cell_NN.0.bias']=meanbias
                l1.load_state_dict(state_dict)
        if MovModel == 0: 
            optimizer = optim.Adam(params=[{'params': [l1.Cell_NN[0].weight],'lr':params['lr_w'][1]}, 
                                            {'params': [l1.Cell_NN[0].bias],'lr':params['lr_b'][1]},])
        elif MovModel == 1:
            if params['train_shifter']:
                optimizer = optim.Adam(params=[{'params': [l1.Cell_NN[0].weight],'lr':params['lr_w'][1],'weight_decay':params['lambdas'][l]}, 
                                            {'params': [l1.Cell_NN[0].bias],'lr':params['lr_b'][1]},
                                            {'params': list(l1.shifter_nn.parameters()),'lr': params['lr_shift'][1],'weight_decay':.0001}])
            else:
                optimizer = optim.Adam(params=[{'params': [l1.Cell_NN[0].weight],'lr':params['lr_w'][1],'weight_decay':params['lambdas'][l]}, 
                                            {'params': [l1.Cell_NN[0].bias],'lr':params['lr_b'][1]},])
        else:
            if params['NoL1']:
                model_type = 'Pytorch_Vis_NoL1'
            else:
                model_type = 'Pytorch_Vis_withL1'
            GLM_LinVis = ioh5.load(params['save_model_Vis']/'GLM_{}_dt{:03d}_T{:02d}_MovModel{:d}_NB{}_Kfold{:02d}_best.h5'.format(model_type, int(params['model_dt']*1000), params['nt_glm_lag'], 1, NepochVis, Kfold))
            state_dict = l1.state_dict()
            for key in state_dict.keys():
                if 'posNN' not in key:
                    state_dict[key] = torch.from_numpy(GLM_LinVis[key].astype(np.float32))
            l1.load_state_dict(state_dict)
            optimizer = optim.Adam(params=[{'params': [param for name, param in l1.posNN.named_parameters() if 'weight' in name],'lr':params['lr_m'][1]},
                                        {'params': [param for name, param in l1.posNN.named_parameters() if 'bias' in name],'lr':params['lr_b'][1]},])
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(params['Nepochs']/5))
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9999)
        # scheduler = None
        return l1, optimizer, scheduler

    def train_model(self, xtr, xte, xtrm, xtem, shift_in_tr, shift_in_te,
                    ytr, yte, Nepochs, l1, optimizer,
                    scheduler=None, pbar=None, track_all=False):

        vloss_trace = np.zeros((Nepochs, ytr.shape[-1]), dtype=np.float32)
        tloss_trace = np.zeros((Nepochs, ytr.shape[-1]), dtype=np.float32)
        Epoch_GLM = {}
        if track_all:
            for name, p in l1.named_parameters():
                Epoch_GLM[name] = np.zeros((Nepochs,) + p.shape, dtype=np.float32)

        if pbar is None:
            pbar = pbar2 = tqdm(np.arange(Nepochs))
        else:
            pbar2 = np.arange(Nepochs)

        for batchn in pbar2:
            out = l1(xtr, xtrm, shift_in_tr)
            loss = l1.loss(out, ytr)
            pred = l1(xte, xtem, shift_in_te)
            val_loss = l1.loss(pred, yte)
            vloss_trace[batchn] = val_loss.clone().cpu().detach().numpy()
            tloss_trace[batchn] = loss.clone().cpu().detach().numpy()
            pbar.set_description('Loss: {:.03f}'.format(np.nanmean(val_loss.clone().cpu().detach().numpy())))
            pbar.refresh()
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if track_all:
                for name, p in l1.named_parameters():
                    Epoch_GLM[name][batchn] = p.clone().cpu().detach().numpy()

        return vloss_trace, tloss_trace, l1, optimizer, scheduler, Epoch_GLM

    def load_model_inputs(self, data, params, train_idx, test_idx, move_medwin=7):
        if params['free_move']:
            move_train = np.hstack((data['train_th'][:, np.newaxis], data['train_phi'][:, np.newaxis],data['train_pitch'][:, np.newaxis],data['train_roll'][:, np.newaxis]))
            move_test = np.hstack((data['test_th'][:, np.newaxis], data['test_phi'][:, np.newaxis],data['test_pitch'][:, np.newaxis],data['test_roll'][:, np.newaxis]))
            model_move = np.hstack((data['model_th'][:, np.newaxis], data['model_phi'][:, np.newaxis],data['model_pitch'][:, np.newaxis],data['model_roll'][:, np.newaxis]))
        else:
            move_train = np.hstack((data['train_th'][:, np.newaxis], data['train_phi'][:, np.newaxis], data['train_pitch'][:, np.newaxis], np.zeros(data['train_phi'].shape)[:, np.newaxis]))
            move_test = np.hstack((data['test_th'][:, np.newaxis], data['test_phi'][:, np.newaxis], data['test_pitch'][:, np.newaxis], np.zeros(data['test_phi'].shape)[:, np.newaxis]))
            model_move = np.hstack((data['model_th'][:, np.newaxis], data['model_phi'][:, np.newaxis], data['model_pitch'][:, np.newaxis], np.zeros(data['model_phi'].shape)[:, np.newaxis]))

        ##### Save dimensions #####    
        params['nks'] = np.shape(data['train_vid'])[1:]
        params['nk'] = params['nks'][0]*params['nks'][1]*params['nt_glm_lag']
        # Reshape data (video) into (T*n)xN array
        if params['train_shifter']:
            rolled_vid = np.hstack([np.roll(data['model_vid_sm'], nframes, axis=0) for nframes in params['lag_list']])
            move_quantiles = np.quantile(model_move,params['quantiles'],axis=0)
            train_range = np.all(((move_train>move_quantiles[0]) & (move_train<move_quantiles[1])),axis=1)
            test_range = np.all(((move_test>move_quantiles[0]) & (move_test<move_quantiles[1])),axis=1)
            x_train = rolled_vid[train_idx].reshape((len(train_idx), params['nt_glm_lag'])+params['nks']).astype(np.float32)[train_range]
            x_test = rolled_vid[test_idx].reshape((len(test_idx), params['nt_glm_lag'])+params['nks']).astype(np.float32)[test_range]
            move_train = move_train[train_range]
            move_test = move_test[test_range]
            shift_in_tr = torch.from_numpy(move_train[:, (0, 1, 3)].astype(np.float32)).to(device)
            shift_in_te = torch.from_numpy(move_test[:, (0, 1, 3)].astype(np.float32)).to(device)
            ytr = torch.from_numpy(data['train_nsp'][train_range].astype(np.float32)).to(device)
            yte = torch.from_numpy(data['test_nsp'][test_range].astype(np.float32)).to(device)
            data['train_nsp']=data['train_nsp'][train_range]
            data['test_nsp']=data['test_nsp'][test_range]
        elif params['NoShifter']:
            rolled_vid = np.hstack([np.roll(data['model_vid_sm'], nframes, axis=0) for nframes in params['lag_list']])
            x_train = rolled_vid[train_idx].reshape(len(train_idx), -1).astype(np.float32)
            x_test = rolled_vid[test_idx].reshape(len(test_idx), -1).astype(np.float32)
            shift_in_tr = None
            shift_in_te = None
            ytr = torch.from_numpy(data['train_nsp'].astype(np.float32)).to(device)
            yte = torch.from_numpy(data['test_nsp'].astype(np.float32)).to(device)
        else:
            model_vid_sm_shift = ioh5.load(params['save_dir']/'ModelWC_shifted_dt{:03d}_MovModel{:d}.h5'.format(int(params['model_dt']*1000), 1))['model_vid_sm_shift']  # [:,5:-5,5:-5]
            params['nks'] = np.shape(model_vid_sm_shift)[1:]
            params['nk'] = params['nks'][0]*params['nks'][1]*params['nt_glm_lag']
            rolled_vid = np.hstack([np.roll(model_vid_sm_shift, nframes, axis=0) for nframes in params['lag_list']])  
            x_train = rolled_vid[train_idx].reshape(len(train_idx), -1).astype(np.float32)
            x_test = rolled_vid[test_idx].reshape(len(test_idx), -1).astype(np.float32)
            shift_in_tr = None
            shift_in_te = None
            ytr = torch.from_numpy(data['train_nsp'].astype(np.float32)).to(device)
            yte = torch.from_numpy(data['test_nsp'].astype(np.float32)).to(device)


        if params['MovModel'] == 0:
            model_type = 'Pytorch_Mot'
        elif params['MovModel'] == 1:
            model_type = 'Pytorch_Vis'
        elif params['MovModel'] == 2:
            model_type = 'Pytorch_Add'
        elif params['MovModel'] == 3:
            model_type = 'Pytorch_Mul'

        if params['train_shifter']:
            params['save_model_shift'] = params['save_dir'] / 'models/Shifter'
            params['save_model_shift'].mkdir(parents=True, exist_ok=True)
            params['NoL1'] = True
            params['do_norm']=True
            model_type = model_type + 'Shifter'
        elif params['NoShifter']:
            model_type = model_type + 'NoShifter'

        if params['NoL1']:
            model_type = model_type + '_NoL1'
        else:
            model_type = model_type + '_withL1'
        
        if params['SimRF']:
            SimRF_file = params['save_dir'].parent.parent.parent/'021522/SimRF/fm1/SimRF_withL1_dt050_T01_Model1_NB10000_Kfold00_best.h5'
            SimRF_data = ioh5.load(SimRF_file)
            model_type = model_type + '_SimRF'
            ytr = torch.from_numpy(SimRF_data['ytr'].astype(np.float32)).to(device)
            yte = torch.from_numpy(SimRF_data['yte'].astype(np.float32)).to(device)
            params['save_model'] = params['save_model'] / 'SimRF'
            params['save_model'].mkdir(parents=True, exist_ok=True)
            meanbias = torch.from_numpy(SimRF_data['bias_sim'].astype(np.float32)).to(device)
        else:
            meanbias = torch.mean(torch.tensor(data['model_nsp'], dtype=torch.float32), axis=0)
        input_size = params['nk']
        output_size = ytr.shape[1]
        params['lr_shift'] = [1e-3,1e-2]
        params['Ncells'] = ytr.shape[-1]
        # Reshape data (video) into (T*n)xN array
        if params['MovModel'] == 0:
            mx_train = move_train.copy()
            mx_test = move_test.copy()
            xtr = torch.from_numpy(mx_train.astype(np.float32)).to(device)
            xte = torch.from_numpy(mx_test.astype(np.float32)).to(device)
            xtrm = None
            xtem = None
            params['nk'] = xtr.shape[-1]
            params['move_features'] = None 
            params['lambdas'] = np.hstack((0, np.logspace(-2, 3, 20)))
            params['alphas'] = np.array([None])
            params['lambdas_m'] = np.hstack((0, np.logspace(-2, 3, 20)))
            params['alphas_m'] = np.array([None]) 
            params['nlam'] = len(params['lambdas'])
            params['nalph'] = len(params['alphas'])
            params['lr_w'] = [1e-6, 1e-3]
            params['lr_b'] = [1e-6, 1e-3]
            input_size = xtr.shape[-1]
        elif params['MovModel'] == 1:
            xtr = torch.from_numpy(x_train).to(device)
            xte = torch.from_numpy(x_test).to(device)
            xtrm = None
            xtem = None
            params['move_features'] = None
            sta_init = torch.zeros((output_size, xtr.shape[-1]))
            params['alphas'] = np.array([.0001 if params['NoL1']==False else None])
            params['lambdas'] = np.hstack((0, np.logspace(-2, 3, 20)))
            params['nlam'] = len(params['lambdas'])
            params['nalph'] = len(params['alphas'])
            params['lambdas_m'] = np.array(params['nlam']*[None])
            params['alphas_m'] = np.array(params['nalph']*[None])
            params['lr_w'] = [1e-5, 1e-3]
            params['lr_b'] = [1e-5, 5e-3]
        else:
            xtr = torch.from_numpy(x_train.astype(np.float32)).to(device)
            xte = torch.from_numpy(x_test.astype(np.float32)).to(device)
            xtrm = torch.from_numpy(move_train.astype(np.float32)).to(device)
            xtem = torch.from_numpy(move_test.astype(np.float32)).to(device)
            params['move_features'] = xtrm.shape[-1]
            sta_init = torch.zeros((output_size, xtr.shape[-1]))
            params['alphas'] = np.array([None])
            params['lambdas'] =  np.hstack((0, np.logspace(-2, 3, 20)))
            params['nalph'] = len(params['alphas'])
            params['alphas_m'] = np.array(params['nalph']*[None])
            params['lambdas_m'] = np.array([0]) # np.hstack((0, np.logspace(-5, 6, 40)))
            params['nlam'] = len(params['lambdas_m'])
            params['lr_w'] = [1e-5, 1e-3]
            params['lr_m'] = [1e-6, 1e-3]
            params['lr_b'] = [1e-6, 1e-3]

        return params, xtr, xtrm, xte, xtem, ytr, yte, shift_in_tr, shift_in_te, input_size, output_size, model_type, meanbias, model_move

    def load_params(self):

    def train_shifter(self):
    
    def train_position