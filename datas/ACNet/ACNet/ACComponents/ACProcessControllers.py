import json
import re
import os
import numpy as np

from TrainingFramework.Evaluator import *
from TrainingFramework.ProcessControllers import Controller, GreedyConfigController, ControllerStatusSaver, \
    EarlyStopController, Saver
from ACComponents.ACDataset.Dataset import *
from TrainingFramework.Initializer import *
from TrainingFramework.Scheduler import *

import torch as t
import torch.optim as optim
from functools import partial

from ACComponents.ACModels import *



class ACExperimentProcessController(Controller):
    def __init__(self, ExpOptions, Params):
        super(ACExperimentProcessController, self).__init__()

        self.ExpOptions = ExpOptions
        self.search = self.ExpOptions['Search']
        self.seedperopt = self.ExpOptions['SeedPerOpt']
        self.subsetsnum = self.ExpOptions['SubsetsNum']
        self.OnlyEval = self.ExpOptions['OnlyEval']
        # todo(zqzhang): updated in ACv8
        if 'Finetune' in self.ExpOptions.keys():
            self.Finetune = self.ExpOptions['Finetune']
        else:
            self.Finetune = False

        # process the params based on different searching methods, determined by the ExpOptions
        if self.search == 'greedy':
            self.BasicParamList, self.AdjustableParamList, self.SpecificParamList = Params


        if self.OnlyEval:
            # todo(zqzhang): updated in ACv8
            self.EvalParams = {'EvalModelPath': self.BasicParamList['EvalModelPath'],
                               'EvalDatasetPath': self.BasicParamList['EvalDatasetPath'],
                               'EvalLogAllPreds': self.BasicParamList['EvalLogAllPreds'],
                               'EvalOptPath': self.BasicParamList['EvalOptPath'],
                               'EvalBothSubsets': self.BasicParamList['EvalBothSubsets'],
                               'EvalSubsetNum': self.BasicParamList['EvalSubsetNum']
                          }

        self.ConfigControllersList = {
            'greedy': GreedyConfigController
        }

        # device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if torch.cuda.is_available() else 'cpu')
        # os.environ['CUDA_VISIBLE_DEVICES'] = self.BasicParamList['CUDA_VISIBLE_DEVICES']
        self.configcontroller = self.ConfigControllersList[self.search](self.BasicParamList, self.AdjustableParamList,
                                                                        self.SpecificParamList)

        self.controllerstatussaver = ControllerStatusSaver(self.configcontroller.opt.args,
                                                           'ExperimentProcessController')

        status = self.LoadStatusFromFile()
        if status:
            self.SetControllerStatus(status)
        else:
            self.InitControllerStatus()

    def ExperimentStart(self):
        if self.OnlyEval:
            # todo(zqzhang): updated in ACv8
            tmp_opt = self.configcontroller.GetOpts()
            tmp_opt.args.update(self.EvalParams)
            # print(opt.args)
            self.onlyevaler = ACOnlyEvaler(tmp_opt)
            self.onlyevaler.OnlyEval()
            return 0

        # Set the Config Controllers
        end_flag = False

        while not end_flag:
            opt = self.configcontroller.GetOpts()

            while self.i < self.seedperopt:
                self.CheckDirectories(opt, self.i)
                opt.set_args('TorchSeed', self.i+4)

                print("The parameters of the current exp are: ")
                print(opt.args)

                # The difference between normal ExpController and the ACExpController is that
                # the ACExpController will train a model by a new trainer for each subset, per opt, per seed.
                self.subsetsreults = []

                opt.set_args('OriginSaveDir', opt.args['SaveDir'])
                # todo(zqzhang): updated in ACv8
                opt.set_args("Finetune", self.Finetune)
                for self.k in range(self.subsetsnum):
                    opt.set_args('SaveDir', opt.args['OriginSaveDir'] + str(self.k) + '/')
                    if not os.path.exists(opt.args['SaveDir']):
                        os.mkdir(opt.args['SaveDir'])
                    print("Training on subset: ", self.k)

                    # todo(zqzhang): updated in ACv8
                    # if not self.Finetune:
                    trainer = ACTrainer(opt, self.k)
                    # else:
                    #     trainer = ACFinetuneTrainer(opt, self.k)
                    ckpt, subset_value = trainer.TrainOneOpt()
                    self.subsetsreults.append(subset_value)

                value = np.mean(self.subsetsreults)
                # trainer = Trainer(opt)
                # ckpt, value = trainer.TrainOneOpt()
                self.cur_opt_results.append(value)
                self.i += 1
                self.SaveStatusToFile()
                self.configcontroller.SaveStatusToFile()

            cur_opt_value = np.mean(self.cur_opt_results)     # the average result value of the current opt on self.seedperopt times running.
            self.opt_results.append(cur_opt_value)
            self.cur_opt_results = []                         # clear the buffer of current opt results.

            self.configcontroller.StoreResults(cur_opt_value)
            self.configcontroller.exp_count += 1
            end_flag = self.configcontroller.AdjustParams()
            self.i = 0

        print("Experiment Finished")
        print("The best averaged value of all opts is: ")
        if opt.args['ClassNum'] == 1:
            best_opt_result = min(self.opt_results)
            print(best_opt_result)
        else:
            best_opt_result = max(self.opt_results)
            print(best_opt_result)
        print("And the corresponding exp num is: ")
        print(self.opt_results.index(best_opt_result))

    def CheckDirectories(self, opt, i):
        opt.set_args('SaveDir', opt.args['ExpDir'] + str(i) + '/')
        if not os.path.exists(opt.args['SaveDir']):
            os.mkdir(opt.args['SaveDir'])

        opt.set_args('ModelDir', opt.args['SaveDir'] + 'model/')
        if not os.path.exists(opt.args['ModelDir']):
            os.mkdir(opt.args['ModelDir'])

    def LoadStatusFromFile(self):
        status = self.controllerstatussaver.LoadStatus()
        return status

    def SaveStatusToFile(self):
        self.GetControllerStatus()
        print(self.status)
        self.controllerstatussaver.SaveStatus(self.status)

    def InitControllerStatus(self):
        self.cur_opt_results = []
        self.opt_results = []
        self.i = 0

    def GetControllerStatus(self):
        self.status = {
            'cur_opt_results': self.cur_opt_results,
            'opt_results': self.opt_results,
            'cur_i': self.i,
            'cur_opt_torchseed_subset_reuslts': self.subsetsreults
        }

    def SetControllerStatus(self, status):
        self.cur_opt_results = status['cur_opt_results']
        self.opt_results = status['opt_results']
        self.i = status['cur_i']
        assert self.i == len(self.cur_opt_results)

class ACTrainer(object):
    def __init__(self, opt, subsetindex=None):
        super(ACTrainer, self).__init__()
        self.opt = opt

        self.train_fn = self.TrainOneEpoch

        t.manual_seed(self.opt.args['TorchSeed'])
        #statussaver = ControllerStatusSaver(self.opt.args, 'Trainer')
        self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')

        self.BuildDataset(subsetindex)


        self.net = self.BuildModel()
        self.BuildIniter()
        if self.initer:
            self.WeightInit()

        self.BuildOptimizer()
        self.StartEpoch = 0


        self.lr_sch = self.BuildScheduler()
        self.BuildCriterion()
        self.evaluator = self.BuildEvaluator()
        self.earlystopcontroller = EarlyStopController(self.opt)

        self.saver = Saver()
        self.controllerstatussaver = ControllerStatusSaver(self.opt.args, 'Trainer', restart=True)

    ########################################################
    def BuildModel(self):
        if self.opt.args['Model'] == 'MLP':
            net = ACPredMLP(self.opt).to(self.device)
        elif self.opt.args['Model'] == 'LSTM':
            net = ACPredLSTM(self.opt).to(self.device)
        elif self.opt.args['Model'] == 'GRU':
            net = ACPredGRU(self.opt).to(self.device)
        elif self.opt.args['Model'] == 'CMPNN':
            net = ACPredCMPNN(self.opt).to(self.device)
        elif self.opt.args['Model'] == 'PyGGCN':
            net = ACPredGCN(self.opt).to(self.device)
        elif self.opt.args['Model'] == 'PyGGIN':
            net = ACPredGIN(self.opt).to(self.device)
        elif self.opt.args['Model'] == 'PyGSGC':
            net = ACPredSGC(self.opt).to(self.device)
        elif self.opt.args['Model'] == 'Graphormer':
            net = ACPredGraphormer(self.opt).to(self.device)


        return net

    def BuildIniter(self):
        init_type = self.opt.args['WeightIniter']
        if init_type == 'Norm':
            self.initer = NormalInitializer(self.opt)

        elif init_type == 'XavierNorm':
            print("Using XavierNorm Initializer.")
            self.initer = XavierNormalInitializer()

        else:
            self.initer = None

    def BuildScheduler(self):
        if self.opt.args['Scheduler'] == 'EmptyLRScheduler':
            lr_sch = EmptyLRSchedular(self.optimizer, lr=10 ** -self.opt.args['lr'])

        elif self.opt.args['Scheduler'] == 'PolynomialDecayLR':
            # tot_updates = self.TrainsetLength * self.opt.args['MaxEpoch'] / self.opt.args['BatchSize']
            # warmup_updates = tot_updates / self.opt.args['WarmupRate']
            warmup_updates = self.opt.args['WarmupEpoch']
            tot_updates = self.opt.args['LRMaxEpoch']
            # warmup_updates = self.opt.args['WarmupUpdates']
            # tot_updates = self.opt.args['TotUpdeates']
            lr = 10 ** -self.opt.args['lr']
            end_lr = self.opt.args['EndLR']
            power = self.opt.args['Power']
            lr_sch = PolynomialDecayLR(self.optimizer, warmup_updates, tot_updates, lr, end_lr, power)

        elif self.opt.args['Scheduler'] == 'StepLR':
            step_size = self.opt.args['LRStep']
            gamma = self.opt.args['LRGamma']
            lr_sch = t.optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)

        return lr_sch

    def BuildEvaluator(self):
        if self.opt.args['Model'] == 'MLP':
            evaluator = MLPEvaluator(self.opt)
        elif self.opt.args['Model'] == 'LSTM':
            evaluator = MLPEvaluator(self.opt)
        elif self.opt.args['Model'] == 'GRU':
            evaluator = MLPEvaluator(self.opt)
        elif self.opt.args['Model'] == 'CMPNN':
            evaluator = MLPEvaluator(self.opt)
        elif self.opt.args['PyG'] :
            evaluator = PyGEvaluator(self.opt)
        elif self.opt.args['Model'] == 'Graphormer':
            evaluator = GeneralEvaluator(self.opt)

        return evaluator

    def BuildDataset(self, subsetindex):
        # todo(zqzhang): updated in ACv8
        if self.opt.args['Finetune']:
            moldatasetcreator = ACPTMDatasetCreator(self.opt, subsetindex)
        else:
            moldatasetcreator = ACMolDatasetCreator(self.opt,subsetindex)

        sets, self.weights = moldatasetcreator.CreateDatasets()

        if len(self.opt.args['SplitRate']) == 2:
            (Trainset, Validset, Testset) = sets
        elif len(self.opt.args['SplitRate']) == 1:
            (Trainset, Validset) = sets
        else:
            (Trainset) = sets

        self.batchsize = self.opt.args['BatchSize']

        if not self.opt.args['PyG']:
            self.TrainsetLength = len(Trainset)
            assert len(Trainset) >= self.batchsize

            # todo(zqzhang): Updated in TPv7 (numworkers and batchsize are modified)
            if self.opt.args['Model'] == 'Graphormer':
                self.trainloader = t.utils.data.DataLoader(Trainset, batch_size = self.batchsize, shuffle = True,
                                                           num_workers = 8,
                                                           drop_last = True, worker_init_fn = np.random.seed(8),
                                                           pin_memory = True,
                                                           collate_fn = partial(ACcollator,
                                                                                max_node = self.opt.args['max_node'],
                                                                                multi_hop_max_dist = self.opt.args[
                                                                                    'multi_hop_max_dist'],
                                                                                spatial_pos_max = self.opt.args[
                                                                                    'spatial_pos_max'])
                                                           )
                self.validloader = t.utils.data.DataLoader(Validset, batch_size = self.batchsize, shuffle = False,
                                                           num_workers = 8,
                                                           drop_last = False, worker_init_fn = np.random.seed(8),
                                                           pin_memory = False,
                                                           collate_fn = partial(ACcollator,
                                                                                max_node = self.opt.args['max_node'],
                                                                                multi_hop_max_dist = self.opt.args[
                                                                                    'multi_hop_max_dist'],
                                                                                spatial_pos_max = self.opt.args[
                                                                                    'spatial_pos_max'])
                                                           )
                self.testloader = t.utils.data.DataLoader(Testset, batch_size = self.batchsize, shuffle = False,
                                                          num_workers = 8,
                                                          drop_last = False, worker_init_fn = np.random.seed(8),
                                                          pin_memory = False,
                                                          collate_fn = partial(ACcollator,
                                                                               max_node = self.opt.args['max_node'],
                                                                               multi_hop_max_dist = self.opt.args[
                                                                                   'multi_hop_max_dist'],
                                                                               spatial_pos_max = self.opt.args[
                                                                                   'spatial_pos_max'])
                                                          )
            else:
            ####
                self.trainloader = t.utils.data.DataLoader(Trainset, batch_size = self.batchsize, shuffle = True, num_workers = 8, \
                                              drop_last = True, worker_init_fn = np.random.seed(8), pin_memory = True)
                self.validloader = t.utils.data.DataLoader(Validset, batch_size = self.batchsize, shuffle = False, num_workers = 8, \
                                              drop_last = False, worker_init_fn = np.random.seed(8), pin_memory = True)
                self.testloader = t.utils.data.DataLoader(Testset, batch_size = self.batchsize, shuffle = False, num_workers = 8, \
                                             drop_last = False, worker_init_fn = np.random.seed(8), pin_memory = True)
            ####
        else:
            import torch_geometric as tg
            self.TrainsetLength = len(Trainset)
            # print(Trainset)
            # print(len(Trainset))
            self.trainloader = tg.loader.DataLoader(Trainset, batch_size = self.batchsize, shuffle = True,
                                                    num_workers = 8, \
                                                    drop_last = True, worker_init_fn = np.random.seed(8),
                                                    pin_memory = True)
            self.validloader = tg.loader.DataLoader(Validset, batch_size = 8, shuffle = False, num_workers = 2, \
                                                    drop_last = False, worker_init_fn = np.random.seed(8),
                                                    pin_memory = True)
            if len(self.opt.args['SplitRate']) == 2:
                self.testloader = tg.loader.DataLoader(Testset, batch_size = 8, shuffle = False, num_workers = 2, \
                                                       drop_last = False, worker_init_fn = np.random.seed(8),
                                                       pin_memory = True)
            else:
                self.testloader = None


    def BuildOptimizer(self):
        self.optimizer = optim.Adam(self.net.parameters(), lr = 10 ** -self.opt.args['lr'],
                               weight_decay = 10 ** -self.opt.args['WeightDecay'])

    def BuildCriterion(self):
        if self.opt.args['ClassNum'] == 2:
            if self.opt.args['Weight']:
                self.criterion = [nn.CrossEntropyLoss(t.Tensor(weight), reduction = 'mean').\
                                      to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')) for weight in self.weights]
            else:
                self.criterion = [nn.CrossEntropyLoss().\
                                      to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')) for i in range(self.opt.args['TaskNum'])]
        elif self.opt.args['ClassNum'] == 1:
            self.criterion = [nn.MSELoss().\
                                  to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')) for i in range(self.opt.args['TaskNum'])]

    ########################################################

    def SaveModelCkpt(self, ckpt_idx):
        model = self.net
        optimizer = self.optimizer

        addr = self.opt.args['SaveDir'] + 'model/'
        self.CheckDirectory(addr)

        ckpt_name = addr + 'model' + str(ckpt_idx)
        ckpt = {'model': model,
                'optimizer': optimizer}
        t.save(ckpt, ckpt_name)
        print("Model Ckpt Saved!")

    def SaveResultCkpt(self, ckpt_idx, valid_result, test_result):
        results = {'cur_opt_valid': valid_result,
                   'cur_opt_test': test_result}
        addr = self.opt.args['SaveDir'] + 'results/'
        self.CheckDirectory(addr)

        result_ckpt_name = addr + 'result' + str(ckpt_idx)
        self.saver.SaveContext(result_ckpt_name, results)
        print('Result saved!')

    def SaveTrainerStatus(self, epoch):
        self.GetControllerStatus(epoch)
        self.controllerstatussaver.SaveStatus(self.status)
        print("Trainer status saved!")

    def GetControllerStatus(self, epoch):
        if self.opt.args['ClassNum'] == 2:
            best_valid = self.BestValidAUC
            test_of_best_valid = self.TestAUCofBestValid


        elif self.opt.args['ClassNum'] == 1:
            best_valid = self.BestValidRMSE
            test_of_best_valid = self.TestRMSEofBestValid


        self.status = {
            'cur_epoch': epoch,
            'best_valid': best_valid,
            'test_of_best_valid': test_of_best_valid
        }

    #def SetControllerStatus(self):

    def CheckDirectory(self, addr):
        if not os.path.exists(addr):
            os.mkdir(addr)

    ########################################################

    def TrainOneOpt(self):
        print("Saving Current opt...")
        self.saver.SaveContext(self.opt.args['SaveDir'] + 'config.json', self.opt.args)

        print("Start Training...")
        epoch = self.StartEpoch
        stop_flag = 0

        if self.opt.args['ClassNum'] == 2:
            self.BestValidAUC = 0
            self.TestAUCofBestValid = 0
        elif self.opt.args['ClassNum'] == 1:
            self.BestValidRMSE = 10e8
            self.TestRMSEofBestValid = 10e8

        while epoch < self.opt.args['MaxEpoch']:
            print('Epoch: ', epoch)

            if stop_flag:
                MaxResult = self.earlystopcontroller.MaxResult
                BestModel = self.earlystopcontroller.MaxResultModelIdx
                print("Early Stop")
                print("The Best Result is: ")
                print(MaxResult)
                print("and its corresponding model ckpt is: ")
                print(BestModel)
                # todo(zqzhang): updated in TPv7
                self.RemoveOtherCkpts(BestModel)
                break

            if self.opt.args['AC']:
                self.TrainOneEpoch_AC(self.net, self.trainloader, self.validloader, self.testloader, self.optimizer,
                                   self.criterion, self.evaluator)
                stop_flag = self.ValidOneTime(epoch, self.net)
            else:
                self.TrainOneEpoch(self.net, self.trainloader, self.validloader, self.testloader, self.optimizer, self.criterion, self.evaluator)
                stop_flag = self.ValidOneTime(epoch, self.net)

            self.SaveModelCkpt(epoch)
            self.SaveTrainerStatus(epoch)
            epoch += 1

        MaxResult = self.earlystopcontroller.MaxResult
        BestModel = self.earlystopcontroller.MaxResultModelIdx
        print("Stop Training.")
        print("The Best Result is: ")
        print(MaxResult)
        print("and its corresponding model ckpt is: ")
        print(BestModel)
        self.RemoveOtherCkpts(BestModel)


        return BestModel, MaxResult


    ################################################################################################################



    def TrainOneEpoch(self, model, trainloader, validloader, testloader, optimizer, criterion, evaluator):
        cum_loss = 0.0     # cum_loss is used to store the entire loss of a print period for printing the average loss.

        for ii, data in enumerate(trainloader):
            [Input, Label, Idx] = data
            Label = Label.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            Label = Label.squeeze(-1)       # [batch, task]
            Label = Label.t()               # [task, batch]

            output, _ = model(Input)
            loss = self.CalculateLoss(output, Label, criterion)
            loss.backward()

            # update the parameters
            if (ii+1) % self.opt.args['UpdateRate'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            cum_loss += loss.detach()

            # Print the loss
            if (ii+1) % self.opt.args['PrintRate'] == 0:
                print("Loss: ", cum_loss.item() / self.opt.args['PrintRate'])
                cum_loss = 0.0

            # Evaluation
            if (ii+1) % self.opt.args['ValidRate'] == 0:
                print("Running on valid set")
                result = evaluator.eval(validloader, model, [MAE(), RMSE()])
                print("Running on test set")
                testresult = evaluator.eval(testloader, model, [MAE(), RMSE()])

    def TrainOneEpoch_AC(self, model, trainloader, validloader, testloader, optimizer, criterion, evaluator):
        cum_loss = 0.0     # cum_loss is used to store the entire loss of a print period for printing the average loss.

        if self.opt.args['PyG']:
            # print("Here b")
            ii = 0
            # print(trainloader)
            for data in trainloader:
                # print(data)
                # print("Here C")
                Label = data.y
            # print(Input)
            # print(model)
            #     print(Label.size())    # [batch, task]
                Label = Label.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
                # Label = Label.squeeze(-1)       # [batch, task]
                Label = Label.t()               # [task, batch]
                # print(Label.size())

                output = model(data)
                # print(model.device)
                # print(output.device)
                loss = self.CalculateLoss(output, Label, criterion)
                loss.backward()
                # print("Here D")
            # update the parameters
                if (ii+1) % self.opt.args['UpdateRate'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                cum_loss += loss.detach()

            # Print the loss
                if (ii+1) % self.opt.args['PrintRate'] == 0:
                    print("Loss: ", cum_loss.item() / self.opt.args['PrintRate'])
                    cum_loss = 0.0

                # Evaluation
                if (ii+1) % self.opt.args['ValidRate'] == 0:
                    if self.opt.args['ClassNum'] == 1:
                        print('Running on Valid set')
                        result = self.evaluator.eval(self.validloader, model, [MAE(), RMSE()])
                        if self.testloader:
                            print('Running on Test set')
                            testresult = self.evaluator.eval(self.testloader, model, [MAE(), RMSE()])
                    else:
                        print("Running on Valid set")
                        result = self.evaluator.eval(self.validloader, model, [AUC(), ACC()])
                        if self.testloader:
                            print("running on Test set.")
                            testresult = self.evaluator.eval(self.testloader, model, [AUC(), ACC()])

                ii += 1
        else:
            for ii, data in enumerate(trainloader):
                if self.opt.args['Model'] == 'Graphormer':
                    (data1, data2) = data

                    x, attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, edge_input, Label, idx = \
                        data1.x, data1.attn_bias, data1.attn_edge_type, data1.spatial_pos, \
                        data1.in_degree, data1.out_degree, data1.edge_input, data1.y, data1.idx
                    Input1 = {
                        'x': x.to(self.device),
                        'attn_bias': attn_bias.to(self.device),
                        'attn_edge_type': attn_edge_type.to(self.device),
                        'spatial_pos': spatial_pos.to(self.device),
                        'in_degree': in_degree.to(self.device),
                        'out_degree': out_degree.to(self.device),
                        'edge_input': edge_input.to(self.device)
                    }
                    x, attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, edge_input, Label, idx = \
                        data2.x, data2.attn_bias, data2.attn_edge_type, data2.spatial_pos, \
                        data2.in_degree, data2.out_degree, data2.edge_input, data2.y, data2.idx
                    Input2 = {
                        'x': x.to(self.device),
                        'attn_bias': attn_bias.to(self.device),
                        'attn_edge_type': attn_edge_type.to(self.device),
                        'spatial_pos': spatial_pos.to(self.device),
                        'in_degree': in_degree.to(self.device),
                        'out_degree': out_degree.to(self.device),
                        'edge_input': edge_input.to(self.device)
                    }
                    Input = [Input1, Input2]
                    Label = Label.unsqueeze(-1)

                else:
                ###
                    [Input, Label, Idx] = data
                ###
                Label = Label.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
                Label = Label.squeeze(-1)       # [batch, task]
                Label = Label.t()               # [task, batch]

                output= model(Input)
            # print("output:")
            # print(output)
                loss = self.CalculateLoss(output, Label, criterion)
                loss.backward()

            # update the parameters
                if (ii+1) % self.opt.args['UpdateRate'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                cum_loss += loss.detach()

            # Print the loss
                if (ii+1) % self.opt.args['PrintRate'] == 0:
                    print("Loss: ", cum_loss.item() / self.opt.args['PrintRate'])
                    cum_loss = 0.0

            # Evaluation
                # todo(zqzhang): updated in TPv7
                if (ii + 1) % self.opt.args['ValidRate'] == 0:
                    if self.opt.args['ClassNum'] == 1:
                        print('Running on Valid set')
                        result = self.evaluator.eval(self.validloader, model, [MAE(), RMSE()])
                        if self.testloader:
                            print('Running on Test set')
                            testresult = self.evaluator.eval(self.testloader, model, [MAE(), RMSE()])
                    else:
                        print("Running on Valid set")
                        result = self.evaluator.eval(self.validloader, model, [AUC(), ACC()])
                        if self.testloader:
                            print("running on Test set.")
                            testresult = self.evaluator.eval(self.testloader, model, [AUC(), ACC()])

        self.lr_sch.step()


    def CalculateLoss(self, output, Label, criterion):

        loss = 0.0
        if self.opt.args['ClassNum'] != 1:
            for i in range(self.opt.args['TaskNum']):
                cur_task_output = output[:, i * self.opt.args['ClassNum']: (i + 1) * self.opt.args['ClassNum']]     # select the output of the current task i
                cur_task_label = Label[i]                               # [batch], the label of the current task i
                valid_index = (cur_task_label != -1)                    # Not all of the samples have labels of the current task i.

                valid_label = cur_task_label[valid_index]               # Only the samples that have labels of the current task i will participate in the loss calculation.
                if len(valid_label) == 0:
                    continue
                else:
                    valid_output = cur_task_output[valid_index]
                    # print(f"valid output.size:{valid_output.size()}")
                    loss += criterion[i](valid_output, valid_label)
        else:
            for i in range(self.opt.args['TaskNum']):
                cur_task_output = output[:, i * self.opt.args['ClassNum']: (i + 1) * self.opt.args['ClassNum']]
                cur_task_label = Label[i].unsqueeze(-1)
                loss += criterion[i](cur_task_output, cur_task_label)

        return loss

    ################################################################################################################
    def ValidOneTime(self,epoch,net):
        if self.opt.args['ClassNum'] == 1:
            print('Running on Valid set')
            result = self.evaluator.eval(self.validloader, net, [MAE(), RMSE()])
            print('Running on Test set')
            testresult = self.evaluator.eval(self.testloader, net, [MAE(), RMSE()])

            valid_result_rmse = result['RMSE']
            test_result_rmse = testresult['RMSE']
            if valid_result_rmse < self.BestValidRMSE:
                self.BestValidRMSE = valid_result_rmse
                self.TestRMSEofBestValid = test_result_rmse
            print('Best Valid: ')
            print(self.BestValidRMSE)
            print('Best Test: ')
            print(self.TestRMSEofBestValid)

            self.SaveResultCkpt(epoch, valid_result_rmse, test_result_rmse)

            stop_flag = self.earlystopcontroller.ShouldStop(result, epoch, testresult)
            # j += 1
            return stop_flag
        else:
            print("Running on Valid set")
            result = self.evaluator.eval(self.validloader, net, [AUC(), ACC()])
            print("running on Test set.")
            testresult = self.evaluator.eval(self.testloader, net, [AUC(), ACC()])

            valid_result_auc = result['AUC']
            test_result_auc = testresult['AUC']
            if valid_result_auc > self.BestValidAUC:
                self.BestValidAUC = valid_result_auc
                self.TestAUCofBestValid = test_result_auc
            print('Best Valid: ')
            print(self.BestValidAUC)
            print('Best Test: ')
            print(self.TestAUCofBestValid)

            self.SaveResultCkpt(epoch, valid_result_auc, test_result_auc)

            stop_flag = self.earlystopcontroller.ShouldStop(result, epoch, testresult)
            # j += 1
            return stop_flag

    def WeightInit(self):
        for param in self.net.parameters():
            self.initer.WeightInit(param)

    def RemoveOtherCkpts(self, bestmodel):
        print(f"Deleting other ckpt models.")
        model_dir = self.opt.args['SaveDir'] + 'model/'
        filenames = os.listdir(model_dir)
        for file in filenames:
            if file != ('model' + str(bestmodel)):
                os.remove(model_dir + file)

        print(f"Deleting other result files.")
        result_dir = self.opt.args['SaveDir'] + 'results/'
        filenames = os.listdir(result_dir)
        for file in filenames:
            if file != ('result' + str(bestmodel)):
                os.remove(result_dir + file)

        print(f"Deleting other TrainerStatus files.")
        status_dir = self.opt.args['SaveDir'] + 'TrainerStatus/'
        filenames = os.listdir(status_dir)
        filename = self.LastFileName(status_dir)
        for file in filenames:
            if file != filename:
                os.remove(status_dir + file)

    def LastFileName(self, Addr):
        dir_files = os.listdir(Addr)
        # os.listdir returns the file names in Addr, only the names, without the Addr path.
        if dir_files:
            dir_files = sorted(dir_files, key=lambda x: os.path.getctime(os.path.join(Addr, x)))
            last_file = dir_files[-1]
        else:
            last_file = ' '
        return last_file

class ACOnlyEvaler(Controller):
    def __init__(self, tmp_opt, evalloaders = None):
        super(ACOnlyEvaler, self).__init__()
        self.tmp_opt = tmp_opt
        # print(self.opt.args)
        self.EvalModelPath = self.tmp_opt.args['EvalModelPath']
        self.EvalDatasetPath = self.tmp_opt.args['EvalDatasetPath']
        self.LogAllPreds = self.tmp_opt.args['EvalLogAllPreds']
        self.EvalOptPath = self.tmp_opt.args['EvalOptPath']
        self.EvalBothSubsets = self.tmp_opt.args['EvalBothSubsets']
        self.EvalSubsetNum = self.tmp_opt.args['EvalSubsetNum']

        self.EvalParams = {'EvalModelPath': self.EvalModelPath,
                           'EvalDatasetPath': self.EvalDatasetPath,
                           'EvalLogAllPreds': self.LogAllPreds,
                           'EvalOptPath': self.EvalOptPath,
                           'EvalBothSubsets': self.EvalBothSubsets,
                           'EvalSubsetNum': self.EvalSubsetNum
                           }

        self.Saver = Saver()

        self.LoadEvalOpt()

        self.opt.args.update(self.EvalParams)

        if not evalloaders:
            self.BuildEvalDataset(self.EvalSubsetNum)
        else:
            self.evalloaders = evalloaders

        self.BuildEvalModel()
        self.evaluator = self.BuildEvaluator()

    def LoadEvalOpt(self):
        args = self.Saver.LoadContext(self.EvalOptPath)
        self.tmp_opt.args.update(args)
        self.opt = self.tmp_opt

    def BuildEvalDataset(self, subsetindex):
        if self.EvalDatasetPath:
            file_loader = JsonFileLoader(self.EvalDatasetPath)
            eval_dataset = file_loader.load()
            self.Evalset = ACDataset(eval_dataset)
        else:
            moldatasetcreator = ACMolDatasetCreator(self.opt,subsetindex)

            sets, self.weights = moldatasetcreator.CreateDatasets()

            if len(self.opt.args['SplitRate']) == 2:
                (Trainset, Validset, Testset) = sets
                if self.EvalBothSubsets:
                    self.Evalset1 = Validset
                    self.Evalset2 = Testset
                else:
                    self.Evalset = Testset
            elif len(self.opt.args['SplitRate']) == 1:
                (Trainset, Validset) = sets
                self.Evalset = Validset
            else:
                raise RuntimeError("No subset to be evaluated!")

        #############
        self.batchsize = self.opt.args['BatchSize']

        if not self.opt.args['PyG']:
            if self.opt.args['Model'] == 'Graphormer':
                if self.EvalBothSubsets:
                    self.evalloaders = [t.utils.data.DataLoader(self.Evalset1, batch_size = self.batchsize, shuffle = False,
                                                           num_workers = 8,
                                                           drop_last = False, worker_init_fn = np.random.seed(8),
                                                           pin_memory = False,
                                                           collate_fn = partial(ACcollator,
                                                                                max_node = self.opt.args['max_node'],
                                                                                multi_hop_max_dist = self.opt.args[
                                                                                    'multi_hop_max_dist'],
                                                                                spatial_pos_max = self.opt.args[
                                                                                    'spatial_pos_max'])
                                                           ),
                                        t.utils.data.DataLoader(self.Evalset2, batch_size = self.batchsize, shuffle = False,
                                                                num_workers = 8,
                                                                drop_last = False, worker_init_fn = np.random.seed(8),
                                                                pin_memory = False,
                                                                collate_fn = partial(ACcollator,
                                                                                     max_node = self.opt.args[
                                                                                         'max_node'],
                                                                                     multi_hop_max_dist = self.opt.args[
                                                                                         'multi_hop_max_dist'],
                                                                                     spatial_pos_max = self.opt.args[
                                                                                         'spatial_pos_max'])
                                                                )]
                else:
                    self.evalloader = t.utils.data.DataLoader(self.Evalset, batch_size = self.batchsize, shuffle = False,
                                                                num_workers = 8,
                                                                drop_last = False, worker_init_fn = np.random.seed(8),
                                                                pin_memory = False,
                                                                collate_fn = partial(ACcollator,
                                                                                     max_node = self.opt.args[
                                                                                         'max_node'],
                                                                                     multi_hop_max_dist = self.opt.args[
                                                                                         'multi_hop_max_dist'],
                                                                                     spatial_pos_max = self.opt.args[
                                                                                         'spatial_pos_max'])
                                                                )
            else:
                if self.EvalBothSubsets:
                    self.evalloaders = [
                        t.utils.data.DataLoader(self.Evalset1, batch_size = self.batchsize,
                                                                   shuffle = False, num_workers = 8,
                                                                   drop_last = False,
                                                                   worker_init_fn = np.random.seed(8),
                                                                   pin_memory = True),
                        t.utils.data.DataLoader(self.Evalset2, batch_size = self.batchsize, shuffle = False,
                                                              num_workers = 8, \
                                                              drop_last = False, worker_init_fn = np.random.seed(8),
                                                              pin_memory = True)]
                else:
                    self.evalloader = t.utils.data.DataLoader(self.Evalset, batch_size = self.batchsize,
                                                                   shuffle = False, num_workers = 8,
                                                                   drop_last = False,
                                                                   worker_init_fn = np.random.seed(8),
                                                                   pin_memory = True)
        else:
            import torch_geometric as tg
            if self.EvalBothSubsets:
                self.evalloaders = [tg.loader.DataLoader(self.Evalset1, batch_size = 8, shuffle = False, num_workers = 2, \
                                                    drop_last = False, worker_init_fn = np.random.seed(8),
                                                    pin_memory = True),
                                    tg.loader.DataLoader(self.Evalset2, batch_size = 8, shuffle = False, num_workers = 2, \
                                                         drop_last = False, worker_init_fn = np.random.seed(8),
                                                         pin_memory = True)
                                    ]
            else:
                self.evalloader = tg.loader.DataLoader(self.Evalset, batch_size = 8, shuffle = False, num_workers = 2, \
                                                    drop_last = False, worker_init_fn = np.random.seed(8),
                                                    pin_memory = True)
#########################

    def BuildEvalModel(self):
        import time
        ckpt = None
        while not ckpt:
            try:
                ckpt = t.load(self.EvalModelPath)
            except:
                time.sleep(1)
        self.net = ckpt['model']

    def BuildEvaluator(self):
        if self.opt.args['Model'] == 'MLP':
            evaluator = MLPEvaluator(self.opt)
        elif self.opt.args['Model'] == 'LSTM':
            evaluator = MLPEvaluator(self.opt)
        elif self.opt.args['Model'] == 'GRU':
            evaluator = MLPEvaluator(self.opt)
        elif self.opt.args['Model'] == 'CMPNN':
            evaluator = MLPEvaluator(self.opt)
        elif self.opt.args['PyG']:
            evaluator = PyGEvaluator(self.opt)
        elif self.opt.args['Model'] == 'Graphormer':
            evaluator = GeneralEvaluator(self.opt)

        return evaluator

    def OnlyEval(self):
        if self.opt.args['ClassNum'] == 1:
            if self.EvalBothSubsets:
                print('Running on Validset')
                evalloader = self.evalloaders[0]
                validresult = self.evaluator.eval(evalloader, self.net, [MAE(), RMSE()])
                print('Running on Testset')
                evalloader = self.evalloaders[1]
                testresult = self.evaluator.eval(evalloader, self.net, [MAE(), RMSE()])
            else:
                print('Running on Evalset')
                result = self.evaluator.eval(self.evalloader, self.net, [MAE(), RMSE()])

        else:
            if self.EvalBothSubsets:
                print('Running on Validset')
                evalloader = self.evalloaders[0]
                validresult = self.evaluator.eval(evalloader, self.net, [AUC(), ACC()])
                print('Running on Testset')
                evalloader = self.evalloaders[1]
                testresult = self.evaluator.eval(evalloader, self.net, [AUC(), ACC()])
            else:
                print("Running on Evalset")
                result = self.evaluator.eval(self.evalloader, self.net, [AUC(), ACC()])

        if self.LogAllPreds:
            print("All samples in Evalset and their predictions:")

            Answers = []
            for i in range(self.opt.args['TaskNum']):
                Answers.append([])

            AC_pos_cnt = 0
            AC_pos_correct_cnt = 0
            for i in range(self.opt.args['TaskNum']):
                for j in range(len(self.Evalset.dataset)):
                    item = self.Evalset.dataset[j]
                    SMILES1 = item['SMILES1']
                    SMILES2 = item['SMILES2']
                    Label = item['Value'][i]
                    # print(self.evaluator.AllLabel)
                    # print(self.evaluator.opt)
                    Label_e = self.evaluator.AllLabel[i][j]

                    # print(Label)
                    # print(Label_e)

                    assert Label == str(Label_e)
                    pred = self.evaluator.AllPred[i][j]

                    if Label == '1':
                        AC_pos_cnt += 1
                        if pred[1]>pred[0]:
                            AC_pos_correct_cnt += 1
                    Answers[i].append({'SMILES1': SMILES1, 'SMILES2': SMILES2, 'Value': Label, 'Pred': pred})

            print(Answers)

            print(f"Total Positive Numbers: {AC_pos_cnt}.")
            print(f"Total correct positive numbers: {AC_pos_correct_cnt}.")
            print(f"precision: {AC_pos_correct_cnt / AC_pos_cnt}.")

            Addr = self.opt.args['SaveDir']+'AllAnswers.json'
            with open(Addr,'w') as f:
                json.dump(Answers, f)

        if self.EvalBothSubsets:
            return (validresult, testresult)
        else:
            return result
