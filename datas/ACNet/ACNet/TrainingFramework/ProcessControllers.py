import json
import os



class Saver(object):
    # Module that package the saving functions
    def __init__(self):
        super(Saver, self).__init__()
        #self.ckpt_state = {}

    def SaveContext(self, context_add, context_obj):
        # if something can be summarized as a dict {}
        # then using SaveContext function to save it into a json file.
        with open(context_add, 'w') as f:
            json.dump(context_obj, f)

    def LoadContext(self, context_add):
        # Using LoadContext to load a json file to a dict {}.
        with open(context_add, 'r') as f:
            obj = json.load(f)
        return obj

class Configs(object):
    def __init__(self, ParamList):
        # initiale a Config object with given paramlist
        super(Configs, self).__init__()
        self.args = {}
        for param in ParamList.keys():
            self.set_args(param, ParamList.get(param))

    def set_args(self, argname, value):
        if argname in self.args:
            print("Arg", argname, "is updated.")
            self.args[argname] = value
        else:
            print('Arg', argname, 'is added.')
            self.args.update({argname: value})

##########################################################################################################

class Controller(object):
    # Controllers are modules to control the entire experiment progress
    # A controller will maintain some values and params and make decisions based on the given results.
    # The values and params maintained by the controller are called running status.
    # Controllers should be able to save and load the running status to/from a json file so that the experimental progress
    # can be continued after accidental terminating.
    def __init__(self):
        super(Controller, self).__init__()

    def BuildControllerStatus(self):
        raise NotImplementedError(
            "Build Status function not implemented.")

    def GetControllerStatus(self):
        raise NotImplementedError(
            "Get Status function not implemented.")

    def SetControllerStatus(self, status):
        raise NotImplementedError(
            "Set Status function not implemented.")

    def AddControllerStatus(self, name, value):
        raise NotImplementedError(
            "Add Status function not implemented.")

class ControllerStatusSaver(object):
    # Package functions for saving and loading status of a controller into/from a file.
    # In a ControllerStatusSaver, it maintains three global variable:
    # self.args: args
    # self.saver: Saver() object for file saving and loading.
    # self.Addr: The Addr to save the status of the controller.

    def __init__(self, args, ControllerType, Addr=None, restart=False):
        super(ControllerStatusSaver, self).__init__()
        self.saver = Saver()
        self.args = args

        if ControllerType == 'ExperimentProcessController':
            self.Addr = self.args['TrialPath'] + 'ExperimentProcessControllerStatus/'
        elif ControllerType == 'ConfigController':
            self.Addr = self.args['TrialPath'] + 'ConfigControllerStatus/'
        elif ControllerType == 'EarlyStopController':
            self.Addr = self.args['SaveDir'] + 'EarlyStopControllerStatus/'
        elif ControllerType == 'Trainer':
            self.Addr = self.args['SaveDir'] + 'TrainerStatus/'
        elif ControllerType == 'CkptController':
            self.Addr = self.args['TrialPath'] + 'ConfigControllerStatus/'
        else:
            if Addr:
                self.Addr = Addr
            else:
                raise KeyError(
                        'Wrong ControllerType given.'
                )
        self.CheckAddr()

        if restart:
            self.DeleteFilesInDir(self.Addr)

    def DeleteFilesInDir(self, addr):
        del_list = os.listdir(addr)
        for f in del_list:
            file_addr = addr + f
            os.remove(file_addr)

    def CheckAddr(self):
        if not os.path.exists(self.Addr):
            os.mkdir(self.Addr)

    def SaveStatus(self, status, restart=False):

        next_cnt = self.CountFileNames(self.Addr)
        if next_cnt != 0:
            assert self.LastFileName(self.Addr) == str(next_cnt-1)
            file_name = self.Addr + str(next_cnt)
        else:
            file_name = self.Addr + '0'
        self.saver.SaveContext(file_name, status)


    def LoadStatus(self, status_idx=None):
        # if the index is not given, then find the last file from the folder. the last file is the file to be loaded.
        # otherwise, the file of the given index is to be loaded.
        if not status_idx:
            file_name = self.Addr + self.LastFileName(self.Addr)
        else:
            file_name = self.Addr + str(status_idx)

        # if no file is to be loaded, then return None.
        # (e.g. empty in the folder or the given index not exists)
        if os.path.exists(file_name):
            return self.saver.LoadContext(file_name)
        else:
            return None

    def LastFileName(self, Addr):
        dir_files = os.listdir(Addr)
        # os.listdir returns the file names in Addr, only the names, without the Addr path.
        if dir_files:
            dir_files = sorted(dir_files, key=lambda x: os.path.getctime(os.path.join(Addr, x)))
            last_file = dir_files[-1]
        else:
            last_file = ' '
        return last_file

    def CountFileNames(self, Addr):
        dir_files = os.listdir(Addr)
        return len(dir_files)

def TestCodesForControllerStatusSaver():
    args = {'TrialPath': './TestExps/test/',
            'SaveDir': './TestExps/test/expi/'}
    controllerstatussaver = ControllerStatusSaver(args, 'ExperimentProcessController')
    status = controllerstatussaver.LoadStatus()
    print(status)
    status = {'1':1, '2':2}
    controllerstatussaver.SaveStatus(status)
    status = {'1':2, '2':3}
    controllerstatussaver.SaveStatus(status)
    status = {'1': 20, '2': 30}
    controllerstatussaver.SaveStatus(status)
    status = controllerstatussaver.LoadStatus()
    print(status)
    status = controllerstatussaver.LoadStatus(1)
    print(status)
    status = controllerstatussaver.LoadStatus(5)
    print(status)

##########################################################################################################

class EarlyStopController(Controller):
    # A module used to control the early stop part of the experimental progress.
    # It maintains the result of each epoch, max results, count of worse results
    # and to make decision whether the training progress should be early stopped.
    def __init__(self, opt):
        super(EarlyStopController, self).__init__()
        self.opt = opt
        # params coming from the opt are constant during the training progress of THIS opt
        self.MetricName = opt.args['MainMetric']
        self.LowerThanMaxLimit = opt.args['LowerThanMaxLimit']
        self.DecreasingLimit = opt.args['DecreasingLimit']
        # Other params are the running status of the EarlyStopController that should be saved and loaded by files.
        # initial MaxResult
        if self.opt.args['ClassNum'] == 1:
            self.MaxResult = 9e8
        else:
            self.MaxResult = 0
        # todo(zqzhang): updated in TPv7
        self.MaxResultModelIdx = 0
        self.LastResult = 0
        self.LowerThanMaxNum = 0
        self.DecreasingNum = 0
        self.TestResult = []

    def ShouldStop(self, score, ckpt_idx, testscore=None):
        # Make decision whether the training progress should be stopped.
        # When the current result is better than the MaxResult, then update thre MaxResult.
        # When the current result is worse that the MaxResult, then start to count.
        # When the num of epochs that the result is worse than the MaxResult exceed the LowerThanMaxLimit threshold, then stop
        # And when the result is persistently getting worse for over DecreasingLimit epochs, then stop.

        # score is the current Validation Result
        # ckpt_idx is the ckpt index
        # testscore is the result of the current model on the test set.

        MainScore = score[self.MetricName]
        if testscore:
            MainTestScore = testscore[self.MetricName]
        else:
            MainTestScore = None
        self.TestResult.append(MainTestScore)

        if self.opt.args['ClassNum'] != 1:
            # Classification, the larger the better
            if MainScore > self.MaxResult:
                self.MaxResult = MainScore
                self.MaxResultModelIdx = ckpt_idx
                self.LowerThanMaxNum = 0
                self.DecreasingNum = 0
                # all counts reset to 0.
            else:
                # decreasing, start to count.
                self.LowerThanMaxNum += 1
                if MainScore < self.LastResult:
                # decreasing consistently.
                    self.DecreasingNum += 1
                else:
                    self.DecreasingNum = 0
            self.LastResult = MainScore
        else:
            # Regression, the lower the better
            if MainScore < self.MaxResult:
                self.MaxResult = MainScore
                self.MaxResultModelIdx = ckpt_idx
                self.LowerThanMaxNum = 0
                self.DecreasingNum = 0
                # all set to 0.
            else:
                # decreasing, start to count.
                self.LowerThanMaxNum += 1
                if MainScore > self.LastResult:
                # decreasing consistently.
                    self.DecreasingNum += 1
                else:
                    self.DecreasingNum = 0
            self.LastResult = MainScore

        if self.LowerThanMaxNum > self.LowerThanMaxLimit:
            return True
        if self.DecreasingNum > self.DecreasingLimit:
            return True
        return False

    def BestModel(self):
        return self.MaxResult, self.MaxResultModelIdx, self.TestResult[self.MaxResultModelIdx]

    def GetControllerStatus(self):
        status = {
            'MaxResult': self.MaxResult,
            'MaxResultModelIdx': self.MaxResultModelIdx,
            'LastResult': self.LastResult,
            'LowerThanMaxNum': self.LowerThanMaxNum,
            'DecreasingNum': self.DecreasingNum,
            'TestResult': self.TestResult
        }
        return status

    def SetControllerStatus(self, status):
        self.MaxResult = status['MaxResult']
        self.MaxResultModelIdx = status['MaxResultModelIdx']
        self.LastResult = status['LastResult']
        self.LowerThanMaxNum = status['LowerThanMaxNum']
        self.DecreasingNum = status['DecreasingNum']
        self.TestResult = status['TestResult']

##########################################################################################################

class ConfigController(Controller):
    # A module to control the Configs of the training progress.
    # Including the configs for training, and the hyperparameters that should be searched.
    # The HyperParam Searching Methods can be modified.
    def __init__(self):
        super(ConfigController, self).__init__()

    def AdjustParams(self):
        raise NotImplementedError(
            "Adjust Params function not implemented.")

    def GetOpts(self):
        raise NotImplementedError(
            "Get Opts function not implemented.")

class GreedyConfigController(ConfigController):
    # Here the basic greedy searching strategy is implemented.

    def __init__(self, BasicHyperparamList, AdjustableHyperparamList, SpecificHyperparamList=None):
        # Basic: Configs for training, not for HyperParam Searching
        # Adjustable: Configs for greedy searching, candidates.
        # Specific: Specific group of HyperParams, not for greedy searching.
        super(ConfigController, self).__init__()
        self.BasicHyperparameterList = BasicHyperparamList
        self.HyperparameterList = AdjustableHyperparamList
        self.SpecificHyperparamList = SpecificHyperparamList
        self.opt = Configs(self.BasicHyperparameterList)
        self.MainMetric = self.BasicHyperparameterList['MainMetric']
        self.OnlySpecific = self.BasicHyperparameterList['OnlySpecific']

        # set the Trial Path for the experiment on this dataset.
        self.opt.set_args('TrialPath', self.opt.args['RootPath'] + self.opt.args['ExpName'] + '/')
        if not os.path.exists(self.opt.args['TrialPath']):
            os.mkdir(self.opt.args['TrialPath'])

        self.controllerstatussaver = ControllerStatusSaver(self.opt.args, 'ConfigController')
        status = self.LoadStatusFromFile()
        if status:
            self.SetControllerStatus(status)
        else:
            self.InitControllerStatus()
            self.CheckSpecificHyperparamList(SpecificHyperparamList)
            self.OptInit(SpecificHyperparamList)


    def CheckSpecificHyperparamList(self, SpecificHyperparamList):
        firstkey = list(SpecificHyperparamList.keys())[0]
        SpecificChoiceNum = len(SpecificHyperparamList[firstkey])
        for key in SpecificHyperparamList.keys():
            assert SpecificChoiceNum == len(SpecificHyperparamList[key])

    def OptInit(self, SpecificHyperparamList):
        if SpecificHyperparamList:
            self.HyperparameterInit(self.SpecificHyperparamList)
        else:
            self.HyperparameterInit(self.HyperparameterList)

    def HyperparameterInit(self, paramlist):
        for param in paramlist.keys():
            self.opt.set_args(param, paramlist.get(param)[0])
        # initially, the hyperparameters are set to be the first value of their candidate lists each.

    def GetOpts(self):
        self.opt.set_args('ExpDir', self.opt.args['TrialPath'] + 'exp' + str(self.exp_count) + '/')
        if not os.path.exists(self.opt.args['ExpDir']):
            os.mkdir(self.opt.args['ExpDir'])
        return self.opt

    def AdjustParams(self):
        # Adjust the hyperparameters by greedy search.
        # The return is the end flag

        # if the Specific Hyperparam List is given, then set the opts as the param group in SpecificParamList
        if self.SpecificHyperparamList:
            keys = self.SpecificHyperparamList.keys()
            if self.exp_count < len(self.SpecificHyperparamList.get(list(keys)[0])):
                for param in self.SpecificHyperparamList.keys():
                    self.opt.set_args(param, self.SpecificHyperparamList.get(param)[self.exp_count])
                return False
            elif self.exp_count == len(self.SpecificHyperparamList.get(list(keys)[0])):
                if self.OnlySpecific:
                    return True
                else:
                    self.HyperparameterInit(self.HyperparameterList)
                    self.result = []
                    return False

        # After trying the given specific params, using greedy search in the AdjustableParamList(HyperParameterList).
        ParamNames = list(self.HyperparameterList.keys())
        cur_param_name = ParamNames[self.parampointer]           # key, string
        cur_param = self.HyperparameterList[cur_param_name]      # list of values
        if self.paramvaluepointer < len(cur_param):
            # set the config
            cur_value = cur_param[self.paramvaluepointer]        # value
            self.opt.set_args(cur_param_name, cur_value)
            self.paramvaluepointer += 1
        else:
            # choose the best param value based on the results.
            assert len(self.result) == len(cur_param)

            if self.opt.args['ClassNum'] == 1:
                best_metric = min(self.result)
            else:
                best_metric = max(self.result)

            loc = self.result.index(best_metric)
            self.result = []
            self.result.append(best_metric)                      # best_metric is obtained by configs: {paraml:[loc], paraml+1:[0]}
                                                                 # so that we don't need to test the choice of paraml+1:[0]
                                                                 # just use the result tested when adjusting paraml.
            cur_param_best_value = cur_param[loc]
            self.opt.set_args(cur_param_name, cur_param_best_value)
            self.parampointer += 1
            self.paramvaluepointer = 1                           # test from paraml+1:[1]

            if self.parampointer < len(ParamNames):
                # set the config
                cur_param_name = ParamNames[self.parampointer]
                cur_param = self.HyperparameterList[cur_param_name]
                cur_value = cur_param[self.paramvaluepointer]
                self.opt.set_args(cur_param_name, cur_value)
                self.paramvaluepointer += 1
                return False
            else:
                return True



    def StoreResults(self, score):
        self.result.append(score)

    def LoadStatusFromFile(self):
        status = self.controllerstatussaver.LoadStatus()
        return status

    def SaveStatusToFile(self):
        self.GetControllerStatus()
        print(self.status)
        self.controllerstatussaver.SaveStatus(self.status)

    def InitControllerStatus(self):
        # running status of the ConfigController.
        self.exp_count = 0
        self.parampointer = 0
        self.paramvaluepointer = 1
        # two pointers indicates the param and its value that next experiment should use.
        self.result = []

    def GetControllerStatus(self):
        self.status = {
            'exp_count': self.exp_count,
            'parampointer': self.parampointer,
            'paramvaluepointer': self.paramvaluepointer,
            'result': self.result,
            'next_opt_args': self.opt.args
        }

    def SetControllerStatus(self, status):
        self.exp_count = status['exp_count']
        self.parampointer = status['parampointer']
        self.paramvaluepointer = status['paramvaluepointer']
        self.result = status['result']
        self.opt.args = status['next_opt_args']
        #print("Config Controller has been loaded. Experiments continue.")

##########################################################################################################

