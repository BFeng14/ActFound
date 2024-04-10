from TrainingFramework.Metrics import *

class Evaluator(object):
    # module to conduct evaluation step of the experimental progress.
    def __init__(self):
        super(Evaluator, self).__init__()

    def eval(self, evalloader, model, metrics):
        raise NotImplementedError(
            'valid function is not implemented.'
        )

class GeneralEvaluator(object):
    def __init__(self, opt):
        super(GeneralEvaluator, self).__init__()
        self.opt = opt
        # todo(zqzhang): updated in TPv7
        self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')


    def eval(self, evalloader, model, metrics):
        # todo(zqzhang): updated in TPv7
        # print(f"length: {len(evalloader)}")
        if len(evalloader) == 0:
            if self.opt.args['ClassNum'] == 1:
                return {'MAE': 1e9, 'RMSE': 1e9}
            else:
                return {'AUC': 0, 'ACC': 0}

        model.eval()
        All_answer = []
        All_label = []
        for i in range(self.opt.args['TaskNum']):
            All_answer.append([])
            All_label.append([])

        for ii, data in enumerate(evalloader):
            # one molecule input, but batch is not 1. Different Frags of one molecule consist of a batch.
            if self.opt.args['Model'] == 'Graphormer':
                if self.opt.args['AC']:
                    (data1, data2) = data
                    x, attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, edge_input, Label, idx = \
                        data1.x, data1.attn_bias, data1.attn_edge_type, data1.spatial_pos, \
                        data1.in_degree, data1.out_degree, data1.edge_input, data1.y, data1.idx
                    Input1 = {
                        'x': x.to(t.device(
                            f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'attn_bias': attn_bias.to(t.device(
                            f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'attn_edge_type': attn_edge_type.to(t.device(
                            f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'spatial_pos': spatial_pos.to(t.device(
                            f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'in_degree': in_degree.to(t.device(
                            f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'out_degree': out_degree.to(t.device(
                            f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'edge_input': edge_input.to(t.device(
                            f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
                    }
                    x, attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, edge_input, Label, idx = \
                        data2.x, data2.attn_bias, data2.attn_edge_type, data2.spatial_pos, \
                        data2.in_degree, data2.out_degree, data2.edge_input, data2.y, data2.idx
                    Input2 = {
                        'x': x.to(t.device(
                            f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'attn_bias': attn_bias.to(t.device(
                            f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'attn_edge_type': attn_edge_type.to(t.device(
                            f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'spatial_pos': spatial_pos.to(t.device(
                            f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'in_degree': in_degree.to(t.device(
                            f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'out_degree': out_degree.to(t.device(
                            f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'edge_input': edge_input.to(t.device(
                            f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
                    }
                    Input = [Input1, Input2]
                else:
                ######
                    x,attn_bias,attn_edge_type,spatial_pos,in_degree,out_degree,edge_input,Label,idx =\
                        data.x, data.attn_bias, data.attn_edge_type,data.spatial_pos,\
                        data.in_degree, data.out_degree, data.edge_input, data.y, data.idx
                    Input = {
                        'x': x.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'attn_bias': attn_bias.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'attn_edge_type': attn_edge_type.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'spatial_pos': spatial_pos.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'in_degree': in_degree.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'out_degree': out_degree.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                        'edge_input': edge_input.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
                    }
                #########
                Label = Label.unsqueeze(-1)

            else:
                [Input, Label, Idx] = data
                # todo(zqzhang): updated in TPv7
                Input = Input.to(self.device)

            # the sample provided by dataloader will add a new dimension 'batch' at dim=0, even if batch size is set to be 1.
            # todo(zqzhang): updated in TPv7
            # print(f"Label: {Label.size()}")
            Label = Label.to(self.device)
            Label = Label.squeeze(-1)  # [batch, task]
            Label = Label.t()  # [task,batch]
            # print(f"Label: {Label.size()}")
            # for Label, different labels in a batch are actually the same, for they are exactly one molecule.
            # so the batch dim of Label is not exactly useful.
            output = model(Input)  # [batch, output_size]
            #output = output.mean(dim = 0, keepdims = True)  # [1, output_size]

            for i in range(self.opt.args['TaskNum']):
                cur_task_output = output[:,
                                  i * self.opt.args['ClassNum']: (i + 1) * self.opt.args['ClassNum']]  # [1, ClassNum]

                # todo(zqzhang): updated in TPv7
                if self.opt.args['Model'] != 'FraGAT':
                    cur_task_label = Label[i]      # [batch]
                    # print(cur_task_label.size())
                    for j in range(len(cur_task_label)):
                        cur_task_cur_sample_label = cur_task_label[j]   #[1]
                        cur_task_cur_sample_output = cur_task_output[j]   #[ClassNum]
                        # print(cur_task_cur_sample_label.size())
                        if cur_task_cur_sample_label == -1:
                            continue
                        else:
                            # print(f"cur_task_cur_sample_label: {cur_task_cur_sample_label}")
                            # print(f"cur_task_cur_sample_label.item(): {cur_task_cur_sample_label.item()}")
                            All_label[i].append(cur_task_cur_sample_label.item())
                            # print(f"cur_task_cur_sample_output_size: {cur_task_cur_sample_output.size()}")
                            All_answer[i].append(cur_task_cur_sample_output.tolist())
                            # for ii, data in enumerate(cur_task_output.tolist()):
                            #     All_answer[i].append(data)
                else:
                ######
                    cur_task_label = Label[i][0]  # all of the batch are the same, so only picking [i][0] is enough.
                    if cur_task_label == -1:
                        continue
                    else:
                        All_label[i].append(cur_task_label.item())
                        for ii, data in enumerate(cur_task_output.tolist()):
                            All_answer[i].append(data)
                ######

        scores = {}
        All_metrics = []
        for i in range(self.opt.args['TaskNum']):
            # for each task, the All_label and All_answer contains the samples of which labels are not missing
            All_metrics.append([])
            label = All_label[i]
            answer = All_answer[i]
            # print(f"label: {len(label)}")
            # print(f"answer: {len(answer)}")
            assert len(label) == len(answer)
            for metric in metrics:
                result = metric.compute(answer, label)
                All_metrics[i].append(result)
                # if multitask, then print the results of each tasks.
                if self.opt.args['TaskNum'] != 1:
                    print("The value of metric", metric.name, "in task", i, 'is: ', result)
        average = t.Tensor(All_metrics).mean(dim = 0)  # dim 0 is the multitask dim.
        # the lenght of average is metrics num

        for i in range(len(metrics)):
            scores.update({metrics[i].name: average[i].item()})
            print("The average value of metric", metrics[i].name, "is: ", average[i].item())

        model.train()

        if self.opt.args['EvalLogAllPreds']:
            print("All samples in Evalset and their predictions:")
            self.AllLabel = All_label
            self.AllPred = All_answer

        return scores

class PyGEvaluator(Evaluator):
    def __init__(self, opt):
        super(PyGEvaluator, self).__init__()
        self.opt = opt

    def eval(self, evalloader, model, metrics):
        model.eval()
        All_answer = []
        All_label = []
        for i in range(self.opt.args['TaskNum']):
            All_answer.append([])
            All_label.append([])

        for data in evalloader:
            Label = data.y
            # Label: [batch, task, 1]
            # print("Input size:")
            # print(Input[0].size())
            # print('Label size:')
            # print(Label.size())

            Label = Label.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))  # [batch, task, 1]
            # Label = Label.squeeze(-1)  # [batch, task]
            Label = Label.t()  # [task, batch]

            output = model(data)  # [batch, TaskNum * ClassNum]
            # print("Output size:")
            # print(output.size())

            for i in range(self.opt.args['TaskNum']):
                cur_task_output = output[:,
                                  i * self.opt.args['ClassNum']: (i + 1) * self.opt.args['ClassNum']]
                # print(cur_task_output.size())   # [batch_size, ClassNum]
                cur_task_label = Label[i]  # [batch_size]
                # print(cur_task_label.size())

                cur_task_cur_batch_valid_labels = []
                cur_task_cur_batch_valid_answers = []
                for j in range(len(cur_task_label)):
                    l = cur_task_label[j]
                    if l == -1:
                        continue
                    else:
                        cur_task_cur_batch_valid_labels.append(l.item())
                        cur_task_cur_batch_valid_answers.append(cur_task_output[j].tolist())

                        # for ii, data in enumerate(cur_task_output[j].tolist()):
                        #     print('data size:')
                        #     print(len(data))
                        # cur_task_cur_batch_valid_answers.append(data)

                # print('cur_task_cur_batch_valid_labels size:')
                # print(len(cur_task_cur_batch_valid_labels))
                # print('cur_task_cur_batch_valid_answers size:')
                # print(len(cur_task_cur_batch_valid_answers))

                for ii, item in enumerate(cur_task_cur_batch_valid_labels):
                    All_label[i].append(item)
                for ii, item in enumerate(cur_task_cur_batch_valid_answers):
                    # print('item size:')
                    # print(len(item))
                    All_answer[i].append(item)

                # if cur_task_label[0] == -1:
                #     continue
                # else:
                #     All_label[i].append(cur_task_label.item())
                #     for ii, data in enumerate(cur_task_output.tolist()):
                #         All_answer[i].append(data)

            # print("")

        scores = {}
        All_metrics = []
        for i in range(self.opt.args['TaskNum']):
            # for each task, the All_label and All_answer contains the samples of which labels are not missing
            All_metrics.append([])
            label = All_label[i]
            answer = All_answer[i]
            # print('label:')
            # print(len(label))
            # print('answer:')
            # print(len(answer))
            assert len(label) == len(answer)
            for metric in metrics:
                result = metric.compute(answer, label)
                All_metrics[i].append(result)
                # if multitask, then print the results of each tasks.
                if self.opt.args['TaskNum'] != 1:
                    print("The value of metric", metric.name, "in task", i, 'is: ', result)
        average = t.Tensor(All_metrics).mean(dim = 0)  # dim 0 is the multitask dim.
        # the lenght of average is metrics num

        for i in range(len(metrics)):
            scores.update({metrics[i].name: average[i].item()})
            print("The average value of metric", metrics[i].name, "is: ", average[i].item())

        model.train()
        return scores

class MLPEvaluator(Evaluator):
    def __init__(self, opt):
        super(MLPEvaluator, self).__init__()
        self.opt = opt
        # todo(zqzhang): updated in TPv7
        self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')

    def eval(self, evalloader, model, metrics):
        model.eval()
        All_answer = []
        All_label = []
        for i in range(self.opt.args['TaskNum']):
            All_answer.append([])
            All_label.append([])

        for ii, data in enumerate(evalloader):
            [Input, Label, _] = data        # Input: [2, batch, nBits]
                                            # Label: [batch, task, 1]
            # print("Input size:")
            # print(Input[0].size())
            # print('Label size:')
            # print(Label.size())

            # todo(zqzhang): updated in TPv7
            if Input.__class__ == t.Tensor:
                Input = Input.to(self.device)
            Label = Label.to(self.device)            #[batch, task, 1]
            Label = Label.squeeze(-1)       #[batch, task]
            Label = Label.t()               #[task, batch]

            output = model(Input)           #[batch, TaskNum * ClassNum]
            # print("Output size:")
            # print(output.size())

            for i in range(self.opt.args['TaskNum']):
                cur_task_output = output[:,
                                  i * self.opt.args['ClassNum']: (i + 1) * self.opt.args['ClassNum']]
                # print(cur_task_output.size())   # [batch_size, ClassNum]
                cur_task_label = Label[i]       # [batch_size]
                # print(cur_task_label.size())

                cur_task_cur_batch_valid_labels = []
                cur_task_cur_batch_valid_answers = []
                for j in range(len(cur_task_label)):
                    l = cur_task_label[j]
                    if l == -1:
                        continue
                    else:
                        cur_task_cur_batch_valid_labels.append(l.item())
                        cur_task_cur_batch_valid_answers.append(cur_task_output[j].tolist())

                        # for ii, data in enumerate(cur_task_output[j].tolist()):
                        #     print('data size:')
                        #     print(len(data))
                            # cur_task_cur_batch_valid_answers.append(data)

                # print('cur_task_cur_batch_valid_labels size:')
                # print(len(cur_task_cur_batch_valid_labels))
                # print('cur_task_cur_batch_valid_answers size:')
                # print(len(cur_task_cur_batch_valid_answers))

                for ii, item in enumerate(cur_task_cur_batch_valid_labels):
                    All_label[i].append(item)
                for ii, item in enumerate(cur_task_cur_batch_valid_answers):
                    # print('item size:')
                    # print(len(item))
                    All_answer[i].append(item)

                # if cur_task_label[0] == -1:
                #     continue
                # else:
                #     All_label[i].append(cur_task_label.item())
                #     for ii, data in enumerate(cur_task_output.tolist()):
                #         All_answer[i].append(data)

            # print("")

        scores = {}
        All_metrics = []
        for i in range(self.opt.args['TaskNum']):
            # for each task, the All_label and All_answer contains the samples of which labels are not missing
            All_metrics.append([])
            label = All_label[i]
            answer = All_answer[i]
            # print('label:')
            # print(len(label))
            # print('answer:')
            # print(len(answer))
            assert len(label) == len(answer)
            for metric in metrics:
                result = metric.compute(answer, label)
                All_metrics[i].append(result)
                # if multitask, then print the results of each tasks.
                if self.opt.args['TaskNum'] != 1:
                    print("The value of metric", metric.name, "in task", i, 'is: ', result)
        average = t.Tensor(All_metrics).mean(dim = 0)  # dim 0 is the multitask dim.
        # the lenght of average is metrics num

        for i in range(len(metrics)):
            scores.update({metrics[i].name: average[i].item()})
            print("The average value of metric", metrics[i].name, "is: ", average[i].item())

        model.train()
        # todo(zqzhang): updated in ACv8
        # print("7654321")
        if self.opt.args['EvalLogAllPreds']:
            # print("1234567")
            self.AllLabel = All_label
            self.AllPred = All_answer
        return scores

