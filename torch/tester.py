import sys
import os
import glob
import math
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import label_binarize
from itertools import cycle

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'common'))
import common.opts as opts
import resources.models as models
import resources.calculator as calc


class Trainer:
    def __init__(self, opt=None):
        self.opt = opt
        self.testX = None
        self.testY = None

    def load_test_data(self):
        test_samples = self.opt.nSamples[self.opt.dataset]
        data = np.load(os.path.join(self.opt.data, self.opt.dataset, 'test_data_{}khz/fold{}_test{}.npz'.format(
            self.opt.sr//1000, self.opt.split, test_samples)), allow_pickle=True)
        self.testX = torch.tensor(np.moveaxis(data['x'], 3, 1)).to(self.opt.device)
        self.testY = torch.tensor(data['y']).to(self.opt.device)

    def __validate(self, net, lossFunc):
        if self.testX is None:
            self.load_test_data()

        net.eval()
        with torch.no_grad():
            y_pred = None
            batch_size = (self.opt.batchSize // self.opt.nCrops) * self.opt.nCrops
            for idx in range(math.ceil(len(self.testX) / batch_size)):
                x = self.testX[idx * batch_size: (idx + 1) * batch_size]
                scores = net(x)
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data))

            acc, loss, precision, recall, f1 = self.__compute_metrics(y_pred, self.testY, lossFunc)
            confusion = self.__confusion_matrix(y_pred, self.testY)
            fpr, tpr, roc_auc = self.__ROC_AUC(y_pred, self.testY)

        net.train()
        return acc, loss, precision, recall, f1, confusion, fpr, tpr, roc_auc

    # Calculating average prediction (10 crops) and final accuracy
    def __compute_metrics(self, y_pred, y_target, lossFunc):
        with ((torch.no_grad())):
            y_pred, y_target = self.__get_pred_actual_y(y_pred, y_target)

        y_pred = y_pred.cpu()
        y_target = y_target.cpu()

        acc = (((y_pred == y_target) * 1).float().mean() * 100).item()
        loss = lossFunc(y_pred.float().log(), y_target.float()).item()
        precision = precision_score(y_target, y_pred, average='macro')
        recall = recall_score(y_target, y_pred, average='macro')
        f1 = f1_score(y_target, y_pred, average='macro')

        return acc, loss, precision, recall, f1

    def __confusion_matrix(self, y_pred, y_target):
        with torch.no_grad():
            y_pred, y_target = self.__get_pred_actual_y(y_pred, y_target)

        y_pred = y_pred.cpu()
        y_target = y_target.cpu()

        confusion = confusion_matrix(y_target, y_pred)
        return confusion

    def __ROC_AUC(self, y_pred, y_target):
        with torch.no_grad():
            y_pred, y_target = self.__get_pred_actual_y(y_pred, y_target)

        y_pred = y_pred.cpu()
        y_target = y_target.cpu()

        # Binarize the labels
        n_classes = self.opt.nClasses[self.opt.dataset]
        true_labels_bin = label_binarize(y_target, classes=range(1, n_classes+1))
        predicted_labels_bin = label_binarize(y_pred, classes=range(1, n_classes+1))

        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], predicted_labels_bin[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        return fpr, tpr, roc_auc

    def __get_pred_actual_y(self, y_pred, y_target):
        # Reshape to shape theme like each sample contains 10 samples, calculate mean and find the indices that
        # has the highest average value for each sample
        if self.opt.nCrops == 1:
            y_pred = y_pred.argmax(dim=1) + 1
            y_target = y_target.argmax(dim=1) + 1
        else:
            y_pred = (y_pred.reshape(y_pred.shape[0] // self.opt.nCrops, self.opt.nCrops, y_pred.shape[1])
                      ).mean(dim=1).argmax(dim=1) + 1
            y_target = (y_target.reshape(y_target.shape[0] // self.opt.nCrops, self.opt.nCrops, y_target.shape[1])
                        ).mean(dim=1).argmax(dim=1) + 1

        return y_pred, y_target

    def __save_metrics(self, acc, loss, precision, recall, f1):
        metrics_path = os.path.join(os.getcwd(), 'torch/metrics/metric_values')
        curr_datetime = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        filename = f'{self.opt.model_name.lower()}-metrics-{format(curr_datetime)}.txt'

        if not os.path.exists(metrics_path):
            os.makedirs(metrics_path)

        file_path = os.path.join(metrics_path, filename)
        with open(file_path, 'w') as file:
            file.write(f'Accuracy: {acc:.3f}\n')
            file.write(f'Loss: {loss:.3f}\n')
            file.write(f'Precision: {precision:.3f}\n')
            file.write(f'Recall: {recall:.3f}\n')
            file.write(f'F1 Score: {f1:.3f}\n')

    def __save_confusion_matrix(self, confusion):
        # Plot the confusion matrix
        plt.figure(figsize=(10, 10))
        labels = self.opt.class_labels[self.opt.dataset]
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        confusion_matrix_path = os.path.join(os.getcwd(), 'torch\\metrics\\confusion_matrices')
        curr_datetime = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        filename = f'{self.opt.model_name.lower()}-confusion_matrix-{format(curr_datetime)}.png'

        if not os.path.exists(confusion_matrix_path):
            os.makedirs(confusion_matrix_path)

        # Save the plot to the specified folder
        save_path = os.path.join(confusion_matrix_path, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=1000)

    def __save_ROC_AUC(self, fpr, tpr, roc_auc):
        n_classes = self.opt.nClasses[self.opt.dataset]
        labels = self.opt.class_labels[self.opt.dataset]

        # Plot ROC curves for each class
        plt.figure(figsize=(24, 24))

        hex_colors = [
            "#FF0000",  # Red
            "#00FF00",  # Green
            "#0000FF",  # Blue
            "#FFFF00",  # Yellow
            "#FF00FF",  # Magenta
            "#00FFFF",  # Cyan
            "#FFA500",  # Orange
            "#FF4500",  # OrangeRed
            "#FFD700",  # Gold
            "#8A2BE2",  # BlueViolet
            "#7FFF00",  # Chartreuse
            "#DC143C",  # Crimson
            "#48D1CC",  # MediumTurquoise
            "#2E8B57",  # SeaGreen
            "#800080",  # Purple
            "#ADFF2F",  # GreenYellow
            "#FF1493",  # DeepPink
            "#9370DB",  # MediumPurple
            "#8B4513",  # SaddleBrown
            "#20B2AA",  # LightSeaGreen
            "#8B008B",  # DarkMagenta
            "#FF6347",  # Tomato
            "#556B2F",  # DarkOliveGreen
            "#6B8E23",  # OliveDrab
            "#BDB76B",  # DarkKhaki
            "#808080",  # Gray
            "#DAA520",  # GoldenRod
        ]

        colors = cycle(hex_colors)  # Adjust as needed for your number of classes

        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{labels[i]} (AUC = {roc_auc[i]:.3f})')

        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')  # Diagonal line for reference
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve and AUC')
        plt.legend(loc='lower right')

        roc_auc_path = os.path.join(os.getcwd(), 'torch\\metrics\\roc_auc')
        curr_datetime = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        filename = f'{self.opt.model_name.lower()}-roc_auc-{format(curr_datetime)}.png'

        if not os.path.exists(roc_auc_path):
            os.makedirs(roc_auc_path)

        # Save the plot to the specified folder
        save_path = os.path.join(roc_auc_path, filename)
        plt.savefig(save_path, bbox_inches='tight')

    def TestModel(self, run=1):
        lossFunc = torch.nn.KLDivLoss(reduction='batchmean')
        dir = os.getcwd()
        net_path = self.opt.model_path
        print(net_path)
        file_paths = glob.glob(net_path)
        for f in file_paths:
            state = torch.load(f, map_location=self.opt.device)
            config = state['config']
            weight = state['weight']
            net = models.GetESCNASModel(self.opt.inputLength, nclass=self.opt.nClasses[self.opt.dataset], sr=self.opt.sr, channel_config=config).to(self.opt.device)
            net.load_state_dict(weight)
            print('Model found at: {}'.format(f))
            calc.summary(net, (1, 1, opt.inputLength))
            self.load_test_data()
            net.eval()
            # Test standard way with 10 crops
            val_acc, val_loss, precision, recall, f1, confusion, fpr, tpr, roc_auc = self.__validate(net, lossFunc)
            print('Testing - Val: Loss {:.3f}  Acc(top1) {:.2f}%'.format(val_loss, val_acc))
            print('Testing - Val: Precision {:.3f}  Recall {:.3f}  F1-Score {:.3f}'.format(precision, recall, f1))
            self.__save_metrics(val_acc, val_loss, precision, recall, f1)
            self.__save_confusion_matrix(confusion)
            self.__save_ROC_AUC(fpr, tpr, roc_auc)


if __name__ == '__main__':
    opt = opts.parse()
    valid_path = False
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    while not valid_path:
        model_path = input("Enter model path\n:")
        file_paths = glob.glob(os.path.join(os.getcwd(), model_path))

        if len(file_paths) > 0 and os.path.isfile(file_paths[0]):
            state = torch.load(file_paths[0], map_location=opt.device)
            opt.model_path = file_paths[0]
            opt.model_name = os.path.basename(opt.model_path).split(".")[0]
            print('Model has been found at: {}'.format(opt.model_path))
            valid_path = True

    valid_fold = False
    while not valid_fold:
        fold = input(
            "Select the fold on which the model was Validated:\n"
            " 1. Fold-1\n"
            " 2. Fold-2\n"
            " 3. Fold-3\n"
            " 4. Fold-4\n"
            " 5. Fold-5\n :")
        if fold in ['1', '2', '3', '4', '5']:
            opt.split = int(fold)
            valid_fold = True

    trainer = Trainer(opt)
    trainer.TestModel()
