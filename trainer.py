import os
import time
import torch
from torch.optim import Adam, Adamax, AdamW
import torch.nn.functional as F
from torchmetrics import MetricCollection, Accuracy, Recall, Specificity, Precision, F1Score, JaccardIndex, Dice, StatScores, ConfusionMatrix
from nn_models import UNet_Attention, R2U_Net, NestedUNet, UNet_3Plus, UNet, AttU_Net_Exp
from testing_deeplab import DeepLabV3Plus
import matplotlib.pyplot as plt
import seaborn as sns
# import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import JaccardLoss, DiceLoss, LovaszLoss, TverskyLoss, FocalLoss, SoftCrossEntropyLoss
import gc
from tqdm import tqdm
import pandas as pd
import numpy as np
# from torch.cuda.amp.grad_scaler import GradScaler
from misc import build_metric_plot
from sklearn.model_selection import KFold
from torch.utils.data import Subset, ConcatDataset, DataLoader
from fastFCN import FastFCN

class Trainer(object):
    def __init__(self, config, train_loader, validation_loader, test_loader):

        # Loading data
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        # Models
        self.u_net_model = None # Model
        self.optimizer = None # Optimizer
        self.img_ch = config.img_ch # Number of channels in image
        self.output_ch = config.output_ch # Number of output channels
        self.new_model = None
        self.averaged_model_path = None

        # Some Hyper-parameters
        self.lr = config.lr # Learning Rate
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.num_epochs = config.num_epochs # Number of Epochs
        self.num_epochs_decay = config.num_epochs_decay # Decay
        self.batch_size = config.batch_size # Batch size
        self.deep_supervision = config.deep_supervision
        self.weight_decay = config.weight_decay
        self.n_splits = config.n_splits

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode
        if self.mode == 'test':
            self.test_model_path = config.test_model_path

        #Devices
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        print("Device is: ", self.device)
        self.model_type = config.model_type # Type of model in config
        self.t = config.t
        self.optimizator = config.optimizator
        self.build_model()
        self.start_time = time.time() # Time of beginning the execution

        self.early_stopping = config.early_stopping
        if self.early_stopping == True:
            self.ear_stopping = EarlyStopping(patience= config.patience, verbose=config.verbose)

        # Setting Loss Criterion and mapping it to device
        if config.criterion == 'BCE':
            self.criterion = torch.nn.BCELoss().to(self.device) # Loss function: variants are BCELoss and BCEWithLogitsLoss
        elif config.criterion == 'LogitBCE':
            self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        elif config.criterion == 'Jaccard':
            self.criterion = JaccardLoss(mode='multiclass', classes = self.output_ch).to(self.device)
        elif config.criterion == 'Dice':
            self.criterion = DiceLoss(mode='multiclass', classes=self.output_ch).to(self.device)
        elif config.criterion == 'CrossEntropy':
            self.criterion = SoftCrossEntropyLoss()
        elif config.criterion == 'Tversky':
            self.criterion = TverskyLoss(mode='multiclass', classes = self.output_ch).to(self.device)
        elif config.criterion == 'Focal':
            self.criterion = FocalLoss(mode='multiclass').to(self.device)
        elif config.criterion == 'Lovasz':
            self.criterion = LovaszLoss(mode='multiclass').to(self.device)


        # Setting up Metric Collection, Source:  https://www.kaggle.com/code/shreydan/pytorch-torchmetrics-scheduler#PLOTS
        metrics = MetricCollection([
            Accuracy(task='multiclass', num_classes=self.output_ch).to(self.device),
            Recall(task='multiclass', num_classes=self.output_ch).to(self.device),
            Specificity(task='multiclass', num_classes=self.output_ch).to(self.device),
            Precision(task='multiclass', num_classes=self.output_ch).to(self.device),
            F1Score(task='multiclass', num_classes=self.output_ch).to(self.device),
            JaccardIndex(task='multiclass', num_classes=self.output_ch).to(self.device),
            Dice(task='multiclass', num_classes=self.output_ch).to(self.device),
            # ConfusionMatrix(task='multiclass', num_classes=self.output_ch).to(self.device)
        ]).to(self.device)

        self.training_metrics = metrics.clone(prefix='train_')
        self.validation_metrics = metrics.clone(prefix='val_')

        self.train_logs = []
        self.val_logs = []

        # Setting up Learning Rate Scheduler what helps to improve, Source: https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863
        self.is_scheduler = config.is_scheduler
        if config.scheduler == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                                  milestones=config.sch_milestones,
                                                                  gamma = config.sch_gamma)
        elif config.scheduler == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                             step_size=config.sch_step_size,
                                                             gamma = config.sch_gamma)
        elif config.scheduler == 'CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer,
                                                                        T_0=config.t_0,
                                                                        T_mult=config.t_mult,
                                                                        eta_min=config.eta_min)
            

        # self.scaler = GradScaler()

    #===== FUNCTION TO BUILD GENERATOR, DISCRIMINATOR AND MODEL DEVICE =====#
    def build_model(self):
        if self.model_type == 'UNet_Attention':
            self.u_net_model = UNet_Attention(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'R2U_Net':
            self.u_net_model = R2U_Net(img_ch=self.img_ch, output_ch=self.output_ch, t=self.t)
        elif self.model_type == 'NestedUNet':
            self.u_net_model = NestedUNet(img_ch=self.img_ch, output_ch=self.output_ch, deep_supervision=self.deep_supervision)
        elif self.model_type == 'UNet_3Plus':
            self.u_net_model = UNet_3Plus(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == 'UNet':
            self.u_net_model = UNet(in_channels=self.img_ch, out_channels=self.output_ch)
        elif self.model_type == 'AttU_Net_Exp':
            self.u_net_model = AttU_Net_Exp(img_ch = self.img_ch, output_ch=self.output_ch, is_ds=False)
        elif self.model_type == 'DeepLabV3Plus':
            self.u_net_model = DeepLabV3Plus(in_channels=self.img_ch, num_classes=self.output_ch)
        elif self.model_type == 'FastFCN':
            self.u_net_model = FastFCN(in_channels=self.img_ch, num_classes=self.output_ch)
            
        if self.optimizator == 'AdamW':
            self.optimizer = AdamW(list(self.u_net_model.parameters()),
                                    self.lr, [self.beta1, self.beta2],
                                    eps=1e-8,
                                    weight_decay=self.weight_decay
                                    )
        elif self.optimizator == 'Adamax':
            self.optimizer = Adamax(list(self.u_net_model.parameters()),
                                    self.lr, [self.beta1, self.beta2],
                                    eps=1e-8,
                                    weight_decay=self.weight_decay
                                    )
        elif self.optimizator == 'Adam':
            self.optimizer = Adam(list(self.u_net_model.parameters()),
                                    self.lr, [self.beta1, self.beta2],
                                    eps=1e-8,
                                    weight_decay=self.weight_decay
                                    )

        self.u_net_model.to(self.device)

    #===== FUNCTION TO ZERO THE GRADIENT BUFFERS =====#
    def reset_grad(self):
        self.u_net_model.zero_grad(set_to_none=True)


    #===== FUNCTION TO CLEAR CACHE IN CUDA =====#
    def clear(self):
        gc.collect()
        torch.cuda.empty_cache()

    #===== FUNCTION TO SAVE METRICS AND INFORMATION ABOUT EPOCH AND LOSS =====#
    def logger(self, epoch, metrics, loss, mode):
        
        metrics = {metric:metrics[metric].cpu().item() for metric in metrics}
    
        log = {
            f'epoch_{epoch-1}': {
                'loss': loss,
                'metrics': metrics
            }
        }
        if mode == 'train':
            self.train_logs.append(log)
        else:
            self.val_logs.append(log)
        # print(log)

    #===== FUNCTION TO PRINT INFORMATION AT THE END OF THE EPOCH =====#
    def print_per_epoch(self, epoch):
        print(f"\n\n{'-'*80}EPOCH {epoch}/{self.num_epochs}{'-'*80}")
        epoch = epoch-1
        time_dif = time.time() - self.start_time
        ex_length_h = time_dif // 3600
        time_dif %= 3600
        ex_length_m = time_dif // 60
        time_dif %= 60
        ex_length_s = time_dif

        train_loss = self.train_logs[epoch][f'epoch_{epoch}']['loss']
        train_acc = self.train_logs[epoch][f'epoch_{epoch}']['metrics']['train_MulticlassAccuracy']
        train_re = self.train_logs[epoch][f'epoch_{epoch}']['metrics']['train_MulticlassRecall']
        train_sp = self.train_logs[epoch][f'epoch_{epoch}']['metrics']['train_MulticlassSpecificity']
        train_pr = self.train_logs[epoch][f'epoch_{epoch}']['metrics']['train_MulticlassPrecision']
        train_f1s = self.train_logs[epoch][f'epoch_{epoch}']['metrics']['train_MulticlassF1Score']
        train_ji = self.train_logs[epoch][f'epoch_{epoch}']['metrics']['train_MulticlassJaccardIndex']
        train_dc = self.train_logs[epoch][f'epoch_{epoch}']['metrics']['train_Dice']

        val_loss = self.val_logs[epoch][f'epoch_{epoch}']['loss']
        val_acc = self.val_logs[epoch][f'epoch_{epoch}']['metrics']['val_MulticlassAccuracy']
        val_re = self.val_logs[epoch][f'epoch_{epoch}']['metrics']['val_MulticlassRecall']
        val_sp = self.val_logs[epoch][f'epoch_{epoch}']['metrics']['val_MulticlassSpecificity']
        val_pr = self.val_logs[epoch][f'epoch_{epoch}']['metrics']['val_MulticlassPrecision']
        val_f1s = self.val_logs[epoch][f'epoch_{epoch}']['metrics']['val_MulticlassF1Score']
        val_ji = self.val_logs[epoch][f'epoch_{epoch}']['metrics']['val_MulticlassJaccardIndex']
        val_dc = self.val_logs[epoch][f'epoch_{epoch}']['metrics']['val_Dice']

        print(f"Train ---> LOSS: {train_loss} | ACC: {train_acc} | RE: {train_re} | SP: {train_sp} | PR: {train_pr} | F1S: {train_f1s} | JI: {train_ji} | DC: {train_dc}")
        print(f"Validation ---> LOSS: {val_loss} | ACC: {val_acc} | RE: {val_re} | SP: {val_sp} | PR: {val_pr} | F1S: {val_f1s} | JI: {val_ji} | DC: {val_dc}")
        print("Training Time: %.2f hours %.2f minutes %.2f seconds\n\n\n"%(ex_length_h, ex_length_m, ex_length_s))


    #===== FUNCTION FOR TRAINING CYCLE =====#
    def training_step(self, epoch, train_loader=None):
        if train_loader == None:
            train_loader = self.train_loader
        # scaler = torch.cuda.amp.grad_scaler()
        # self.u_net_model.train(True)
        epoch_loss = 0.
        # iou_epoch_loss = 0.
        # self.reset_grad()
        num_batches = len(train_loader)
        progress = tqdm(train_loader, total=num_batches)

        for X, y in progress:
            X, y = X.to(self.device,
                        #  non_blocking=True
                         ), y.to(self.device,
                                #   non_blocking=True
                                  )
            # with torch.cuda.amp.autocast():
            preds = self.u_net_model(X)
            loss = self.criterion(preds, y)
            epoch_loss += loss.item()

            # Scale loss and send backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.training_metrics(torch.argmax(preds, dim=1), y)
            progress.set_description(f"Train Loss Step: {loss.item():.3f}")

            del X,y,preds,loss
        
        # Apply scheduler if exists in config
        if self.is_scheduler == True:
            self.scheduler.step()

        # Compute metrics and loss over the epoch 
        metrics = self.training_metrics.compute()
        epoch_loss /= num_batches

        self.logger(epoch,metrics,epoch_loss, 'train')

        del metrics

    
    #===== FUNCTION FOR VALIDATION CYCLE =====#
    @torch.no_grad()
    def validation_step(self, epoch, validation_loader=None):
        if validation_loader == None:
            validation_loader = self.validation_loader
        epoch_loss = 0.
        num_batches = len(validation_loader)
        progress = tqdm(validation_loader, total=num_batches)
        for X,y in progress:
            X, y = X.to(self.device,
                        #  non_blocking=True
                         ), y.to(self.device,
                                #   non_blocking=True
                                  )
            # with torch.cuda.amp.autocast():
            preds = self.u_net_model(X)
            loss = self.criterion(preds, y)
            epoch_loss += loss.item()

            self.validation_metrics(torch.argmax(preds, dim=1), y)
            progress.set_description(f"Validation Loss Step: {loss.item():.3f}")
            del X,y,preds,loss

        metrics = self.validation_metrics.compute()
        epoch_loss /= num_batches
        self.logger(epoch,metrics,epoch_loss, 'val')

        del metrics


    #===== UPDATED FUNCTION FOR TRAINING AND VALIDATION BASED ON FUNCTIONAL APPROACH =====#
    def train(self):
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.allow_tf32 = True
        # torch.backends.cuda.matmul.allow_tf32 = True

        u_net_path = os.path.join(self.model_path, '%s-%d-%.4f-%d.pkl' %(
            self.model_type, self.num_epochs, self.lr, self.num_epochs_decay
        )) # Create a path to the model to save model config file in *.pkl format
        # u_net_path_pth = os.path.join(self.model_path, '%s-%d-%.4f-%d.pth' %(
        #     self.model_type, self.num_epochs, self.lr, self.num_epochs_decay
        # )) # Create a path to the model to save model config file in *.pth format

        # Create a variable for best score
        best_model_score = 0.

        # Show a progress
        fit_progress = tqdm(
            range(1, self.num_epochs+1),
            leave = True,
            desc="Training..."
        )

        for epoch in fit_progress:
            self.u_net_model.train()
            fit_progress.set_description(f"EPOCH {epoch} / {self.num_epochs} | Training...")
            print("\n")
            self.training_step(epoch)
            self.clear()

            print("\n\n\n")

            self.u_net_model.eval()
            fit_progress.set_description(f"EPOCH {epoch} / {self.num_epochs} | Validation...")
            print("\n")
            self.validation_step(epoch)
            self.clear()

            print("\n\n\n")

            # Reset Metrics
            self.training_metrics.reset()
            self.validation_metrics.reset()

            # Print Metrics
            self.print_per_epoch(epoch)

            # Evaluate a model score as sum of Jaccard Index and Dice and save the model if this is a best one
            model_score = self.val_logs[epoch-1][f'epoch_{epoch-1}']['metrics']['val_MulticlassJaccardIndex'] + self.val_logs[epoch-1][f'epoch_{epoch-1}']['metrics']['val_Dice']

            if model_score > best_model_score:
                best_model_score = model_score
                print('Best %s Model Score = %.4f'%(self.model_type, best_model_score))
                torch.save(self.u_net_model.state_dict(), u_net_path) # Saving in Pickle Format
                # torch.save(self.u_net_model.state_dict(), u_net_path_pth) # Saving in *.pth format
                print("Model saved\n")
            
            if self.early_stopping == True:
                self.ear_stopping(self.val_logs[epoch-1][f'epoch_{epoch-1}']['loss'], self.u_net_model)
                if self.ear_stopping.early_stop:
                    print("Early stopping")
                    break

        self.clear()

    def train_k_val(self, train_ds, val_ds):

        # Initialize the KFold 
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=8)

        # Concatenate Train and Validation Datasets
        concat_ds = ConcatDataset([train_ds, val_ds])

        #Initialize the very best model score
        very_best_model_score = 0.

        for fold, (train_idx, val_idx) in enumerate(kf.split(concat_ds)):
            
            # Initialize model best score
            best_model_score = 0.

            # Create Data Loaders for the fold
            train_subset = Subset(concat_ds, train_idx)
            val_subset = Subset(concat_ds, val_idx)
            train_loader = DataLoader(train_subset, batch_size = self.batch_size, shuffle=True, drop_last=True, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size = self.batch_size, shuffle=True, drop_last=True, pin_memory=True)

            # Initialize model and path for the model
            self.build_model()
            self.u_net_path = os.path.join(self.model_path, '%s-%d.pt' %(
                                           self.model_type, fold))

            # Initialize Empty logs for Every fold
            self.train_logs = []
            self.val_logs = []
            print(f"\n\n{'-'*80}FOLD {fold+1}/{self.n_splits}{'-'*80}")
            # Show a progress
            fit_progress = tqdm(
                range(1, self.num_epochs+1),
                leave = True,
                desc="Training..."
            )
            for epoch in fit_progress:
                self.u_net_model.train()
                fit_progress.set_description(f"EPOCH {epoch} / {self.num_epochs} | Training...")
                print("\n")
                self.training_step(epoch, train_loader = train_loader)
                self.clear()

                print("\n\n\n")

                self.u_net_model.eval()
                fit_progress.set_description(f"EPOCH {epoch} / {self.num_epochs} | Validation...")
                print("\n")
                self.validation_step(epoch, validation_loader = val_loader)
                self.clear()

                print("\n\n\n")

                # Reset Metrics
                self.training_metrics.reset()
                self.validation_metrics.reset()

                # Print Metrics
                self.print_per_epoch(epoch)

                # Evaluate a model score as sum of Jaccard Index and Dice and save the model if this is a best one
                model_score = self.val_logs[epoch-1][f'epoch_{epoch-1}']['metrics']['val_MulticlassJaccardIndex'] + self.val_logs[epoch-1][f'epoch_{epoch-1}']['metrics']['val_Dice']

                if model_score > best_model_score:
                    best_model_score = model_score
                    print('Best %s Model Score = %.4f'%(self.model_type, best_model_score))
                    torch.save(self.u_net_model, self.u_net_path) # Saving in Pickle Format
                    print("Model saved\n")
                    if model_score > very_best_model_score:
                        very_best_model_score = model_score
                        print('Best %s Model score among all folds = %.4f'%(self.model_type, best_model_score))
                        path = os.path.join(self.model_path, '%s-best.pt' %(
                                           self.model_type))
                        torch.save(self.u_net_model, path)

            self.clear()

            # Building plots
            try:
                build_metric_plot(self.num_epochs, self.train_logs, self.val_logs, 'loss', self.model_type + str(fold))
                build_metric_plot(self.num_epochs, self.train_logs, self.val_logs, 'MulticlassAccuracy', self.model_type + str(fold))
                build_metric_plot(self.num_epochs, self.train_logs, self.val_logs, 'MulticlassJaccardIndex', self.model_type + str(fold))
            except Exception as e:
                print("During Plot building an exception occured: ", e)

    # Function that allows to load all models that were created during K-Fold
    def create_avgd_model(self):
        """
           Function that allows to load all models that were created during K-Fold training
           and create a new model as an average of weights of all the models
        """
        
        # Initialize a new model with the same architecture as the trained models
        self.build_model()
        self.new_model = self.u_net_model
#         self.new_model = UNet(in_channels=self.img_ch, out_channels=self.output_ch)
        
        models = []
        
        for i in range(self.n_splits):
            path = os.path.join(self.model_path, '%s-%d.pt' %(
                                       self.model_type, i))
            model = torch.load(path)
            models.append(model)
        for key in models[0].state_dict():
            models[0].state_dict()[key] = (models[0].state_dict()[key] 
                                           + models[1].state_dict()[key] 
                                           + models[2].state_dict()[key] 
                                           + models[3].state_dict()[key] 
                                           + models[4].state_dict()[key]
                                           ) / self.n_splits

        # Update the weights of the new model with the averaged weights
        self.new_model.load_state_dict(models[0].state_dict())
        
        # Save the Averaged Model
        self.averaged_model_path = os.path.join(self.model_path, '%s-averaged.pkl' %(self.model_type))
        torch.save(self.new_model.state_dict(), self.averaged_model_path) # Saving in Pickle Format


    @torch.no_grad()
    def testing_loop(self):
        stat_scores = StatScores(task='multiclass', reduce='macro', num_classes=self.output_ch,
                                 mdmc_reduce='global', average=None).to(self.device)

        acc = Accuracy(task='multiclass', num_classes=self.output_ch, average='micro',
                       mdmc_average='global').to(self.device)

        jaccard = JaccardIndex(task='multiclass', num_classes=self.output_ch).to(self.device)

        conf_matrix = ConfusionMatrix(task='multiclass', num_classes=self.output_ch, normalize='true', validate_args=True).to(self.device)

#         u_net_path = self.test_model_path
        if self.n_splits == None:
            u_net_path = self.test_model_path
            self.u_net_model.load_state_dict(torch.load(u_net_path))
            self.u_net_model.train(False)
            self.u_net_model.eval()

            # class_probs = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            # num_samples = {0:0, 1:0, 2:0, 3:0, 4:0}

            class_probs = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5:0}
            num_samples = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

            num_batches = len(self.test_loader)
            progress = tqdm(self.test_loader, total=num_batches)

            for X,y in progress:
                X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                logits = F.softmax(self.u_net_model(X), dim=1)
                aggr = torch.max(logits, dim=1)

                probs, preds = aggr[0], aggr[1]

                for label in class_probs.keys():
                    class_probs[label] += probs[preds==label].flatten().sum()
                    num_samples[label] += preds[preds==label].flatten().size(dim=0)

                stat_scores.update(preds, y)
                acc.update(preds, y)
                jaccard.update(preds, y)
                conf_matrix.update(preds, y)
        else:
            if self.new_model != None:
                if self.averaged_model_path != None:
                    self.new_model.load_state_dict(torch.load(self.averaged_model_path))
                    self.new_model.train(False)
                    self.new_model.eval()
                    
                    # class_probs = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                    # num_samples = {0:0, 1:0, 2:0, 3:0, 4:0}

                    class_probs = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5:0}
                    num_samples = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}

                    num_batches = len(self.test_loader)
                    progress = tqdm(self.test_loader, total=num_batches)

                    for X,y in progress:
                        X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                        logits = F.softmax(self.new_model(X), dim=1)
                        aggr = torch.max(logits, dim=1)

                        probs, preds = aggr[0], aggr[1]

                        for label in class_probs.keys():
                            class_probs[label] += probs[preds==label].flatten().sum()
                            num_samples[label] += preds[preds==label].flatten().size(dim=0)

                        stat_scores.update(preds, y)
                        acc.update(preds, y)
                        jaccard.update(preds, y)
                        conf_matrix.update(preds, y)

        for label in class_probs.keys():
            class_probs[label] /= num_samples[label]

        return stat_scores.compute(), acc.compute(), jaccard.compute(), conf_matrix.compute(), class_probs

    def class_report(self, classes, scores, acc, jaccard, conf_matrix, class_probs):
        print(f"{10*' '}precision{10*' '}recall{10*' '}f1-score{10*' '}support\n")
        acc = float(acc.cpu())
        jaccard = float(jaccard.cpu())

        for i, target in enumerate(classes):
            precision = float((scores[i,0]/(scores[i,0]+scores[i,1])).cpu())
            recall = float((scores[i,0]/(scores[i,0]+scores[i,3])).cpu())
            f1 = (2*precision*recall)/(precision + recall)
            print(f"{target}{10*' '}{precision:.2f}{10*' '}{recall:.2f}{10*' '}{f1:.2f}{10*' '}{scores[i,4]}")
        
        print(f"\n- Total accuracy:{acc:.4f}\n")
        print(f"- Mean IoU: {jaccard:.4f}\n")
        print("- Class probs")
        for idx in class_probs.keys():
            print(f"{classes[idx]}:{class_probs[idx].cpu():.3f}")

        df_cm = pd.DataFrame(conf_matrix.cpu(), index = [i for i in classes], columns = [i for i in classes])

        plt.figure(figsize=(12, 7))
        sns.heatmap(df_cm, annot=True)
        plt.savefig(self.model_type + "_conf_matr.png")
#**********************************************************************************************************************#


########################################################################################################################
################### CLASS THAT ALLOWS TO PERFORM EARLY STOPPING IF VALIDATION LOSS DOES NOT IMPROVE ####################
########################################################################################################################
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
##########################################################
##########################################################