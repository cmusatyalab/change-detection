import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import pdb
import argparse
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from scipy import ndimage
import torchvision.models as M
import torch.nn.functional as F
from vl_cmu_cd_dataset import *
import torchvision.utils
from skimage import io

from networks import *
#import utils
from tensorboardX import SummaryWriter
import os

from sklearn import metrics
import matplotlib.pyplot as plt


class ExperimentRunner(object):
    """
    Main Class for running experiments
    This class creates the UNET
    This class also creates the datasets
    """
    def __init__(self, model_arch, train_dataset_path, test_dataset_path, train_batch_size, test_batch_size, 
                model_save_dir, num_epochs=100, num_data_loader_workers=10, num_classes=10, encoder_path=None, decoder_path=None):
        # GAN Network + VGG Loss Network
        if model_arch == 'vgg16':
            self.model = UNet()
        
        self.model.encoder.load_state_dict(torch.load(encoder_path))
        self.model.decoder1.load_state_dict(torch.load(decoder_path))
        self.model.decoder2.load_state_dict(torch.load(decoder_path))       

        # for param in self.model.encoder.parameters():
        #     param.requires_grad = False
        # for param in self.model.decoder1.parameters():
        #     param.requires_grad = False
        # for param in self.model.decoder2.parameters():
        #     param.requires_grad = False

        for param in self.model.encoder.convblock1.parameters():
            param.requires_grad = False
        for param in self.model.encoder.convblock2.parameters():
            param.requires_grad = False
        for param in self.model.encoder.convblock3.parameters():
            param.requires_grad = False
        for param in self.model.encoder.convblock4.parameters():
            param.requires_grad = False
        for param in self.model.encoder.convblock5.parameters():
            param.requires_grad = False
        # Network hyperparameters
        self.lr = 1.e-4
        self.num_classes = num_classes
        self.optimizer = torch.optim.Adam([ {'params': self.model.parameters(), 'lr': self.lr}
                                            #{'params': self.gan.discriminator.parameters(), 'lr': self.disc_lr}
                                         ], betas=(0.5, 0.999))
        # self.optimizer = torch.optim.SGD([ {'params': self.model.parameters(), 'lr': self.lr}
        #                                     #{'params': self.gan.discriminator.parameters(), 'lr': self.disc_lr}
        #                                  ], momentum=0)
        # Network losses
        self.criterion = nn.CrossEntropyLoss().cuda()

        # Train settings + log settings
        self.num_epochs = num_epochs
        self.log_freq = 10  # Steps
        self.test_freq = 500  # Steps
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        # Create datasets
        self.train_dataset = VLCMUCDDataset(train_dataset_path, training=True)
        self.test_dataset = VLCMUCDDataset(test_dataset_path, training=False)

        self.train_dataset_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=num_data_loader_workers)
        self.test_dataset_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.test_batch_size)

        # Use the GPU if it's available.
        self.cuda = torch.cuda.is_available()

        if self.cuda:
            self.model.cuda()

        # Tensorboard logger
        self.txwriter = SummaryWriter()
        self.model_save_dir = model_save_dir
        self.save_freq = 26
        self.display_freq = 100
        self.test_display_freq = 50

        self.lr_decay_frequency = 5000
        self.lr_decay_rate = 0.5
        self.save_thresh = 0.7



        #self.model = nn.DataParallel(self.model)

    def _optimize(self, y_pred, y_gt):
        """
        VGGLoss + GAN loss
        """
        self.optimizer.zero_grad()
        loss = self.criterion(y_pred, y_gt)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def _adjust_learning_rate(self, epoch):
        """
        TODO
        """
        #if (epoch % self.lr_decay_frequency) == self.lr_decay_frequency-1:
        if self.lr_decay_frequency == epoch:
            #self.lr *= (self.lr_decay_rate ** (epoch // self.lr_decay_frequency))
            self.lr *= self.lr_decay_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        return

    def _clip_weights(self):
        """
        TODO
        """
        raise NotImplementedError()
    
    def test(self, epoch):
        num_batches = len(self.test_dataset_loader)
        test_accuracies = AverageMeter()
        test_bin_accuracies = AverageMeter()
        test_multi_accuracies = AverageMeter()
        test_bin_precision = AverageMeter()
        test_bin_recall = AverageMeter()
        test_bin_fscore = AverageMeter()
        gt_all = []
        pos_class_all = []
        # gt_all = np.zeros((429, 224*224),dtype=np.bool_)
        # pos_class_all = np.zeros((429, 224*224),dtype=np.float32)
        # assumes test batch size of 1
        for batch_id, batch_data in enumerate(self.test_dataset_loader):
            current_step = batch_id
            # Set to eval()
            self.model.eval()        
            # Get data from dataset
            im1 = batch_data['im1'].cuda(async=True)
            im2 = batch_data['im2'].cuda(async=True)
            gt = batch_data['gt'].cuda(async=True)
            gt = gt.view(gt.shape[0],gt.shape[2],gt.shape[3])
            dont_ignore = gt != -1
            ignore = gt == -1
            gt[ignore] = 0

            # ========================
            # Make prediction
            # ========================
            pred = self.model(im1, im2)
            binary_score_nochange = pred[:,0,:,:]
            (binary_score_change, _ ) = torch.max(pred[:,1:,:,:],dim=1)
            raw_scores = F.softmax(pred, dim=1 )
            
            pred_ans = torch.argmax( raw_scores, dim=1 )
            pred_ans[ignore] = 0
            bin_pred_ans = pred_ans != torch.zeros_like(pred_ans).cuda(async=True)
            bin_pred_gt = gt != torch.zeros_like(pred_ans).cuda(async=True)

            roc_bin = torch.cat( (binary_score_nochange , binary_score_change),dim=0)
            roc_bin = F.softmax(roc_bin, dim=0 )
            roc_bin = roc_bin[1,:,:]

            acc = 100.0 * torch.mean( (pred_ans[dont_ignore] == gt[dont_ignore]).float())
            change_detected = bin_pred_ans == bin_pred_gt
            bin_acc = 100.0 * torch.mean( change_detected[dont_ignore].float() )
            bin_true_positives = change_detected
            bin_true_positives[bin_pred_ans == 0] = 0

            bin_precision = 100.0 * torch.sum(bin_true_positives[dont_ignore].float()) / torch.sum(bin_pred_ans[dont_ignore].float())
            bin_recall = 100.0 * torch.sum(bin_true_positives[dont_ignore].float()) / torch.sum(bin_pred_gt[dont_ignore].float())
            bin_fscore = 2.0*(bin_precision*bin_recall) / (bin_precision+bin_recall)
            # Logic: Multi class accuracy is union of change detected (pred_ans > 0) and if change was correctly determined
            change_detected[bin_pred_gt == 0] = 0
            multi_acc = 100.0 * torch.mean( (pred_ans[change_detected] == gt[change_detected] ) .float() )

            test_accuracies.update(acc.item(), gt.shape[0])
            test_bin_accuracies.update(bin_acc.item(), gt.shape[0])
            if torch.sum(change_detected) > 0:
                test_multi_accuracies.update(multi_acc.item(), gt.shape[0])
            if torch.sum(bin_pred_ans.float()) > 0:
                test_bin_precision.update(bin_precision.item(), gt.shape[0]) 
            if torch.sum(bin_pred_gt.float()) > 0:
                test_bin_recall.update(bin_recall.item(), gt.shape[0])
            if not np.isnan(bin_fscore):
                test_bin_fscore.update(bin_fscore.item(), gt.shape[0])

            # ========================
            # For ROC Curve calculation
            # ========================
            #pdb.set_trace()
            # gt_all[batch_id, :] = bin_pred_gt[dont_ignore].view(bin_pred_gt.shape[0], bin_pred_gt.shape[1]*bin_pred_gt.shape[2]).cpu().numpy()
            # pos_class_all[batch_id, :] = roc_bin[dont_ignore].view(1, -1).detach().cpu().numpy()
            gt_all.extend( bin_pred_gt[dont_ignore].view(-1).cpu().numpy().tolist() )
            pos_class_all.extend( roc_bin[dont_ignore.squeeze()].view(-1).detach().cpu().numpy().tolist() )
            # ========================
            # Print and Plot
            # ========================
            print("TEST: Step: {}, Batch {}/{} has acc {:.5f}, multi acc {:.5f}, binary acc {:.5f}, bin_prec {:.5f}, bin_rec {:.5f}, bin_fscore {:.5f}".format(
                            current_step, batch_id, num_batches, test_accuracies.avg, test_multi_accuracies.avg, test_bin_accuracies.avg,
                            test_bin_precision.avg, test_bin_recall.avg, test_bin_fscore.avg))
            if current_step % self.test_display_freq == 0:
                im1 = im1[0,:,:,:].cpu()
                im2 = im2[0,:,:,:].cpu()
                name = '{0}_{1}_{2}'.format(epoch, current_step, "image")
                #pdb.set_trace()
                mask = labelVisualize(pred_ans[0,:,:].detach().cpu().numpy(),self.num_classes)
                gt_label = labelVisualize(gt[0,:,:].detach().cpu().numpy(),self.num_classes)
                #pdb.set_trace()
                combined = visualizeAllImages(im1,im2, transforms.ToTensor()(gt_label) , transforms.ToTensor()(mask))
                self.txwriter.add_image("Test/"+name,transforms.ToTensor()(combined))
        #pdb.set_trace()
        gt_all = np.asarray(gt_all, dtype=np.bool_)
        pos_class_all = np.asarray(pos_class_all, dtype=np.float32)
        fpr, tpr, thresholds = metrics.roc_curve(gt_all.ravel() , pos_class_all.ravel())
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        #plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.grid()
        #plt.show()
        name = './plots/ROC_Bin_Change_{0}.png'.format(epoch)
        plt.savefig(name)
        plt.clf()

        precision, recall, thresholds_pr = metrics.precision_recall_curve(gt_all.ravel() , pos_class_all.ravel())
        ap = metrics.average_precision_score(gt_all.ravel() , pos_class_all.ravel())
        plt.title('Precision-Recall Curve')
        plt.plot(recall, precision, 'b', label = 'AP = %0.2f' % ap)
        plt.legend(loc = 'lower right')
        #plt.plot([0, 1], [1, 0],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.grid()
        #plt.show()
        name = './plots/PR_Bin_Change_{0}.png'.format(epoch)
        plt.savefig(name)
        plt.clf()

        """
        Save Model periodically
        """
        #if ap >= self.save_thresh:
        self.save_thresh = ap
        save_name = 'unet_{0}_test_{1:d}.pth'.format(epoch,int(ap*100))
        torch.save(self.model.state_dict(), save_name)
        print('Saved model to {}'.format(save_name))

        return test_accuracies.avg , test_bin_accuracies.avg , test_multi_accuracies.avg, test_bin_precision.avg, test_bin_recall.avg, test_bin_fscore.avg

    def train(self):
        #torch.autograd.set_detect_anomaly(True)
        """
        Main training loop
        """
        for epoch in range(self.num_epochs):
            num_batches = len(self.train_dataset_loader)
            # Initialize running averages
            train_accuracies = AverageMeter()
            train_bin_accuracies = AverageMeter()
            train_multi_accuracies = AverageMeter()
            train_losses = AverageMeter()
            train_bin_precision = AverageMeter()
            train_bin_recall = AverageMeter()
            train_bin_fscore = AverageMeter()
            # ============
            # Adjust learning rate currently
            self._adjust_learning_rate(epoch)

            for batch_id, batch_data in enumerate(self.train_dataset_loader):
                self.model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id
                # Get data from dataset
                im1 = batch_data['im1'].cuda(async=True)
                im2 = batch_data['im2'].cuda(async=True)
                gt = batch_data['gt'].cuda(async=True)
                gt = gt.view(gt.shape[0],gt.shape[2],gt.shape[3])
                dont_ignore = gt != -1
                ignore = gt == -1
                gt[ignore] = 0

                # ============
                # Make prediction, backpropagation
                pred = self.model(im1, im2)
                loss = self._optimize(pred, gt)
                train_losses.update(loss.item(), gt.shape[0])
                #print(torch.argmax( F.softmax(pred, dim=1 ), dim=1 ).shape)
                #print(gt.shape)
                

                pred_ans = torch.argmax( F.softmax(pred, dim=1 ), dim=1 )
                pred_ans[ignore] = 0
                bin_pred_ans = pred_ans != torch.zeros_like(pred_ans).cuda(async=True)
                bin_pred_gt = gt != torch.zeros_like(pred_ans).cuda(async=True)
                
                acc = 100.0 * torch.mean( (pred_ans[dont_ignore] == gt[dont_ignore]).float())
                change_detected = bin_pred_ans == bin_pred_gt
                bin_acc = 100.0 * torch.mean( change_detected[dont_ignore].float() )
                bin_true_positives = change_detected
                bin_true_positives[bin_pred_ans == 0] = 0

                bin_precision = 100.0 * torch.sum(bin_true_positives[dont_ignore].float()) / torch.sum(bin_pred_ans[dont_ignore].float())
                bin_recall = 100.0 * torch.sum(bin_true_positives[dont_ignore].float()) / torch.sum(bin_pred_gt[dont_ignore].float())
                bin_fscore = 2.0*(bin_precision*bin_recall) / (bin_precision+bin_recall)
                # Logic: Multi class accuracy is union of change detected (pred_ans > 0) and if change was correctly determined
                change_detected[bin_pred_gt == 0] = 0
                multi_acc = 100.0 * torch.mean( (pred_ans[change_detected] == gt[change_detected] ) .float() )

                train_accuracies.update(acc.item(), gt.shape[0])
                train_bin_accuracies.update(bin_acc.item(), gt.shape[0])
                if torch.sum(change_detected) > 0:
                    train_multi_accuracies.update(multi_acc.item(), gt.shape[0])
                if torch.sum(bin_pred_ans.float()) > 0:
                    train_bin_precision.update(bin_precision.item(), gt.shape[0])
                if torch.sum(bin_pred_gt.float()) > 0:
                    train_bin_recall.update(bin_recall.item(), gt.shape[0])
                if not np.isnan(bin_fscore):
                    train_bin_fscore.update(bin_fscore.item(), gt.shape[0])
                                        
                    
                if current_step % self.log_freq == 0:
                    print("Step: {}, Epoch: {}, Batch {}/{}, LR {}, has loss {:.5f}, acc {:.5f}, multi acc {:.5f}, binary acc {:.5f}, bin_prec {:.5f}, bin_rec {:.5f}, bin_fscore {:.5f}".format(
                                current_step, epoch, batch_id, num_batches,self.lr, train_losses.avg, train_accuracies.avg, train_multi_accuracies.avg, 
                                train_bin_accuracies.avg, train_bin_precision.avg, train_bin_recall.avg, train_bin_fscore.avg))
                    self.txwriter.add_scalar('train/loss', train_losses.avg, current_step)
                    self.txwriter.add_scalar('train/accuracy', train_accuracies.avg, current_step)
                    self.txwriter.add_scalar('train/multi_class_accuracy', train_multi_accuracies.avg, current_step)
                    self.txwriter.add_scalar('train/binary_accuracy', train_bin_accuracies.avg, current_step)
                    self.txwriter.add_scalar('train/learning_rate', self.lr, current_step)
                    self.txwriter.add_scalar('train/bin_precision', train_bin_precision.avg, current_step)
                    self.txwriter.add_scalar('train/bin_recall', train_bin_recall.avg, current_step)
                    self.txwriter.add_scalar('train/bin_fscore', train_bin_fscore.avg, current_step)
                """
                Visualize some images
                """
                if current_step % self.display_freq == 0:
                    im1 = im1[0,:,:,:].cpu()
                    im2 = im2[0,:,:,:].cpu()
                    name = '{0}_{1}_{2}'.format(epoch, current_step, "image")
                    #pdb.set_trace()
                    mask = labelVisualize(pred_ans[0,:,:].detach().cpu().numpy(),self.num_classes)
                    gt_label = labelVisualize(gt[0,:,:].detach().cpu().numpy(),self.num_classes)
                    #pdb.set_trace()
                    combined = visualizeAllImages(im1,im2, transforms.ToTensor()(gt_label) , transforms.ToTensor()(mask))
                    self.txwriter.add_image("Train/"+name,transforms.ToTensor()(combined))

                
                # Test accuracies
                if current_step % self.test_freq == 0:
                    self.model.eval()
                    test_accuracy, test_bin_accuracy, test_multi_accuracy, test_bin_precision, test_bin_recall, test_bin_fscore = self.test(epoch)
                    print("Epoch: {} has val accuracy {:.5f}".format(epoch, test_accuracy))
                    self.txwriter.add_scalar('test/accuracy', test_accuracy, current_step)
                    self.txwriter.add_scalar('test/multi_class_accuracy', test_multi_accuracy, current_step)
                    self.txwriter.add_scalar('test/binary_accuracy', test_bin_accuracy, current_step)
                    self.txwriter.add_scalar('test/bin_precision', test_bin_precision, current_step)
                    self.txwriter.add_scalar('test/bin_recall', test_bin_recall, current_step)
                    self.txwriter.add_scalar('test/bin_fscore', test_bin_fscore, current_step)
                
    def test_thresh(self, pretrain_net, fpr_thresh):
        self.model.load_state_dict(torch.load(pretrain_net))
        num_batches = len(self.test_dataset_loader)
        test_accuracies = AverageMeter()
        test_bin_accuracies = AverageMeter()
        test_multi_accuracies = AverageMeter()
        test_bin_precision = AverageMeter()
        test_bin_recall = AverageMeter()
        test_bin_fscore = AverageMeter()
        gt_all = []
        pos_class_all = []
        #pos_class = [[]]
        pos_class=[[] for k in range(9)]
        gt_labels = []
        pos_class_labels = []
        # gt_all = np.zeros((429, 224*224),dtype=np.bool_)
        # pos_class_all = np.zeros((429, 224*224),dtype=np.float32)
        # assumes test batch size of 1
        for batch_id, batch_data in enumerate(self.test_dataset_loader):
            current_step = batch_id
            # Set to eval()
            self.model.eval()        
            # Get data from dataset
            im1 = batch_data['im1'].cuda(async=True)
            im2 = batch_data['im2'].cuda(async=True)
            gt = batch_data['gt'].cuda(async=True)
            gt = gt.view(gt.shape[0],gt.shape[2],gt.shape[3])
            dont_ignore = gt != -1
            ignore = gt == -1
            gt[ignore] = 0

            # ========================
            # Make prediction
            # ========================
            pred = self.model(im1, im2)
            binary_score_nochange = pred[:,0,:,:]
            (binary_score_change, _ ) = torch.max(pred[:,1:,:,:],dim=1)
            raw_scores = F.softmax(pred, dim=1 )
            
            pred_ans = torch.argmax( raw_scores, dim=1 )
            pred_ans[ignore] = 0
            bin_pred_ans = pred_ans != torch.zeros_like(pred_ans).cuda(async=True)
            bin_pred_gt = gt != torch.zeros_like(pred_ans).cuda(async=True)

            
            roc_bin = torch.cat( (binary_score_nochange , binary_score_change),dim=0)
            roc_bin = F.softmax(roc_bin, dim=0 )
            #pdb.set_trace()
            roc_bin = roc_bin[1,:,:]
            

            acc = 100.0 * torch.mean( (pred_ans[dont_ignore] == gt[dont_ignore]).float())
            change_detected = bin_pred_ans == bin_pred_gt
            bin_acc = 100.0 * torch.mean( change_detected[dont_ignore].float() )
            bin_true_positives = change_detected
            bin_true_positives[bin_pred_ans == 0] = 0

            bin_precision = 100.0 * torch.sum(bin_true_positives[dont_ignore].float()) / torch.sum(bin_pred_ans[dont_ignore].float())
            bin_recall = 100.0 * torch.sum(bin_true_positives[dont_ignore].float()) / torch.sum(bin_pred_gt[dont_ignore].float())
            bin_fscore = 2.0*(bin_precision*bin_recall) / (bin_precision+bin_recall)
            # Logic: Multi class accuracy is union of change detected (pred_ans > 0) and if change was correctly determined
            change_detected[bin_pred_gt == 0] = 0
            multi_acc = 100.0 * torch.mean( (pred_ans[change_detected] == gt[change_detected] ) .float() )

            test_accuracies.update(acc.item(), gt.shape[0])
            test_bin_accuracies.update(bin_acc.item(), gt.shape[0])
            if torch.sum(change_detected) > 0:
                test_multi_accuracies.update(multi_acc.item(), gt.shape[0])
            if torch.sum(bin_pred_ans.float()) > 0:
                test_bin_precision.update(bin_precision.item(), gt.shape[0]) 
            if torch.sum(bin_pred_gt.float()) > 0:
                test_bin_recall.update(bin_recall.item(), gt.shape[0])
            if not np.isnan(bin_fscore):
                test_bin_fscore.update(bin_fscore.item(), gt.shape[0])

            # ========================
            # For ROC Curve calculation
            # ========================
            #pdb.set_trace()
            # gt_all[batch_id, :] = bin_pred_gt.view(bin_pred_gt.shape[0], bin_pred_gt.shape[1]*bin_pred_gt.shape[2]).cpu().numpy()
            # pos_class_all[batch_id, :] = roc_bin.view(1, -1).detach().cpu().numpy()
            gt_all.extend( bin_pred_gt[dont_ignore].view(-1).cpu().numpy().tolist() )
            pos_class_all.extend( roc_bin[dont_ignore.squeeze()].view(-1).detach().cpu().numpy().tolist() )
            for i in range(0,raw_scores.shape[1]-1):
                rs = raw_scores[:,i+1,:,:]
                pos_class[i].extend( rs[dont_ignore].detach().cpu().numpy().tolist() )
            gt_labels.extend( gt[dont_ignore].view(-1).cpu().numpy().tolist() )
                #pos_class_labels.extend( pred_ans[dont_ignore].view(-1).detach().cpu().numpy().tolist() )
            # ========================
            # Print and Plot
            # ========================
            print("TEST: Step: {}, Batch {}/{} has acc {:.5f}, multi acc {:.5f}, binary acc {:.5f}, bin_prec {:.5f}, bin_rec {:.5f}, bin_fscore {:.5f}".format(
                            current_step, batch_id, num_batches, test_accuracies.avg, test_multi_accuracies.avg, test_bin_accuracies.avg,
                            test_bin_precision.avg, test_bin_recall.avg, test_bin_fscore.avg))
            # if current_step % self.test_display_freq == 0:
            #     im1 = im1[0,:,:,:].cpu()
            #     im2 = im2[0,:,:,:].cpu()
            #     name = '{0}_{1}_{2}'.format(epoch, current_step, "image")
            #     #pdb.set_trace()
            #     mask = labelVisualize(pred_ans[0,:,:].detach().cpu().numpy(),self.num_classes)
            #     gt_label = labelVisualize(gt[0,:,:].detach().cpu().numpy(),self.num_classes)
            #     #pdb.set_trace()
            #     combined = visualizeAllImages(im1,im2, transforms.ToTensor()(gt_label) , transforms.ToTensor()(mask))
            #     self.txwriter.add_image("Test/"+name,transforms.ToTensor()(combined))
        #pdb.set_trace()
        gt_all = np.asarray(gt_all, dtype=np.bool_)
        pos_class_all = np.asarray(pos_class_all, dtype=np.float32)
        fpr, tpr, thresholds = metrics.roc_curve(gt_all.ravel() , pos_class_all.ravel())
        fpr, tpr, thresholds = metrics.roc_curve(gt_all.ravel() , pos_class_all.ravel())
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'c')#, label = 'AUC = %0.2f' % roc_auc)
        #plt.legend(loc = 'lower right')
        #plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        #plt.grid()
        #plt.show()
        name = './plots/final_ROC_Bin_Change.png'
        plt.savefig(name)
        plt.clf()

        for i in range(0,10-1):
            #indx = gt_labels == i or gt_labels == 0 or
            gt_class = gt_labels
            gt_class[gt_labels != 0 or gt_labels != i] = 0
            gt_class = np.asarray(gt_class, dtype=np.bool_)
            pos_class_cls = np.asarray( pos_class[i], dtype=np.float32)
            #pdb.set_trace()
            precision, recall, thresholds_pr = metrics.precision_recall_curve(gt_class.ravel() , pos_class_cls.ravel())
            ap = metrics.average_precision_score(gt_class.ravel() , pos_class_cls.ravel())
            plt.title('Precision-Recall Curve')
            plt.plot(recall, precision, label = CHANGE_CLASSES[i+1], linewidth = 2.0)
            #plt.plot([0, 1], [1, 0],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('Precision')
            plt.xlabel('Recall')
            print('Plotting {0:}'.format(i))
            #plt.grid()
        plt.grid()
        plt.legend(loc = 'upper right')
        plt.show()
        name = './plots/final_PR_Multi_Bin_Change.png'
        #plt.savefig(name)
        plt.clf()

        precision, recall, thresholds_pr = metrics.precision_recall_curve(gt_all.ravel() , pos_class_all.ravel())
        ap = metrics.average_precision_score(gt_all.ravel() , pos_class_all.ravel())
        plt.title('Precision-Recall Curve')
        plt.plot(recall, precision, 'c')#, label = 'AP = %0.2f' % ap)
        #plt.legend(loc = 'lower right')
        #plt.plot([0, 1], [1, 0],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        #plt.grid()
        #plt.show()
        name = './plots/final_PR_Bin_Change.png'
        plt.savefig(name)
        plt.clf()

        
        fpr_index = np.where(fpr >= fpr_thresh)
        fpr_index = fpr_index[0][0]
        fpr_recall = tpr[fpr_index]
        #pdb.set_trace()
        prec_index = np.where(recall <= fpr_recall)
        prec_index = prec_index[0][0]
        fpr_precision = precision[prec_index]
        thresh = thresholds[fpr_index]
        
        print('FPR Threshold {0:d}: Pr: {1:.5f} , Re: {2:.5f}'.format(int(100*fpr_thresh), fpr_precision, fpr_recall) )
        

        for batch_id, batch_data in enumerate(self.test_dataset_loader):
            current_step = batch_id
            # Set to eval()
            self.model.eval()        
            # Get data from dataset
            im1 = batch_data['im1'].cuda(async=True)
            im2 = batch_data['im2'].cuda(async=True)
            gt = batch_data['gt'].cuda(async=True)
            gt = gt.view(gt.shape[0],gt.shape[2],gt.shape[3])
            dont_ignore = gt != -1
            ignore = gt == -1
            #gt[ignore] = 0

            # ========================
            # Make prediction
            # ========================
            pred = self.model(im1, im2)
            binary_score_nochange = pred[:,0,:,:]
            (binary_score_change, _ ) = torch.max(pred[:,1:,:,:],dim=1)
            raw_scores = F.softmax(pred, dim=1 )
            
            pred_ans = torch.argmax( raw_scores, dim=1 )
            pred_ans[ignore] = 0
            bin_pred_ans = pred_ans != torch.zeros_like(pred_ans).cuda(async=True)
            bin_pred_gt = gt != torch.zeros_like(pred_ans).cuda(async=True)

            
            roc_bin = torch.cat( (binary_score_nochange , binary_score_change),dim=0)
            roc_bin = F.softmax(roc_bin, dim=0 )
            #pdb.set_trace()
            roc_bin = roc_bin[1,:,:]
            

            acc = 100.0 * torch.mean( (pred_ans[dont_ignore] == gt[dont_ignore]).float())
            change_detected = bin_pred_ans == bin_pred_gt
            bin_acc = 100.0 * torch.mean( change_detected[dont_ignore].float() )
            bin_true_positives = change_detected
            bin_true_positives[bin_pred_ans == 0] = 0

            bin_precision = 100.0 * torch.sum(bin_true_positives[dont_ignore].float()) / torch.sum(bin_pred_ans[dont_ignore].float())
            bin_recall = 100.0 * torch.sum(bin_true_positives[dont_ignore].float()) / torch.sum(bin_pred_gt[dont_ignore].float())
            bin_fscore = 2.0*(bin_precision*bin_recall) / (bin_precision+bin_recall)
            # Logic: Multi class accuracy is union of change detected (pred_ans > 0) and if change was correctly determined
            change_detected[bin_pred_gt == 0] = 0
            multi_acc = 100.0 * torch.mean( (pred_ans[change_detected] == gt[change_detected] ) .float() )

            test_accuracies.update(acc.item(), gt.shape[0])
            test_bin_accuracies.update(bin_acc.item(), gt.shape[0])
            if torch.sum(change_detected) > 0:
                test_multi_accuracies.update(multi_acc.item(), gt.shape[0])
            if torch.sum(bin_pred_ans.float()) > 0:
                test_bin_precision.update(bin_precision.item(), gt.shape[0]) 
            if torch.sum(bin_pred_gt.float()) > 0:
                test_bin_recall.update(bin_recall.item(), gt.shape[0])
            if not np.isnan(bin_fscore):
                test_bin_fscore.update(bin_fscore.item(), gt.shape[0])

            # ========================
            # For ROC Curve calculation
            # ========================
            
            # ========================
            # Print and Plot
            # ========================
            print("TEST: Step: {}, Batch {}/{} has acc {:.5f}, multi acc {:.5f}, binary acc {:.5f}, bin_prec {:.5f}, bin_rec {:.5f}, bin_fscore {:.5f}".format(
                            current_step, batch_id, num_batches, test_accuracies.avg, test_multi_accuracies.avg, test_bin_accuracies.avg,
                            test_bin_precision.avg, test_bin_recall.avg, test_bin_fscore.avg))
            
            im1 = im1[0,:,:,:].cpu()
            im2 = im2[0,:,:,:].cpu()
            mask_dont_ignore = dont_ignore.detach().cpu().long()
            # mask_dont_ignore = dont_ignore.view(1, im1.shape[1], im1.shape[2]).detach().cpu().float().numpy()
            mask = pred_ans[0,:,:].detach().cpu() * mask_dont_ignore
            #pdb.set_trace()
            name1 = './test_output/{0}_{1:d}_{2}.png'.format(current_step,int(fpr_thresh*100), "image1")
            name2 = './test_output/{0}_{1:d}_{2}.png'.format(current_step,int(fpr_thresh*100), "image2")
            name3 = './test_output/{0}_{1:d}_{2}.png'.format(current_step, int(fpr_thresh*100), "masked")
            mask = labelVisualize(mask.view(im1.shape[1],im1.shape[2]).numpy(),self.num_classes)
            gt_label = labelVisualize(gt[0,:,:].detach().cpu().numpy(),self.num_classes)
            #mask_dont_ignore = dont_ignore[0,:,:,:].detach().cpu().float().numpy()
            #pdb.set_trace()
            combined = visualizeMaskedImages(im1,im2, transforms.ToTensor()(gt_label) , transforms.ToTensor()(mask))
            io.imsave(name1,np.uint8(combined))
            #self.txwriter.add_image("Test/"+name,transforms.ToTensor()(combined))


        return 0
   
                   
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def denormalizeImage(image):
    #mean=np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    #std=np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    image = image
    return image.astype(np.float32)

if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Train Change Detection UNet')
    parser.add_argument('--model_arch', type=str, default='vgg16')
    parser.add_argument('--train_dataset_path', type=str, default='./train/train.txt')
    parser.add_argument('--test_dataset_path', type=str, default='./test/test.txt')
    parser.add_argument('--train_batch_size', type=int, default=10)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_data_loader_workers', type=int, default=10)
    parser.add_argument('--model_save_dir', type=str, default='./models')
    parser.add_argument('--pretrained_encoder', type=str, default='./unet_encoder_checkpoint.pth')
    parser.add_argument('--pretrained_decoder', type=str, default='./unet_decoder_checkpoint.pth')



    args = parser.parse_args()

    # Create experiment runner object
    # Loads data, creates models
    experiment_runner = ExperimentRunner( model_arch=args.model_arch,
                                          train_dataset_path=args.train_dataset_path,
                                          test_dataset_path=args.test_dataset_path, 
                                          train_batch_size=args.train_batch_size,
                                          test_batch_size=args.test_batch_size, 
                                          model_save_dir=args.model_save_dir,
                                          num_epochs=args.num_epochs, 
                                          num_data_loader_workers=args.num_data_loader_workers,
                                          num_classes=10,
                                          encoder_path=args.pretrained_encoder,
                                          decoder_path=args.pretrained_decoder)
    # Train Models
    good_models = ['./pytorch_models/unet_132_test_77.pth', './pytorch_models/unet_186_test_77.pth']
    experiment_runner.test_thresh(good_models[1], 0.1)
    #experiment_runner.test_thresh(good_models[0], 0.01)
    #experiment_runner.train()
