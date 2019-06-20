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

from networks import *
#import utils
from tensorboardX import SummaryWriter
import os


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
        else:
            self.model = ResNetUNet()
        # self.model.encoder.load_state_dict(torch.load(encoder_path))
        # self.model.decoder1.load_state_dict(torch.load(decoder_path))
        # self.model.decoder2.load_state_dict(torch.load(decoder_path))       

        # for param in self.model.encoder.parameters():
        #     param.requires_grad = False

        # for param in self.model.encoder.convblock1.parameters():
        #     param.requires_grad = False
        # for param in self.model.encoder.convblock2.parameters():
        #     param.requires_grad = False
        # for param in self.model.encoder.convblock3.parameters():
        #     param.requires_grad = False
        # for param in self.model.encoder.convblock4.parameters():
        #     param.requires_grad = False
        # for param in self.model.encoder.convblock5.parameters():
        #     param.requires_grad = False
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
        self.test_freq = 1000  # Steps
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        # Create datasets
        self.train_dataset = VLCMUCDDataset(train_dataset_path, training=True)
        self.test_dataset = VLCMUCDDataset(test_dataset_path, training=False)

        self.train_dataset_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=num_data_loader_workers)
        self.test_dataset_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=True, num_workers=self.test_batch_size)

        # Use the GPU if it's available.
        self.cuda = torch.cuda.is_available()

        if self.cuda:
            self.model.cuda()

        # Tensorboard logger
        self.txwriter = SummaryWriter()
        self.model_save_dir = model_save_dir
        self.save_freq = 5000
        self.display_freq = 500
        self.test_display_freq = 300

        self.lr_decay_frequency = 20
        self.lr_decay_rate = 0.5



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
        return
        # if (epoch % self.lr_decay_frequency) == self.lr_decay_frequency-1:
        #     #self.lr *= (self.lr_decay_rate ** (epoch // self.lr_decay_frequency))
        #     self.lr *= self.lr_decay_rate
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = self.lr
        # return

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

        for batch_id, batch_data in enumerate(self.test_dataset_loader):
            current_step = batch_id
            # Set to eval()
            self.model.eval()        
            # Get data from dataset
            im1 = batch_data['im1'].cuda(async=True)
            im2 = batch_data['im2'].cuda(async=True)
            gt = batch_data['gt'].cuda(async=True)
            gt = gt.view(gt.shape[0],gt.shape[2],gt.shape[3])

            # ============
            # Make prediction
            pred = self.model(im1, im2)
            pred_ans = torch.argmax( F.softmax(pred, dim=1 ), dim=1 )
            bin_pred_ans = pred_ans != torch.zeros_like(pred_ans).cuda(async=True)
            bin_pred_gt = gt != torch.zeros_like(pred_ans).cuda(async=True)

            acc = 100.0 * torch.mean( (pred_ans == gt).float())
            change_detected = bin_pred_ans == bin_pred_gt
            bin_acc = 100.0 * torch.mean( change_detected.float() )
            bin_true_positives = change_detected
            bin_true_positives[bin_pred_ans == 0] = 0

            bin_precision = 100.0 * torch.sum(bin_true_positives.float()) / torch.sum(bin_pred_ans.float())
            bin_recall = 100.0 * torch.sum(bin_true_positives.float()) / torch.sum(bin_pred_gt.float())
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

                
            # ============
            # Print and Plot
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
                bin_pred_ans = pred_ans != torch.zeros_like(pred_ans).cuda(async=True)
                bin_pred_gt = gt != torch.zeros_like(pred_ans).cuda(async=True)
                
                acc = 100.0 * torch.mean( (pred_ans == gt).float())
                change_detected = bin_pred_ans == bin_pred_gt
                bin_acc = 100.0 * torch.mean( change_detected.float() )
                bin_true_positives = change_detected
                bin_true_positives[bin_pred_ans == 0] = 0

                bin_precision = 100.0 * torch.sum(bin_true_positives.float()) / torch.sum(bin_pred_ans.float())
                bin_recall = 100.0 * torch.sum(bin_true_positives.float()) / torch.sum(bin_pred_gt.float())
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
                
                """
                Save Model periodically
                """
                if (current_step % self.save_freq == 0) and current_step > 0:
                    save_name = 'resnet_unet.pth'
                    torch.save(self.model.state_dict(), save_name)
                    print('Saved model to {}'.format(save_name))
                   
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
    parser.add_argument('--model_arch', type=str, default='resnet')
    parser.add_argument('--train_dataset_path', type=str, default='./train/train.txt')
    parser.add_argument('--test_dataset_path', type=str, default='./test/test.txt')
    parser.add_argument('--train_batch_size', type=int, default=10)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=250)
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
    experiment_runner.train()
