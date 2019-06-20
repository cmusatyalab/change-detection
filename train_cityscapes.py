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
from cityscapes_pretrain_dataset import *
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
    def __init__(self, train_dataset_path, test_dataset_path, train_batch_size, test_batch_size, model_save_dir, num_epochs=100, 
                    num_data_loader_workers=10, num_classes=10, image_size=(512,1024)):
        # GAN Network + VGG Loss Network
        self.model = UNetSeg()
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
        # Network losses
        self.criterion = nn.CrossEntropyLoss(reduction='none').cuda()

        # Train settings + log settings
        self.num_epochs = num_epochs
        self.log_freq = 10  # Steps
        self.test_freq = 1000  # Steps
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.image_size = image_size

        # Create datasets
        self.train_dataset = CityscapesSegmentation(training=True, image_size=self.image_size)
        self.test_dataset = CityscapesSegmentation(split="val",training=False, image_size=self.image_size)

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
        self.edge_lambda = 1.


        #self.model = nn.DataParallel(self.model)

    def _optimize(self, y_pred, y_gt, y_edges):
        """
        VGGLoss + GAN loss
        """
        self.optimizer.zero_grad()
        loss = self.criterion(y_pred, y_gt)
        #pdb.set_trace()
        loss += loss * self.edge_lambda * y_edges
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def _adjust_learning_rate(self, epoch):
        """
        TODO
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= divide_lr
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
        for batch_id, batch_data in enumerate(self.test_dataset_loader):
            current_step = batch_id
            # Set to eval()
            self.model.eval()        
            # Get data from dataset
            im = batch_data['im'].cuda(async=True)
            gt = batch_data['gt'].cuda(async=True)
            gt = gt.view(gt.shape[0],gt.shape[2],gt.shape[3])

            # ============
            # Make prediction
            pred = self.model(im)
            pred_ans = torch.argmax( F.softmax(pred, dim=1 ), dim=1 )
            bin_pred_ans = pred_ans != torch.zeros_like(pred_ans).cuda(async=True)
            bin_pred_gt = gt != torch.zeros_like(pred_ans).cuda(async=True)

            acc = 100.0 * torch.mean( (pred_ans == gt).float())
            change_detected = bin_pred_ans == bin_pred_gt
            bin_acc = 100.0 * torch.mean( change_detected.float() )
            # Logic: Multi class accuracy is union of change detected (pred_ans > 0) and if change was correctly determined
            change_detected[bin_pred_ans == 0] = 0
            multi_acc = 100.0 * torch.mean( (pred_ans[change_detected] == gt[change_detected] ) .float() )

            test_accuracies.update(acc.item(), gt.shape[0])
            test_bin_accuracies.update(bin_acc.item(), gt.shape[0])
            if torch.sum(change_detected) > 0:
                test_multi_accuracies.update(multi_acc.item(), gt.shape[0])
            # ============
            # Print and Plot
            print("TEST: Step: {}, Batch {}/{} has acc {}, multi acc {}, and binary acc {}".format(
                            current_step, batch_id, num_batches, test_accuracies.avg, test_multi_accuracies.avg, test_bin_accuracies.avg))
            if current_step % self.test_display_freq == 0:
                im = im[0,:,:,:].cpu()
                name = '{0}_{1}_{2}'.format(epoch, current_step, "image")
                #pdb.set_trace()
                mask = labelVisualize(pred_ans[0,:,:].detach().cpu().numpy(),self.num_classes, self.image_size)
                gt_label = labelVisualize(gt[0,:,:].detach().cpu().numpy(),self.num_classes, self.image_size)
                #pdb.set_trace()
                combined = visualizeAllCityImages(im, transforms.ToTensor()(gt_label) , transforms.ToTensor()(mask))
                self.txwriter.add_image("Test/"+name,transforms.ToTensor()(combined))
        return test_accuracies.avg , test_bin_accuracies.avg , test_multi_accuracies.avg

    def train(self):
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

            for batch_id, batch_data in enumerate(self.train_dataset_loader):
                self.model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id
                # Get data from dataset
                im = batch_data['im'].cuda(async=True)
                gt = batch_data['gt'].cuda(async=True)
                edges = batch_data['edges'].cuda(async=True)
                gt = gt.view(gt.shape[0],gt.shape[2],gt.shape[3])

                # ============
                # Make prediction
                pred = self.model(im)
                loss = self._optimize(pred, gt, edges)
                train_losses.update(loss.item(), gt.shape[0])
                #print(torch.argmax( F.softmax(pred, dim=1 ), dim=1 ).shape)
                #print(gt.shape)
                pred_ans = torch.argmax( F.softmax(pred, dim=1 ), dim=1 )
                bin_pred_ans = pred_ans != torch.zeros_like(pred_ans).cuda(async=True)
                bin_pred_gt = gt != torch.zeros_like(pred_ans).cuda(async=True)

                acc = 100.0 * torch.mean( (pred_ans == gt).float())
                change_detected = bin_pred_ans == bin_pred_gt
                bin_acc = 100.0 * torch.mean( change_detected.float() )
                # Logic: Multi class accuracy is union of change detected (pred_ans > 0) and if change was correctly determined
                change_detected[bin_pred_ans == 0] = 0
                #pdb.set_trace()
                multi_acc = 100.0 * torch.mean( (pred_ans[change_detected] == gt[change_detected] ) .float() )

                train_accuracies.update(acc.item(), gt.shape[0])
                train_bin_accuracies.update(bin_acc.item(), gt.shape[0])
                if torch.sum(change_detected) > 0:
                    train_multi_accuracies.update(multi_acc.item(), gt.shape[0])
                # ============
                # Not adjusting learning rate currently
                # if epoch % 100 == 99:
                #     self._adjust_learning_rate(epoch)
                # # Not Clipping Weights
                # self._clip_weights()

                if current_step % self.log_freq == 0:
                    print("Step: {}, Epoch: {}, Batch {}/{} has loss {}, acc {}, multi acc {}, and binary acc {}".format(
                                current_step, epoch, batch_id, num_batches, train_losses.avg, train_accuracies.avg, train_multi_accuracies.avg, train_bin_accuracies.avg))
                    self.txwriter.add_scalar('train/loss', train_losses.avg, current_step)
                    self.txwriter.add_scalar('train/accuracy', train_accuracies.avg, current_step)
                    self.txwriter.add_scalar('train/multi_class_accuracy', train_multi_accuracies.avg, current_step)
                    self.txwriter.add_scalar('train/binary_accuracy', train_bin_accuracies.avg, current_step)
                """
                Visualize some images
                """
                if current_step % self.display_freq == 0:
                    im = im[0,:,:,:].cpu()
                    name = '{0}_{1}_{2}'.format(epoch, current_step, "image")
                    #pdb.set_trace()
                    mask = labelVisualize(pred_ans[0,:,:].detach().cpu().numpy(),self.num_classes, self.image_size)
                    gt_label = labelVisualize(gt[0,:,:].detach().cpu().numpy(),self.num_classes, self.image_size)
                    #pdb.set_trace()
                    combined = visualizeAllCityImages(im, transforms.ToTensor()(gt_label) , transforms.ToTensor()(mask))
                    self.txwriter.add_image("Train/"+name,transforms.ToTensor()(combined))

                
                # Test accuracies
                if current_step % self.test_freq == 0:
                    self.model.eval()
                    test_accuracy, test_bin_accuracy, test_multi_accuracy = self.test(epoch)
                    print("Epoch: {} has val accuracy {}".format(epoch, test_accuracy))
                    self.txwriter.add_scalar('test/accuracy', test_accuracy, current_step)
                    self.txwriter.add_scalar('test/multi_class_accuracy', test_multi_accuracy, current_step)
                    self.txwriter.add_scalar('test/binary_accuracy', test_bin_accuracy, current_step)
                
                """
                Save Model periodically
                """
                if (current_step % self.save_freq == 0) and current_step > 0:
                    save_name1 = 'unet_encoder_checkpoint.pth'
                    save_name2 = 'unet_decoder_checkpoint.pth'
                    torch.save(self.model.encoder.state_dict(), save_name1)
                    torch.save(self.model.decoder.state_dict(), save_name2)
                    #torch.save(self.model.state_dict(), save_name)
                    print('Saved model to {}'.format(save_name1))
                    print('Saved model to {}'.format(save_name2))
                   
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
    parser = argparse.ArgumentParser(description='Train a Segmentation Network.')
    parser.add_argument('--train_dataset_path', type=str, default='./train/train.txt')
    parser.add_argument('--test_dataset_path', type=str, default='./test/test.txt')
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=70)
    parser.add_argument('--num_data_loader_workers', type=int, default=1)
    parser.add_argument('--model_save_dir', type=str, default='./models')
    parser.add_argument('--pretrained_unet', type=str, default='./unet.pth')


    args = parser.parse_args()

    # Create experiment runner object
    # Loads data, creates models
    experiment_runner = ExperimentRunner( train_dataset_path=args.train_dataset_path,
                                          test_dataset_path=args.test_dataset_path, 
                                          train_batch_size=args.train_batch_size,
                                          test_batch_size=args.test_batch_size, 
                                          model_save_dir=args.model_save_dir,
                                          num_epochs=args.num_epochs, 
                                          num_data_loader_workers=args.num_data_loader_workers,
                                          num_classes=10)
    # Train Models
    experiment_runner.train()
