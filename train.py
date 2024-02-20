import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
import matplotlib.pyplot as plt
import os
# Data parameters
data_folder = 'C:/Image-to-Caption-100/Flickr8k_preprocessed'  # folder with data files saved by create_input_files.py
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'  # base name shared by data files
plot_dir = 'C:/Image-to-Caption-100/plots'  # folder where plots are saved
# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 256  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
# testing git commit

# Training parameters
start_epoch = 0
epochs = 200  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 16
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-3  # learning rate for encoder if fine-tuning
decoder_lr = 4e-3  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_meteor = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none

train_losses = []  # List to store training losses
validation_metrics = []  # List to store validation metrics (METEOR)


def main():
    """
    Training and validation.
    """
    global best_meteor, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, train_losses, validation_metrics
    # checkpoint = 'D:/Image-to-Caption-Final/checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar' 
    
    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Create reverse word map 
    word_map_rev = {v: k for k, v in word_map.items()}
    # Initialize / load checkpoint

    if checkpoint is None:
        print("----------------This is if block ---------------------")
        decoder = DecoderWithAttention(attention_dim=attention_dim,
embed_dim=emb_dim,
decoder_dim=decoder_dim,
vocab_size=len(word_map),
dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
lr=encoder_lr) if fine_tune_encoder else None

    else:

        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_meteor = checkpoint['meteor']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
lr=encoder_lr)
            


    # Move to GPU, if available
    decoder = decoder.to(device)
    print("Epochs since last improvement: %d\n" % (epochs_since_improvement,))
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


    # Epochs
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        # if epochs_since_improvement == 20:
        #     break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)


        # One epoch's training
        epoch_train_loss =  train(train_loader=train_loader,
encoder=encoder,
decoder=decoder,
criterion=criterion,
encoder_optimizer=encoder_optimizer,
decoder_optimizer=decoder_optimizer,
epoch=epoch)
        
        train_losses.append(epoch_train_loss)

        # One epoch's validation
        recent_meteor = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                word_map_rev=word_map_rev)
        
        validation_metrics.append(recent_meteor)

        # Check if there was an improvement
        
        is_best = recent_meteor > best_meteor
        best_meteor = max(recent_meteor, best_meteor)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_meteor, is_best)
        
        # Plot progress at the end of each epoch
        plot_progress(train_losses, validation_metrics, epoch)



def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()
 
    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
  
        data_time.update(time.time() - start)

   
        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores= pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
batch_time=batch_time,
data_time=data_time, loss=losses,
top5=top5accs))
            
    
    epoch_loss = losses.avg
        
    return epoch_loss

    


# After each batch, you can plot the loss and METEOR progress
def plot_progress(train_losses, validation_metrics, epoch):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, validation_metrics, label='Validation METEOR')
    plt.title('Validation METEOR')
    plt.xlabel('Epoch')
    plt.ylabel('METEOR Score')

    plt.tight_layout()

    # Save the plot instead of showing it
    plot_filename = os.path.join(plot_dir, f'plot_epoch_{epoch}.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the figure to free memory

def validate(val_loader, encoder, decoder, criterion, word_map_rev):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
       
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            sort_ind = sort_ind.to(device)
            sort_ind = sort_ind.to(allcaps.device)
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                # img_captions = list(
                #     map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                #         img_caps))  # remove <start> and pads
                
                img_captions = list(
                                map(lambda c: [word_map_rev[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                                img_caps))  # remove <start>, <end> and pads and keep as list of words

                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
            #     temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            # preds = temp_preds

                temp_preds.append([word_map_rev[w] for w in preds[j][:decode_lengths[j]] if w not in {word_map['<end>'], word_map['<pad>']}])  # remove pads, <end> and keep as list of words
            preds = temp_preds   

            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)
            

        # Calculate BLEU-4 scores
        # bleu4 = corpus_bleu(references, hypotheses)
        
        
        # Calculate METEOR scores
        meteor_scores = [meteor_score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
        meteor = sum(meteor_scores) / len(meteor_scores)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, METEOR - {meteor}\n'.format(
                loss=losses,
                top5=top5accs,
                meteor=meteor))

    return meteor


if __name__ == '__main__':
    main()
