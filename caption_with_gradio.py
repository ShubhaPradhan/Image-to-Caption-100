import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from PIL import Image
import imageio

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OUTPUT DIRECTORY TO STORE THE ATTENTION MECHANISM IMAGES
output_dir = 'C:/Image-to-Caption-100/uploaded_images_with_attention'

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imageio.v2.imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = np.resize(img, (256, 256, 3))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = (top_k_words // vocab_size).long()  # (s)
        next_word_inds = (top_k_words % vocab_size).long()  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds.long()], alpha[prev_word_inds.long()].unsqueeze(1)],
dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    print("shubha")

    # DEBUGGIN CODE: SHUBHA
    alphas_np = np.array(alphas)
    print(alphas_np.shape)

    return seq, alphas

# THE ATTENTION IMAGE FOR EACH WORD IS BEING GENERATED AS A DIFFERENT IMAGE FILE HAVE TO CHANGE THIS TO A SINGLE IMAGE FILE

# def visualize_att(image_path, seq, alphas, rev_word_map, output_dir, smooth=True):
#     """
#     Visualizes caption with weights at every word.
#     Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
#     :param image_path: path to image that has been captioned
#     :param seq: caption
#     :param alphas: weights
#     :param rev_word_map: reverse word mapping, i.e. ix2word
#     :param smooth: smooth weights?
#     """
#     image = Image.open(image_path)
#     image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
#      # Create the output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     words = [rev_word_map[ind] for ind in seq]

#     for t in range(len(words)):
#         if t > 50:
#             break
#         plt.subplot(int(np.ceil(len(words) / 5.)), 5, t + 1)
#         plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
#         plt.imshow(image)
#         current_alpha = alphas[t, :]
#         if smooth:
#             alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
#         else:
#             alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
#         if t == 0:
#             plt.imshow(alpha, alpha=0)
#         else:
#             plt.imshow(alpha, alpha=0.8)
#         plt.set_cmap(cm.Greys_r)
#         plt.axis('off')
#         # Save the attention images to the output directory
#         plt.savefig(os.path.join(output_dir, f'attention_{t}.png'))
#         plt.clf()

def visualize_att(image_path, seq, alphas, rev_word_map, output_file, smooth=True):
    """
    Visualizes caption with weights at every word and saves the images in a single file.
    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param output_file: filename for the combined image
    :param smooth: smooth weights?
    """

    # Get image name
    image_name = os.path.basename(image_path)

    # remove the file extension
    image_name = os.path.splitext(image_name)[0]
    
    # Open the image
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    num_words = len(words)
    num_rows = (num_words + 3) // 4  # Calculate the number of rows needed

    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(16, num_rows * 4))

    for t in range(num_words):
        row = t // 4  # Calculate the row index
        col = t % 4    # Calculate the column index

        axes[row, col].text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        axes[row, col].imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            axes[row, col].imshow(alpha, alpha=0)
        else:
            axes[row, col].imshow(alpha, alpha=0.8)
        axes[row, col].set_axis_off()

    
    # Remove empty subplots if there are any
    for t in range(num_words, num_rows * 4):
        row = t // 4
        col = t % 4
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{image_name}_with_attention.png'))

import os
import datetime
import gradio as gr

def generate_caption_with_gradio(image):
    # Generate a unique filename for the uploaded image
    image_filename = 'uploaded_image_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.jpg'
    image_path = os.path.join('C:/Image-to-Caption-100/uploaded_images', image_filename)
    
    # Save the uploaded image
    image.save(image_path)

    # Load the model
    checkpoint = torch.load('C:/Image-to-Caption-100/BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar', map_location='cpu')

    # Load the word ma
    with open('C:/Image-to-Caption-100/Flickr8k_preprocessed/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json', 'r') as j:
        word_map = json.load(j)

    # Construct the model
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']

    # Move the model to GPU if available

    # Move the model to CPU
    encoder = encoder.to('cpu')
    decoder = decoder.to('cpu')
    
    
    # Call the existing function to generate the caption
    caption, alphas = caption_image_beam_search(encoder, decoder, image_path, word_map, 5)  # Assuming beam size is 5
    alphas = torch.FloatTensor(alphas)
    
    # Convert the sequence of token IDs to words
    rev_word_map = {v: k for k, v in word_map.items()}  # Reverse word map
    caption_words = [rev_word_map.get(token, '<unk>') for token in caption]
    sentence = ' '.join(caption_words)

    # Visualize attention
    visualize_att(image_path, caption, alphas, rev_word_map, output_dir)

    # static graph image file path to show the plots
    static_graph_path = 'D:/Image-to-Caption-Final/plots/plot_epoch_540.png'

    # Get the attention image file path to show the attention images in gradio
    # Get image name
    image_name = os.path.basename(image_filename)

    # remove the file extension
    image_name = os.path.splitext(image_name)[0]
    attention_image_path = os.path.join(output_dir, f'{image_name}_with_attention.png')

    return caption, sentence, static_graph_path, attention_image_path

iface = gr.Interface(
    fn=generate_caption_with_gradio, 
    inputs=gr.Image(type='pil', label='Upload Image'), 
    outputs=[gr.Textbox(label='Word Map Caption'),gr.Textbox(label='Caption in Words'), gr.Image(label='Training Graph Plot' ), gr.Image(label='Attention Image')], 
    title='Image Caption Generator', 
    description='Upload an image to generate a caption.'
)

iface.launch(share=True)
