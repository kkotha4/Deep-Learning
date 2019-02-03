import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

loss_1=torch.nn.MSELoss()
def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    logits_real_target=torch.ones(logits_real.size(0),1).cuda()
    logits_fake_target=torch.zeros(logits_fake.size(0),1).cuda()
    
    real_loss=bce_loss(logits_real.reshape(-1,1),logits_real_target)

    fake_loss=bce_loss(logits_fake.reshape(-1,1),logits_fake_target)
    loss=real_loss+fake_loss
    
    
    ##########       END      ##########
    
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    
   
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    logits_real_target=torch.ones(logits_fake.size(0),1).cuda()
    loss=bce_loss(logits_fake.reshape(-1,1),logits_real_target)
   
    
    
    ##########       END      ##########
    
    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    logits_real_target=torch.ones(scores_real.size(0),1).cuda()
    logits_fake_target=torch.zeros(scores_fake.size(0),1).cuda()
    #print(scores_real.reshape(-1,1).size())
    #print(logits_real_target.size())
    
    real_loss=loss_1(scores_real.reshape(-1,1),logits_real_target)
    fake_loss=loss_1(scores_fake.reshape(-1,1),logits_fake_target)
    loss=(real_loss+fake_loss)/2
    
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    
    
    ##########       END      ##########
    
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    logits_real_target=torch.ones(scores_fake.size(0),1).cuda()
    loss=loss_1(scores_fake.reshape(-1,1),logits_real_target)
    
    
    ##########       END      ##########
    
    return loss
