"""
A simple GAN, based on https://towardsdatascience.com/pytorch-and-gans-a-micro-tutorial-804855817a6b
"""

import torch
from torch import nn
import torch.optim as optim


class Generator(nn.Module):
    def __init__(self, latent_dim, output_activation=None):
        """A generator for mapping a latent space to a sample space.
        Args:
            latent_dim (int): latent dimension ("noise vector")
            layers (List[int]): A list of layer widths including output width
            output_activation: torch activation function or None
        """
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 64) # Dense layer with input width latent_dim, output width 64
        self.leaky_relu = nn.LeakyReLU() # LeakyReLU activation
        self.linear2 = nn.Linear(64, 32) # Linear layer with input width 64 and output width 32
        self.linear3 = nn.Linear(32, 1) # Linear layer with input width 32 and output width 1
        self.output_activation = output_activation # Specified output activation

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples.

        This defines the structure of the network. In PyTorch, define-by-run framework is used,
        so the computational graph is built automatically as simple computations are chained
        together.
        """
        intermediate = self.linear1(input_tensor) # Step through the generator's modules
        intermediate = self.leaky_relu(intermediate) # and apply them to the output of the previous
        intermediate = self.linear2(intermediate) # to eventually return the final output
        intermediate = self.leaky_relu(intermediate) # This forward method is what's called to calculate the output
        intermediate = self.linear3(intermediate)
        if self.output_activation is not None:
            intermediate = self.output_activation(intermediate)
        return intermediate

    # Note! No backward. PyToch uses autograd for automatic differnetiation,
    # so PyTorch automatically keeps track of the computational graph and doesn't
    # need to be told how to backpropagate the gradients


class Discriminator(nn.Module):
    def __init__(self, input_dim, layers):
        """A discriminator for discerning real from generated samples.
        params:
            input_dim (int): width of the input
            layers (List[int]): A list of layer widths including output width
        Output activation is Sigmoid.
        """
        super(Discriminator, self).__init__()
        self.input_dim = input_dim # save input dim as object variable
        self._init_layers(layers) # calls initi layers

    def _init_layers(self, layers):
        """Initialize the layers and store as self.module_list.

        Kept separate for reasons of cleanliness: separate the instantiation
        and the model building.
        """
        self.module_list = nn.ModuleList() # Initialize the module list so PyTorch recognizes it
        last_layer = self.input_dim # with the last layer as input dimension size
        for index, width in enumerate(layers): # Chain together linear modules
            self.module_list.append(nn.Linear(last_layer, width))
            last_layer = width
            if index + 1 != len(layers):
                self.module_list.append(nn.LeakyReLU()) # as well as Leaky ReLU activations after each layer
        else:
            self.module_list.append(nn.Sigmoid()) # and a sigmoid activation after the final layers

    def forward(self, input_tensor):
        """Forward pass; map samples to confidence they are real [0, 1].

        Works the same way as the generator forward, but we simply iterate over the layer
        list since we've been more clever about how to put that together.
        This applies each module in turn.
        """
        intermediate = input_tensor
        for layer in self.module_list:
            intermediate = layer(intermediate)
        return intermediate



class VanillaGAN():
    def __init__(self, generator, discriminator, noise_fn, data_fn,
                 batch_size=32, device='cpu', lr_d=1e-3, lr_g=2e-4):
        """A GAN class for holding and training a generator and discriminator
        Args:
            generator: a Ganerator network
            discriminator: A Discriminator network
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
            data_fn: function f(num: int) -> pytorch tensor, (real samples)
            batch_size: training batch size
            device: cpu or CUDA
            lr_d: learning rate for the discriminator
            lr_g: learning rate for the generator
        """
        self.generator = generator # Generator object
        self.generator = self.generator.to(device) # Specifies if it'll be a GPU or CPU
        self.discriminator = discriminator # Discriminator object
        self.discriminator = self.discriminator.to(device) # GPU or CPU
        self.noise_fn = noise_fn # Base noise function for sampling initial latent vectors for mapping to sample space
        self.data_fn = data_fn # Function the generator is tasked to learn
        self.batch_size = batch_size # Training batch size
        self.device = device #GPU or CPU
        self.criterion = nn.BCELoss() #Binary Cross Entropy Loss: https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
        self.optim_d = optim.Adam(discriminator.parameters(), # Discriminator optimizer
                                  lr=lr_d, betas=(0.5, 0.999)) # this manages updates to neural network parameters via inheritance from PyTorch.
                                                                # Also pass learning rate and beta parameters that are known to work well
        self.optim_g = optim.Adam(generator.parameters(), # Generator optimizer
                                  lr=lr_g, betas=(0.5, 0.999)) # works the same way, but we give it a slower learning rate
        self.target_ones = torch.ones((batch_size, 1)).to(device) # Labels for training
        self.target_zeros = torch.zeros((batch_size, 1)).to(device)

    def generate_samples(self, latent_vec=None, num=None):
        """Sample from the generator. Helper function. Generates batch size samples by default.
        Args:
            latent_vec: A pytorch latent vector or None
            num: The number of samples to generate if latent_vec is None
        If latent_vec and num are None then us self.batch_size random latent
        vectors.
        """
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        with torch.no_grad():
            samples = self.generator(latent_vec)
        return samples

    def train_step_generator(self):
        """Train the generator one step and return the loss as float.
        This is the heart of the algorithm.
        """
        self.generator.zero_grad() # Clear gradients. PyTorch helps keep track of these, but we want to clear before optimizing

        latent_vec = self.noise_fn(self.batch_size) # Sample from the noise function
        generated = self.generator(latent_vec) # Feed noise samples into the generator and get output
        classifications = self.discriminator(generated) # Feed output samples and get confidence that samples are real or not
        loss = self.criterion(classifications, self.target_ones) # Calculate loss for the generator, Binary Cross Entropy.
            # This is a PyTorch tensor, so it's still connected to the full computational graph
        loss.backward() # This is where the magic happens! The method calculates gradient d_loss/d_x for every parameter
            # in the computational graph automatically since PyTorch manages that graph.
        self.optim_g.step() # Nudge parameters down the gradient via the optimizer
        return loss.item() # return the loss and store for later visualization. Make sure ewe return this as float so it doesn't hang on to the whole
            # computational graph, just the float value we want for later.

    def train_step_discriminator(self):
        """Train the discriminator one step and return the losses."""
        self.discriminator.zero_grad() # Same as above

        # real samples
        real_samples = self.data_fn(self.batch_size) # Get some real samples
        pred_real = self.discriminator(real_samples) # Get the discriminator confidence that they're real
        loss_real = self.criterion(pred_real, self.target_ones) # Calculate the loss function

        # generated samples
        latent_vec = self.noise_fn(self.batch_size) # repeat this process with generated samples
        with torch.no_grad(): # we don't care so much about gradients in the generator here because we're training the discriminator
            fake_samples = self.generator(latent_vec) # This context manager detaches the genetaor from its computational graph and saves computing overhead.
        pred_fake = self.discriminator(fake_samples)
        loss_fake = self.criterion(pred_fake, self.target_zeros)

        # combine
        loss = (loss_real + loss_fake) / 2 # average the graphs for the loss using simple python arithmetic. PyTorch is great.
        loss.backward()  # Calculate gradients
        self.optim_d.step() # Nudge along the slope
        return loss_real.item(), loss_fake.item() # Return losses

    def train_step(self): # Just iterates one step through
        """Train both networks and return the losses."""
        loss_d = self.train_step_discriminator()
        loss_g = self.train_step_generator()
        return loss_g, loss_d


def main():
    from time import time # Time it, so we have an idea of how long it takes. Good practice.
    epochs = 600 # How many epochs with a fixed number of batches
    batches = 10 # How many batches per epoch
    generator = Generator(1) # Instantiate generator
    discriminator = Discriminator(1, [64, 32, 1]) # Instantiate discriminator with layer width
    noise_fn = lambda x: torch.rand((x, 1), device='cpu') # specify noise func
    data_fn = lambda x: torch.randn((x, 1), device='cpu') # specify data func
    gan = VanillaGAN(generator, discriminator, noise_fn, data_fn, device='cpu') # Instantiate VanillaGAN
    loss_g, loss_d_real, loss_d_fake = [], [], [] # Instantiate lists
    start = time() # start timing
    for epoch in range(epochs): # start training
        loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0 # Initialize loss at 0
        for batch in range(batches): # update loss values
            lg_, (ldr_, ldf_) = gan.train_step()
            loss_g_running += lg_
            loss_d_real_running += ldr_
            loss_d_fake_running += ldf_
        loss_g.append(loss_g_running / batches) # take mean loss over batches
        loss_d_real.append(loss_d_real_running / batches)
        loss_d_fake.append(loss_d_fake_running / batches)
        print(f"Epoch {epoch+1}/{epochs} ({int(time() - start)}s):" # output training information
              f" G={loss_g[-1]:.3f},"
              f" Dr={loss_d_real[-1]:.3f},"
              f" Df={loss_d_fake[-1]:.3f}")

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(1)

    plt.plot(np.arange(0,epochs),loss_g,label='G Loss')
    plt.plot(np.arange(0,epochs),loss_d_real,label='D Loss (real)')
    plt.plot(np.arange(0,epochs),loss_d_fake,label='D Loss (fake)')

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    #plt.show()

    plt.figure(2)

    final_samples = np.array(gan.generate_samples(num=1000)).flatten()

    plt.hist(final_samples,bins=100,density=True)

    plt.xlabel("Generated sample")
    plt.ylabel("Density")

    plt.show()

if __name__ == "__main__":
    main()
