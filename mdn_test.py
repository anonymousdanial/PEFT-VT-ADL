import torch
import mdn1

# Simulate a sample transformer output (batch_size=2, num_vectors=1, features=512)
sample_output = torch.rand(2, 1, 512)


# Instantiate the MDN
gmm = mdn1.MDN()

# Pass the transformer output through the MDN
pi, mu, sigma_sq = gmm(sample_output)


print('Mixture weights (pi) shape:', pi.shape)
print('Mixture weights (pi) values:')
print(pi)
print('Means (mu) shape:', mu.shape)
print('Variances (sigma_sq) shape:', sigma_sq.shape)

# Optionally, compute the MDN loss for demonstration
loss = mdn1.mdn_loss_function(sample_output, mu, sigma_sq, pi, test=True)
print('MDN loss output:', loss.shape)