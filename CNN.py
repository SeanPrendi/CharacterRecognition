import torch
import torch.nn.functional
import torchvision.datasets as dataset
import torch.optim as optim
from torchvision import transforms

# Optical Character Recognition
class OCR(object):
    def __init__(self,
                 use_cuda=torch.cuda.is_available(), # use GPU or not
                 batch_size=25, # Batch size
                 num_epochs=10, # Number of Epochs
                 lr=0.001, # Learning Rate
                 momentum=0.9 # Optimization momentum
                 ):
        self.use_cuda = use_cuda
        if use_cuda:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.momentum=momentum

    def fit(self):
        # Training Data
        cvt_to_tensor = transforms.Compose([transforms.ToTensor()])
        mnist_train = dataset.MNIST(root="data", download=True, train=True, transform=cvt_to_tensor)
        dataloader_train = torch.utils.data.DataLoader(mnist_train, batch_size=25, num_workers=0)
        self.iterator_train = dataloader_train

        # Testing Data
        mnist_test = dataset.MNIST(root="data", download=True, train=False, transform=cvt_to_tensor)
        dataloader_test = torch.utils.data.DataLoader(mnist_test, batch_size=self.batch_size, num_workers=0)
        self.iterator_test = dataloader_test

        # Possible labels for data
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Simple CNN
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3) # Convolutional Layer
                self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2) # Max Pooling Layer
                self.linear = torch.nn.Linear(507, 10) # Fully Connected Layer; Reduces all outputs into a flat layer then rehsapes into appropriate size to predict classes (10 classes, 10 outputs)

            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                x = torch.nn.functional.relu(x) # ReLU Activation function
                x = torch.flatten(x,start_dim=1) # Flattens layers without losing batches
                x = self.linear(x)
                return torch.nn.functional.softmax(x, dim=0) # Softmax activation function for determining probabilities of each class

        model = Model()
        model = model.to(self.device) # Send model (and parameters) to GPU
        self.model = model

        # Loss function
        criterion = torch.nn.CrossEntropyLoss() # Cross entropy loss function, good for >2 categories, punishes high confidence incorrect answers harshly

        # Optimization function
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum) # Stochastic Gradient Descent optimization function
        # - Model.parameters() takes the parameters that need to be trained from the model object created above
        # - lr == learning rate, the size of the steps in each training steps
        # - momentum is the amount that each step is influence by previous steps, reduces dangers of small local minima


        for epoch in range(self.num_epochs): # Loop over data set 10 times
            count = 1
            current_loss = 0.0

            for data in self.iterator_train: # Loop over all batches in the training data
                batch, correct_labels = data # Batch: batch of images; correct_labels: proper classes for that batch
                batch, correct_labels = batch.to(self.device), correct_labels.to(self.device) # Send variables to GPU
                optimizer.zero_grad() # Have to zero out gradients in optimizer as Pytorch optimization is cumulative of gradients
                    # For more on zeroing: https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/13

                output = self.model(batch) # Gets model's probabiltiies of each label for each image in batch
                loss = criterion(output, correct_labels) # Determine how far off predictions were from correct labels
                loss.backward() # Computes dloss/dx for every parameter x (back propogate)
                optimizer.step() # Update all x terms based on dloss/dx (optimize)

                current_loss += loss.item() # Increase value of loss

                count += 1
                if count % 200 == 0:  # print loss every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, count, current_loss / 200))
                    current_loss = 0.0

                self.save("models/char_recognition_ver" + str(epoch+1)) # Save model after every epoch

        print('Finished Training')

    def test(self):
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # Testing on test data
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad(): # Run tests with no back propogation
            for data in self.iterator_test:
                batch, correct_labels = data
                batch, correct_labels = batch.to(self.device), correct_labels.to(self.device)
                outputs = self.model(batch) # Make predictions
                _, predicted = torch.max(outputs, dim=1) # Gets class with highest probability for each image
                c = (predicted == correct_labels).squeeze() # Find the predictions that match the correct labels
                for i in range(4):
                    label = correct_labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        # Printing accuracy
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))

    # Save model
    def save(self, out_path):
        if self.model is not None and out_path is not None:
            torch.save(self.model.state_dict(), out_path)

        # Load model

    def load(self, in_path):
        if in_path is not None:
            self.model.load_state_dict(torch.load(in_path))

def main():
    ocr = OCR()
    ocr.fit()
    ocr.test()

main()
