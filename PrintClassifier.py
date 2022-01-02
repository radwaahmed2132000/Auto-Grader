import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils
import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784 # 28x28
hidden_size = 200
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001
saved = True

lexicon = dict([(0,'0'),(1,'1'), (2,'2'),(3,'3'),(4,'4'),(5,'5'),(6,'6'),(7,'7'),(8,'8'),(9,'9')])

def prepare_dataset():

    printed_dataset = torchvision.datasets.ImageFolder(root='./PrintedDataset', transform=transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)]))

    train_size = int(0.8 * len(printed_dataset))
    test_size = len(printed_dataset) - train_size
    train_dataset, test_dataset = utils.random_split(printed_dataset, [train_size, test_size])

    # Data loader: now they are converted to batches of [100, 1, 28, 28]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    examples = iter(test_loader)
    example_data, example_targets = examples.next()     #hands on the first batch

    for i in range(6):
        ax = plt.subplot(2,3,i+1)
        ax.title.set_text(example_targets[i+6])
        plt.imshow(example_data[i+6].squeeze(), cmap='gray')  #first 6 images in the first batch. Squeeze so 1x28x28 -> 28x28
    #plt.show()
    return train_loader, test_loader


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.netLayers= nn.Sequential(
        nn.Linear(input_size, hidden_size), nn.ReLU(),                          
        nn.Linear(hidden_size, num_classes) 
        )
        self.input_size = input_size
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)  
       
    def forward(self, x):
        logits = self.netLayers(x)
        return logits

    def train(self, num_epochs, train_loader):
        num_batches = len(train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(tqdm(train_loader)):  #each batch is a tuple of images and their corresponding labels.
                
                images = nn.Flatten()(images).to(device)        # from [100, 1, 28, 28] to [100, 784]
                labels = labels.to(device)                      #[100]
                
                # Forward pass
                logits = self(images)                         #[100, 10] 
                loss = self.criterion(logits, labels)          #Free Softmax inside.
                
                # Backward and optimize
                self.optimizer.zero_grad()                     #clear the gradients for all network parameters (e.g. due to a previous batch)
                loss.backward()                                #accumulate all the gradients due to the current batch
                self.optimizer.step()                          #update the network's weights and biases
                
        
    
    def test(self, test_loader):
        with torch.no_grad():
            n_correct = 0
            for images, labels in test_loader:
                images = images.reshape(-1, 784).to(device)     #just like we used flatten above. 
                labels = labels.to(device)
                logits = self(images)
                predicted = logits.argmax(1)                    #the highest logit is also the highest softmax probability. This has shape (100,)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / (len(test_loader) * 100)
            print(f'Accuracy: {acc} %')
    
def predict_print(characters, saved=True):
    train_loader, test_loader = prepare_dataset()
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)      #so it's done on the GPU if available.

    # Load or Train the model
    if saved:
        model.load_state_dict(torch.load('./Intelligence/PrintIntelligence.pth'))
    else:
        # Train the model
        model.train(num_epochs, train_loader)
        # Test the model
        model.test(test_loader)
        # Save the model
        torch.save(model.state_dict(), './Intelligence/PrintIntelligence.pth')

    magic_word = []

    for char in characters:
        char = torch.from_numpy((char/255)).reshape(-1, 784).float()
        prediction = model(char).argmax()
        magic_word.append(lexicon[prediction.item()])

    magic_word = ''.join(magic_word)
    #print(magic_word)
    return magic_word